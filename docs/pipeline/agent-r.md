# Agent-R: MCTS revision trajectories (weight fix)

Agent-R (Yuan et al., arxiv 2501.11425, code `ByteDance-Seed/Agent-R`) trains an agent to
notice its own mistake in the middle of a run and recover, instead of only imitating clean
runs. It is a second source of weight-fix training data next to the teacher runs
(`docs/pipeline/training.md`), built from the student's own rollouts.

Nothing here spends credits unless you run a command without `--dry-run`.

## The three steps

| Step | Script | Paper |
|---|---|---|
| 1. MCTS search over the actor's own tau2 runs | `scripts/agent_r_search.py` | section 3.1, Eq. 5 |
| 2. Pairs, model-guided critique (find t'), splicing | `scripts/agent_r_revise.py` | Eq. 3, Eq. 4, section 3.2 |
| 3. SFT data with loss masks, eta mix, AgentTune LoRA | `scripts/build_sft.py --agent-r`, `scripts/train_lora.py` | Eq. 6 |

All the logic that does not need tau2 or a model is in `src/shipwright/heal/agent_r.py`
(tests: `tests/heal/test_agent_r.py`).

### 1. MCTS search

- A **node** is a conversation prefix that ends where the agent acts next. An **edge** is one
  agent step: one assistant message plus what came back (tool results, the user's reply).
  Steps are cut by `agent_r.split_steps`; two steps are the same edge when their roles,
  text and tool calls match (ids, timestamps and usage are ignored).
- **Branching** uses tau2 itself: a rollout from a node is a normal tau2 run of the task
  with `Task.initial_state.message_history` set to the node's prefix. tau2 replays the
  prefix's tool calls against a fresh DB (strict: each replayed result must match) and the
  agent, the user simulator and the evaluator continue from there. The final reward is
  tau2's own reward for the whole conversation.
- **Selection**: UCT, `Q + c * sqrt(ln N_parent / N)`, c = 0.25, descending to the first
  node with fewer than `--children` rollouts started from it.
- **Expansion and simulation**: `--children` (4) rollouts at once from the selected node,
  temperature 1. Each rollout runs to the end of the conversation; its whole path is added
  to the tree, so each rollout is one leaf with a final reward.
- **Backpropagation**: the reward is added along the path (visits + 1, value + reward).
- **Limits**: `--simulations` rollouts per task (default 24; the paper used 200 to 300),
  `--max-depth` 20 agent steps (deeper nodes are not expanded).

### 2. Pairs, critique, revision

- **Pairs (Eq. 4)**: a good leaf has reward >= alpha; a bad leaf is at least beta (0.2) below
  it. Each bad leaf gets the good leaf with the longest shared path (the "adjacent correct
  path"); ties go to the shorter good run. They diverge at step t.
  `--iteration 1|2|3` sets alpha to 0.5, 0.7, 1.0. tau2 retail rewards are 0 or 1, so in
  practice every pair is a failed run and a passed run.
- **Transition point t' (`--transition`)**:
  - `model` (default, Agent-R): the actor judges each bad step from t on with the paper's
    verifier prompt (`agent_r.CRITIC_SYSTEM`: Good / Bad / Uncertain, reasons first, then a
    `Judgment:` line). t' is the first Bad step. If no step is judged Bad, t' is the last
    step (a direct revision, as in the paper's code), and the record says
    `critic_found_bad: false`.
  - `direct`: t' is the last bad step (the paper's Direct-Revision baseline).
  - `divergence`: t' = t, no model calls (our cheap ablation, not in the paper).
  The critic is the actor itself (Tiny Aya on the HF router, or the round's LoRA on
  `--critic-api-base`), as the paper does ("within its current capability"). It sees the
  policy, the conversation so far, the step and its result; the prompt stays near 5k tokens
  so it fits Tiny Aya's 8k context.
- **Revision run (Eq. 3)**: shared prefix + bad steps t..t' + revision thought + good steps
  t..end. A user stop message (`###STOP###` and the other stop tokens) in the kept bad
  steps is removed. One of ten revision thoughts (`agent_r.REVISION_THOUGHTS`) is drawn per
  pair.

### 3. Training data

- Every assistant turn of the shared prefix and the bad steps is **context only**; the
  revision thought and the good suffix are **targets** (Eq. 6). `build_sft.py` writes a
  `loss_mask` (one bool per message, True = masked, AgentTune's convention) on these
  records; `train_lora.py` passes it to AgentTune's `EventLog` rows, so the bad steps are
  never trained, also inside windows of long runs.
- The good leaves of each task are added as normal examples (source `good`), as Eq. 6 trains
  on good trajectories too.
- **Eta mix**: Agent-R mixes agent data with general chat data (ShareGPT), agent share
  eta = 0.2, against forgetting. `--general FILE --eta 0.2` adds that many general examples
  (jsonl, `messages` or ShareGPT `conversations`); `sft_stats.json` `eta_mix` reports the
  share reached if the file is too small. Without `--general` there is no general data.

## Format choice: inline revision thought

The paper's signal is an extra turn pair (`Assistant: <reflection>` / `Human: OK.`). In
tau2 the user is a simulator that would have to answer it at deployment, so by default the
thought goes in front of the first good action in the same assistant turn:

````
I realize my approach was flawed. I need to revise it.
```json
{"name": "get_order_details", "arguments": {"order_id": "#W2378156"}}
```
````

The prompted tool parser already accepts text before the JSON block, so the student can
emit this live with no protocol change. `--revision-format turn` gives the paper's
literal form.

## Commands

```sh
# Offline: the whole path with a noisy-oracle fake actor (no network)
make agent-r-dry RUN=runs/dry

# 1. Search (spends credits: one rollout = one tau2 conversation with Tiny Aya + the user
#    simulator, so tasks x --simulations conversations; a human must ask for this run)
.venv/bin/python scripts/agent_r_search.py --from-run runs/baseline-<ts> \
    --user-model vertex/openai/gpt-oss-120b-maas --simulations 24

# 2. Revisions (critic calls: at most one per bad step after the divergence point)
.venv/bin/python scripts/agent_r_revise.py runs/agent-r-search-<ts> --iteration 1

# 2b. Check the revisions (offline): tau2's evaluator re-grades each spliced run (strict replay
#     catches spliced tool results that disagree with the real DB), plus format, loss masks and
#     the 8192-token limit. Writes check.jsonl and check_summary.json; exit 1 if none is valid.
.venv/bin/python scripts/agent_r_check.py runs/agent-r-revise-<ts>

# 3. SFT data, then the usual push and training job (docs/pipeline/training.md)
.venv/bin/python scripts/build_sft.py --agent-r runs/agent-r-revise-<ts> \
    --teacher runs/teacher-<ts> --general data/sharegpt_sample.jsonl --eta 0.2 \
    --output-dir runs/sft-agent-r-1
```

## Outputs

- Search dir: `traces.jsonl` (one line per rollout, with `mcts.path`), `trees/<task>.json`
  (per node: depth, visits, Q, expansions, reward, one line for its action),
  `search_summary.json` (rollouts, passed leaves, possible pairs per task), plus the usual
  `summary.json`, `prompts.json`, `run_config.json`, logs and spans.
- Revise dir: `revisions.jsonl`, `good.jsonl`, `critic.jsonl` (every judgment with the raw
  reply), `revise_summary.json` (critic verdict counts, how often a Bad step was found,
  mean bad steps kept).

## Iterations

The paper runs three rounds; each round searches again with the newest model and raises
alpha. Round 1 uses base Tiny Aya on the HF router. From round 2 the actor (and the critic)
is the last round's LoRA: pass `--agent-api-base` (search) and keep the default critic
endpoint (revise reads it from the search run). Serving the adapter for a long search is
not automated yet; `scripts/gate_job.py` shows how vLLM serves base + LoRA on an HF Job.

## Known limits

- The good suffix's tool results were produced on the good run's DB state. When a kept bad
  step changed the DB (a write that succeeded), the spliced observations can disagree with
  the real state. The paper accepts the same mismatch. The critic usually stops at the
  first wrong step, which keeps this rare; `t_prime - t` in `revisions.jsonl` shows how many
  bad steps were kept.
- Searching is costly compared with plain baseline runs (the paper's 200 to 300 rollouts per
  task are far beyond our budget); the default 24 is a compromise. `search_summary.json`
  reports how many tasks produced at least one pair.
- With 0/1 rewards, a task the actor never solves in its rollouts gives no pairs. Those
  tasks need the teacher path instead.
