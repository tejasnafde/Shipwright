# Shipwright: a support agent that finds, sorts and fixes its own failures

**One-line claim.** A 3.35B Tiny Aya support agent on tau2-bench retail, with a self-healing loop that sorts each failure into a tool fix, a weight fix or no fix, and only redeploys when an AuditKIT/AgentTune gate measures an improvement. The best system more than doubles task success over the base model (10% to 23%) and cuts tool-call errors from 60% to 8%.

> DRAFT, 2026-11-14. Cells marked **PENDING** wait for gate 11 and the clean teacher ceiling, which are running now.

## Tracks and problem statements

- **Track 3, Agentic Fine-tuning.**
- **Problem statement 8, Self-healing agents** (primary): "Detect failing runs, classify why they failed, generate corrective training examples, retrain, and gate redeployment on measured accuracy."
- Also measured: **#1 Reliable tool calling** (tool-call error rate, BFCL) and **#3 Agentic distillation** (a DeepSeek teacher's trajectories distilled into Tiny Aya).

## Abstract

Small open models are cheap to run as customer-support agents. In practice they fail in ways that are easy to see but hard to fix by hand: they skip identity checks, invent order ids, change orders without asking, and hand off to a human at the first message. Shipwright closes the loop on those failures automatically.

The agent is **Tiny Aya Global (3.35B)** with a LoRA adapter. It works as a retail support agent in **tau2-bench**: a simulated customer, 16 shop tools, a written company policy, and a score based on whether the shop database ends up right. Every run is logged.

Failures are detected and classified with AgentTune's `FailureDetector` and `FailureClassifier`, plus our own rules. A router then sends each failure down one of three routes:

- **Tool fix:** guards that check each tool call against the policy before it runs, plus rewritten tool descriptions.
- **Weight fix:** a DeepSeek V4.1 Flash teacher replays the failed tasks, and its passing conversations plus corrections of the student's own wrong turns train a new LoRA with AgentTune.
- **No fix:** the fault lies with the environment, not the agent.

A candidate is deployed only if it passes all 8 checks of a frozen gate: AuditKIT pass^1 and pass^k on 20 held-out tasks × 3 trials, target and non-target failure rates, BFCL, and AgentTune's `DeploymentGate`. A further 20 tasks stay hidden for the final report.

**Headline (gate 8, same 60 held-out conversations):** task success **10.0% to 23.3%** (paired McNemar 8 vs 0, p = 0.008), tool-call errors **60% to 8%**, invented ids 70% to 10% of runs, BFCL 43 to 71. The gate still refused this candidate on one check: longer conversations overflow Tiny Aya's 8,192-token window. That refusal is the system working as designed.

The loop also found faults in the environment. The gpt-oss customer simulator sometimes wrote past its own turn, or sent the end-of-conversation signal while its own reasoning said it meant to answer. That hit the teacher and the polite, question-asking student far more than the base model. Both are now fixed, and the numbers re-measured with the fixes are **PENDING**.

## Problem and approach

**Use case.** An online shop's first-line support: look up orders, cancel or modify pending orders, return or exchange delivered ones, and update addresses, all under a written policy. The policy requires authenticating first, taking ids only from the customer or the tools, getting an explicit "yes" before any change, and transferring to a human only when no tool can help. A 3B model that does this reliably runs on a single small GPU at a fraction of an API model's cost.

**Approach: the self-healing loop.**

```
run (tau2 retail, train split) -> detect -> classify -> route
     tool fix  : guards + tool-description patches (ledger with rollback)
     weight fix: teacher replays failures -> SFT data (+ corrections of the student's own wrong turns) -> LoRA
     no fix    : environment fault, recorded
-> gate (20 held-out tasks x 3, 8 checks) -> promote (ledger) or reject (keep the deployed system)
```

**What we built** on top of AgentTune and AuditKIT:
- the 3-way router;
- runtime tool-call guards: authenticate first, no invented ids, no early transfer, a "yes" before a change, no repeat of a failed call, order status must fit the tool;
- automatic repair of a wrong-kind id lookup;
- per-turn status reminders;
- context compaction for the 8K window;
- a tool-patch ledger with rollback;
- the gate with frozen thresholds and splits;
- an unattended round driver (`scripts/run_round.py all`) with spend caps.

## Hackathon repos used and how they connect

| Repo | Used for |
|---|---|
| **AgentTune** | `FailureDetector` and `FailureClassifier` (detect and classify), `RetrainingTrigger` (decides whether a round trains), `create_distill_trainer` (LoRA SFT on teacher trajectories), `DeploymentGate.evaluate_decision` (one of the 8 gate checks, with an explicit minimum-sample floor) |
| **AuditKIT** | `agent_eval` reliability: pass^1 and pass^k over recorded trajectories (every gate number); the BFCL comparison check in the gate |
| CuratorKIT | Not used: its exporters have no tool-call schema, so trajectories would have to be flattened to text |

How they connect: tau2 traces → AgentTune detect and classify → our router → AgentTune distill trainer → candidate → AuditKIT scores + AgentTune `DeploymentGate` → promote or reject.

## Models and licenses

| Role | Model | License / access |
|---|---|---|
| Student (deployed) | `CohereLabs/tiny-aya-global` + our LoRA | CC-BY-NC-4.0 (non-commercial; we make no commercial claims) |
| Teacher (training data only) | DeepSeek V4.1 Flash via Vertex AI | Used as a supplementary teacher, as the FAQ allows |
| Customer simulator and judge | gpt-oss-120b (HF router / DeepInfra; Vertex AI) | Apache-2.0 |
| Failure classifier | gpt-oss-120b via AgentTune `FailureClassifier` | Apache-2.0 |

## Datasets, benchmarks and licenses

| Data | Source | License | How used |
|---|---|---|---|
| tau2-bench retail (114 tasks) | sierra-research/tau2-bench | MIT | Environment, tasks, reward |
| BFCL v3 (simple, multiple, irrelevance) | Gorilla BFCL | Apache-2.0 | Gate check (first 100 per category); general-replay training examples come only from samples after the first 100 (no overlap, checked by `tests/test_bfcl_replay.py`); BFCL has no official train split |
| Our SFT data | Teacher passes + corrections + BFCL replay | follows sources | `mitushigoyal5/shipwright-sft` (branch per round) |

**Splits.** These are frozen on 2026-11-11 in `configs/splits.yaml`:
- the train pool is tau2's 74 train tasks;
- held-out is 20 of the 40 test tasks, and the gate scores on them;
- hidden is the other 20 test tasks, opened once for the final report.

**Leak check.**
- `scripts/baseline.py` refuses hidden ids, and refuses a test run without explicit ids.
- Training data is built only from train-split runs (`build_sft.py` checks the split).
- Held-out failures are never used as training data.

## Tool/task environment and trajectories (Track 3)

The environment is tau2-bench retail: 16 tools (lookups, writes, `transfer_to_human_agents`), a 1,500-token policy, and a gpt-oss customer with a hidden scenario. A run passes only if the final database matches and the required statements were made.

Tiny Aya has no native tool calling, so tools are described in the prompt and the model answers with one JSON block per call (`src/shipwright/llm/prompted_tools.py`). Training data uses exactly the same format.

Every run's trajectory is saved: messages, the raw model output per turn, parse status, guard rejections, context trimming, and the reminder shown. Traces and gate outputs are in `mitushigoyal5/shipwright-runs`.

## Metrics and why

| Metric | Direction | Why |
|---|---|---|
| **pass^1** (task success, AuditKIT) | higher | The headline: did the customer's problem get solved? It's what PS8 means by "measured accuracy" |
| **pass^k**, k=3 | higher | Reliability: is the task solved every time? |
| Tool-call error rate | lower | Reliable tool calling (PS1) |
| Failure-label rates (target / non-target) | lower | Did the fix remove what it targeted without adding new failure kinds? |
| BFCL (simple, multiple, irrelevance) | higher | General tool-use capability: catches catastrophic forgetting |
| Avg agent steps | lower | Trajectory efficiency and context use |
| **Gap closed** = (student − base) / (teacher − base) | higher | Distillation progress against the teacher run in the identical harness |

## Method and experiments

**Training setup.** LoRA via AgentTune `create_distill_trainer`:
- r 16, alpha 32, dropout 0.05;
- max sequence 8,192, batch 1 × gradient accumulation 8;
- fresh from the base model every round, so earlier lessons are kept by pooling their data rather than by stacking adapters;
- loss only on assistant turns; a correction row trains only the teacher's corrected turn.

The best adapter (round 7) trained on 776 rows: 433 teacher turns, with failed tasks ×3, plus 300 BFCL general-replay rows and 43 corrections of its own wrong turns. Hardware: HF Jobs, L40S, about 40 minutes.

**Rounds.** Ten gated rounds, each refused, each refusal telling us something:

| Gate | Change | pass^1 (base to candidate) | Why refused / what it taught |
|---|---|---|---|
| 3 | First LoRA on teacher passes | 10 to 15% | BFCL fell 43 to 35: the adapter forgot general tool calling; the fix was BFCL replay data |
| 4 | Retrained | 12 to 7% | Worse overall |
| 5 | + 300 general-replay rows | 10 to 13% | BFCL fixed (72); new failure kinds remained |
| 6 | + tool guards | 10 to 13% | Tool errors 58% to 13% |
| 7 | + 43 on-policy corrections | 10 to 13% | 59 "Product not found" errors: item ids sent to the product lookup |
| **8** | Same adapter + id-kind repair + repeat guard (no training) | **10 to 23%** | 7 of 8 checks; only context overflow failed |
| 10 | Fresh adapter, no corrections, half the training, + reminders | 10 to 12% | Lost the corrections: made-up answers back to base level (13 vs 6 runs) |
| 11 | Gate 8's adapter + simulator fixes + context savers | **PENDING** | **PENDING** |

## Results

**Gate 8, base vs candidate, same 20 held-out tasks × 3 trials:**

| Metric | Base Tiny Aya | + LoRA + guards | Change |
|---|---|---|---|
| pass^1 (95% CI) | 10.0% (4.7–20.2) | **23.3%** (14.4–35.4) | +13.3 pts; McNemar 8 vs 0, p = 0.008 |
| pass^k (k=3) | 10.0% | 15.0% | +5.0 pts |
| Tool-call error rate | 60.4% | **7.6%** | −52.8 pts |
| Runs with an invented id | 70% | 10% | −60 pts |
| Premature transfer | 17% | 0% | |
| BFCL | 43.3 | **71.3** | +28.0 |
| Avg agent steps | 10.7 | 14.1 | more lookups (the overflow cause) |

**Teacher vs student (same harness).**

| System | pass^1 |
|---|---|
| Teacher, DeepSeek V4.1 Flash | 31.7% before the simulator stop fix; **PENDING** after it |
| Base Tiny Aya | 10.0% |
| Tiny Aya + LoRA + guards | 23.3% (gate 8); **PENDING** (gate 11) |
| Gap closed | about 62% vs the pre-fix ceiling; **PENDING** |

**ROI.** The teacher costs about $0.01 per conversation through an API. The student is a 3.35B model that runs on one small GPU, or locally.

**Findings worth keeping.**
1. Most of the student's gains came from **runtime guards plus on-policy corrections**, not from more teacher data alone. A fresh adapter without corrections fell back to base-level hallucination (gate 10).
2. **The environment can be the failing component.** The customer simulator's format leaks and unmeant stops cut 15 of the teacher's 41 failures short, and hit the question-asking student far more than the base (19 vs 3 hang-ups). This is the "no fix" route: an environment fault, not the agent's.
3. **Narrow fine-tuning forgets general tool use** (BFCL 43 to 35). Mixing in general function-calling examples restores it (72).

## Limitations

- Tiny Aya's 8,192-token window is the binding constraint: 2,700 tokens go to the fixed policy and tool prompt, and long lookup chains overflow.
- Only 20 held-out tasks × 3 trials, so pass^1 moves by about 3 passes between equal candidates. We report intervals and paired tests.
- Eight gates looked at the same held-out tasks, which means selection on them. The final number should come from the hidden split (**PENDING**: opened once).
- The customer simulator and judge are an LLM. We found and fixed two failure modes, and there may be others.
- No candidate has been promoted yet, so "deployment" is a promotion ledger, not a live endpoint.

## With more credits

- **A condensed policy** (about 1,500 to 500 tokens) with a retrained adapter, to free about 1,000 tokens of context per turn. Expected: context overflow under the gate's allowance, so the first promotion.
- **More on-policy correction rounds** (DAgger-style), and DPO on the 300+ recorded guard rejections (rejected call vs corrected call).
- **A reply-text guard** for invented order ids and amounts in messages to the customer, not only in tool calls.
- **A larger student** (Aya Expanse 8B) and a stronger simulator, to separate model limits from environment noise.

## Artifacts

- **HF models collection:** TODO link (adapters `mitushigoyal5/shipwright-lora`, branch per round; best: `round-7`)
- **HF datasets collection:** TODO link (`mitushigoyal5/shipwright-sft`, `mitushigoyal5/shipwright-runs`: all gate traces and decisions)
- **GitHub:** TODO public link to the repo
- Demo / Space: TODO (`infra/space/`, a read-only control room over recorded runs)

## Documentation and reproduction

- **Repo layout:**
  - `src/shipwright/llm/`: prompted tools, guards, reminders, context, patches;
  - `src/shipwright/heal/`: classify, router, SFT data, gate metrics, recipe, round chain;
  - `scripts/`: baseline, teacher runs, build_sft, train and gate jobs, `run_round.py`, `teacher_student_report.py`;
  - `configs/`: frozen splits and gate, systems, rounds.
- **Setup and runs:** `docs/setup/*.md`, `docs/pipeline/*.md`. One round, unattended: `scripts/run_round.py all --config configs/rounds/<name>.yaml --launch`. Offline: `--dry-run`. Tests: `pytest` (368 pass).
- **Reproducibility:** every run writes `run_config.json` (models, seeds, code sha, observation version) and `repro.json`, and every gate records the sha256 of the frozen configs.
- **Declaration of work done outside the hackathon:** none. All code, data and models were produced between Nov 1 and Nov 15, 2026.

## Teammates

TODO: tag on HF. Tejas Nafde, Ishank, Mitushi Goyal (`mitushigoyal5`).
