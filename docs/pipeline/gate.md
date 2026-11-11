# Deploy gate (step 05 GATE)

The gate compares a **candidate** system with the **current** system on the held-out split
and decides `promote` or `reject`. It never promotes by default: too few valid runs, a
missing metric or a tie on the target label means reject.

- Config: `configs/gate.yaml` (status `proposed`: the team must freeze it before round 1).
- Systems: `configs/systems/*.yaml` (base model, optional adapter, optional tool patches,
  target failure labels). Format in `src/shipwright/heal/systems.py`.
- Decision logic: `src/shipwright/heal/gate.py` (pure). Metrics: `src/shipwright/heal/eval_metrics.py`
  (AuditKIT). BFCL: `src/shipwright/heal/bfcl_eval.py` (AuditKIT).
- Runner: `scripts/gate_eval.py`. HF Jobs launcher: `scripts/gate_job.py`.

## Systems under test

| Route | Candidate config |
|---|---|
| tool fix | the current system plus `tool_patches` (ledger ids, or `ids: applied`) |
| weight fix | `adapter: {repo, revision}` (e.g. `mitushigoyal5/shipwright-lora@round-1`) plus the current patches |

Each candidate names its `targets`: the failure labels (from `src/shipwright/heal/classify.py`)
it should lower. `--target-label` overrides them. `ids: applied` is resolved when the gate
runs; the exact ids go into `gate_inputs.json` and `metrics.jsonl`.

## Checks (all must pass)

| Check | Rule (proposed values) |
|---|---|
| comparable | both ran the 20 held-out `retail` `test` tasks (`configs/splits.yaml`), the same task ids, k = 3 |
| valid_runs | each system: >= 54 valid (non infra-error) runs, >= 90% of tasks x k, >= 18 tasks with all 3 trials valid |
| pass_1 | candidate pass^1 >= current pass^1 (a tie passes) |
| pass_k | candidate pass^3 >= current pass^3 (a tie passes) |
| target | target-label failure rate drops by > 0 and >= 5 points (3 of 60 runs); a tie rejects |
| non_target | other failure labels rise by <= 5 points |
| bfcl | BFCL v3 accuracy (simple, multiple, irrelevance; 100 each) drops by <= 2 points; missing BFCL rejects |
| agenttune_gate | AgentTune `DeploymentGate.evaluate_decision` approves, on >= 54 paired held-out runs |

Rates use valid runs as the denominator. Infra-error runs are excluded everywhere and
counted in `num_infra_errors`. Labels come from `scripts/classify.py`; `classify.classifier_model:
null` means AgentTune's detector and our rule labels only (deterministic, free).

## AgentTune DeploymentGate and AuditKIT together

Promote = AgentTune's `DeploymentGate` approves AND every AuditKIT threshold above passes.

- AgentTune: `agenttune.decide.closed_loop.deployment_gate.DeploymentGate.evaluate_decision`
  (`agenttune_bridge.agenttune_gate_check`). The test set is the paired held-out runs: per task,
  the trials that both systems decided (from AuditKIT's per-task counts). Each system's verdict
  per case is PASS or FAIL, so AgentTune compares the two pass rates and blocks a drop of more
  than `agenttune_gate.task_regression_tol` (0.0).
- AgentTune approves an empty test set (both scores fall back to 0). Our check fails when there
  are fewer paired runs than `min_test_samples`, which is the `valid_runs` floor (54).
- AuditKIT: `auditkit.agent_eval.AgentEvalRunner` (recorded mode) with
  `FinalStateAssertion` for pass^1 and pass^k, `ToolCallF1` and the other diagnostics, and
  BFCL v3 scoring (`src/shipwright/heal/bfcl_eval.py`).
- `gate_decision.json` has `verdicts.agenttune` (`approved`, `reason`, `num_cases`,
  `min_test_samples`) and `verdicts.auditkit_thresholds` (`passed`, `failed_checks`). The
  `metrics.jsonl` row has the same `verdicts`.
- Trajectory scores are not passed to `evaluate_decision`: they need AgentTune's
  `AgenticEvalResult` (its own LLM-judged metrics), and our trajectory checks are AuditKIT's.

## Metrics (AuditKIT)

- Each trace becomes an AuditKIT `AgentEpisode` (`episode_from_openai_messages`, after a small
  adapter from tau2's message format). The oracle is `FinalStateAssertion("tau2_pass", True)`
  on tau2's verified result; infra-error runs have no state, so they score `unknown`.
- pass^1 and pass^k: `AgentEvalRunner` in recorded mode, `trials=k`, `reliability_k=k`,
  `reset_confirmed=True`. pass^k per task is AuditKIT's `all_k` (`pass_hat_k`), only for
  tasks with k decided trials.
- Diagnostics: `tool_call_f1` (`ToolCallF1` against tau2's expected actions),
  `tool_call_validity` (tool schemas from `prompts.json`), `redundant_tool_calls`.
- Ours: per-label failure rate, tool-call error rate (malformed, unknown tool or tool error),
  prompted-JSON parse failures, average agent steps.
- BFCL: `ak.load_bfcl` + `ToolCallF1(arg_match=ak.bfcl_arg_match)`. Tiny Aya has no native
  tool calls, so `bfcl_eval.py` sends each question in our prompted format and parses the
  ```json block (one call per reply, so the parallel categories are not used).

## Serving the candidate (HF Job)

The HF router serves only the base model, so the job serves both systems itself with vLLM
(`serve.vllm` in the config, run in its own `uv run --with vllm` env):

- `serve.mode: lora`: one server, `vllm serve CohereLabs/tiny-aya-global --enable-lora
  --lora-modules tiny-aya-<system>=<adapter dir>`. vLLM maps `Cohere2ForCausalLM` to
  `CohereForCausalLM`, which implements `SupportsLoRA` (checked in the vLLM v0.32.0 source,
  2026-11-08; not yet run).
- If that server does not start, the job falls back to `merged`: peft `merge_and_unload`,
  then one server per system. `--serve-mode merged` forces it; `--no-merge-fallback` stops it.
- The agent goes to `http://localhost:8000/v1` (`baseline.py --agent-api-base`); the user
simulator goes to the HF router, or to Vertex AI for a `vertex/` id. Served names contain `tiny-aya`, so the prompted-tools
  layer applies.

The repo is private, so `gate_job.py --launch` uploads `git archive HEAD` (src, scripts,
configs, patches, pyproject.toml) to `<runs-repo>/gate/round-N/code/`. The job unpacks it,
runs `scripts/setup_tau2.sh`, and re-runs `gate_eval.py` in that venv. Commit first:
uncommitted changes are not in the bundle. Launched from the Cloud Run job (where the HF token is),
the image has no `.git`, so the bundle is the same paths as built into the image, named after its `SHIPWRIGHT_GIT_SHA`.

## Commands

```sh
# 0. Offline: tests, and the whole pipeline with the fake LLM (expected decision: reject)
.venv/bin/python -m pytest -q
.venv/bin/python scripts/gate_eval.py --round 0 --local-dry-run \
    --current configs/systems/base.yaml --candidate configs/systems/r1-tool.yaml --num-tasks 2

# 1. Offline: decide from two run dirs that already have failures.jsonl (exit 0 promote, 1 reject)
.venv/bin/python scripts/gate_eval.py --round 1 --current-run runs/<current> \
    --candidate-run runs/<candidate> --target-label tool_example_copied

# 2. Plan a gate job: prints flavor, timeout, max cost; uploads and launches nothing
.venv/bin/python scripts/gate_job.py --round 1 --current configs/systems/base.yaml \
    --candidate configs/systems/r1-tool.yaml --user-model <USER_MODEL>

# 3. Launch it (paid: GPU + user simulator calls). Add --launch to the command above.
hf jobs logs <job id>      # follow
hf jobs cancel <job id>    # stop early
```

`<USER_MODEL>` has no default. Team choice (2026-11-09): `vertex/openai/gpt-oss-120b-maas`.
An HF router id (e.g. `openai/openai/gpt-oss-120b:cerebras`) also works. An HF Job cannot run
gcloud, so for a `vertex/` user model `gate_job.py --launch` mints a token on your machine and passes
it as the job secret `SHIPWRIGHT_VERTEX_TOKEN`. It lasts about 1 hour with no refresh: keep such a
job under 1 hour, or use an HF router user model for a longer one. Default flavor `a100-large` at $2.50/h with a
2 h timeout: at most $5.00 GPU, plus about $1.20 of user simulator calls on the HF router, or about $0.60 on Vertex (240 tau2 runs).

## Output

Local: `runs/gate-round-N-<mode>-<stamp>/`. Job: `mitushigoyal5/shipwright-runs` under
`gate/round-N/<candidate>/`.

- `<system>/`: the tau2 run dir (`traces.jsonl`, `failures.jsonl`, `summary.json`, ...) and `bfcl.json`
- `metrics_current.json`, `metrics_candidate.json`
- `gate_decision.json`: `decision`, `verdicts` (AgentTune and AuditKIT), `reasons`, `checks`,
  `metrics_current`, `metrics_candidate`, `thresholds`, `warnings` (e.g. config not frozen)
- `gate_inputs.json`: config, systems with resolved patch ids, targets, user model
- `metrics.jsonl`: one row per gate run (round, systems, headline metrics, decision). The job
  also appends the row to `metrics.jsonl` at the root of the dataset.

After the decision: promote keeps the patch status `applied` (or the adapter revision becomes
the current system); reject means `scripts/tool_fix.py --rollback <id>` for the patches, or the
old adapter stays current.
