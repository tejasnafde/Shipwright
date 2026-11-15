# Running the gate

What the gate decides and why is in `docs/evaluation.md`. This page is how it runs.

## Systems under test

A system is one YAML file in `configs/systems/` (format and validation in `src/shipwright/heal/systems.py`): the base
model (only allowed models load), an optional LoRA adapter (`repo`, `revision`), optional tool patches, the serving
switches (`docs/pipeline/baseline.md`), the route and the failure labels it targets. The deployed system is
`configs/systems/current.yaml`, written from the promotion ledger; never edit it by hand. A candidate made by the loop
is written into its round directory and shipped to the job inside the code bundle (`configs/systems/generated/`).

## How a gate runs

1. **Serve and play (HF Job, GPU).** `scripts/gate_job.py` uploads a code bundle to the runs dataset and starts an HF
   Job that runs `scripts/gate_eval.py --in-job --runs-only`. It starts one vLLM server with the base model and each
   adapter as a LoRA module (`serve.mode: lora`; a merged-weights fallback exists), plays both systems on the 20
   held-out tasks x 3 trials with the HF-router user simulator, scores BFCL, and pushes the run directories.
2. **Classify and decide (Cloud Run).** The loop's gate stage pulls the runs, classifies every failure of both systems
   (`docs/pipeline/classify.md`), computes the AuditKIT metrics and the eight checks, and pushes
   `gate_decision.json`. Classification runs here because a Vertex token cannot be refreshed inside an HF Job.
3. **Promote or refuse.** On approval the promotion ledger on the Hub gets a row and `current.yaml` follows it; a
   refusal writes the failed checks, which the recipe reads to choose the next remedy (`docs/pipeline/loop.md`).

Promote requires AgentTune's `DeploymentGate.evaluate_decision` to approve and every AuditKIT threshold to pass. AgentTune
approves an empty test set by default; our check requires at least 54 paired runs.

## Metrics

- **Task success:** each trace becomes an AuditKIT `AgentEpisode`; `AgentEvalRunner` in recorded mode with
  `FinalStateAssertion` on tau2's verified result gives pass^1 and pass^k (k = 3, only on tasks with all 3 trials
  decided). Infra-error runs score `unknown` and are excluded.
- **Diagnostics:** AuditKIT `ToolCallF1` against tau2's expert actions, tool-call validity, redundant calls; our
  per-label failure rates, tool-call error rate, parse failures, average steps.
- **BFCL v3:** simple, multiple and irrelevance, 100 samples each at a pinned dataset revision, sent in the prompted
  format and scored with AuditKIT (`src/shipwright/heal/bfcl_eval.py`).

## Commands

```sh
# Offline: the whole gate with fake models (CI runs this path)
.venv/bin/python scripts/gate_eval.py --round 0 --local-dry-run \
    --current configs/systems/base.yaml --candidate configs/systems/r1-tool.yaml --num-tasks 2

# Decide from two finished run dirs (exit 0 promote, 1 refuse)
.venv/bin/python scripts/gate_eval.py --round N --current-run runs/<current> --candidate-run runs/<candidate> \
    --target-label invented_id

# Plan a gate job (prints flavor, timeout and worst-case cost; launches nothing), then launch it
.venv/bin/python scripts/gate_job.py --round N --current configs/systems/current.yaml \
    --candidate configs/systems/<candidate>.yaml --user-model openai/openai/gpt-oss-120b:deepinfra
.venv/bin/python scripts/gate_job.py ... --launch     # paid: GPU and simulator calls
```

`gate_job.py` defaults to an `a10g-large` with a 60-minute timeout (about $1.50 of GPU at most; a gate takes about 40
minutes of running time). `--runs-only` plays and pushes without deciding; `--bfcl-screen` scores BFCL alone on the
candidate (about 8 minutes) before paying for a full gate; `--hidden` is the one-time final run on the hidden split.

## Output

In the runs dataset under `gate/round-N/<candidate>/`, and in `runs/gate-rN/` once pulled:

| File | Content |
|---|---|
| `<system>/` | the run directory of each system, with `failures.jsonl` and `bfcl.json` |
| `metrics_current.json`, `metrics_candidate.json` | every metric, per task and per label, with label coverage |
| `gate_decision.json` | `decision`, every check with its numbers, both verdicts (AgentTune, AuditKIT), thresholds, `config_sha` |
| `gate_inputs.json` | the resolved systems, patch ids, targets and user model |
| `metrics.jsonl` | one row per gate, also appended at the root of the runs dataset |
