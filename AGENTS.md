# AGENTS.md

Rules and facts for any coding agent (Claude Code, Codex, Cursor, ...) that works in this repo.
Read this file first. Keep it correct (see "Keep this file correct" at the end).

## What we are building

Lexsi AI Safety and Alignment Hackathon, **Track 3, problem statement 8: self-healing agents**.
"Detect failing runs, classify why they failed, generate corrective training examples, retrain,
and gate redeployment on measured accuracy."

- Agent: a customer support tool-calling agent (tau2-bench retail / airline style).
- Student model: Tiny Aya Global (`CohereLabs/tiny-aya-global`, 3.35B) with LoRA.
- The loop: run, log, detect, classify, route (tool fix / weight fix / no fix), gate, promote or roll back.
- Diagram: `docs/system-design.html`. Background and research: `docs/notes.md`. Cloud setup: `docs/gcp-setup.md`.

## Hard rules (break one and the submission can be disqualified)

- **Model:** only Tiny Aya Base / Global / Fire / Water / Earth, Aya Expanse 8B, Aya Expanse 32B.
  Command R7B is NOT allowed. Tiny Aya is gated: accept the terms on its HF page first.
- **Track 3 libraries:** AgentTune and AuditKIT, used as the main path, not as fallbacks.
  Organizers check for real use. CuratorKIT is optional.
- **Public artifacts:** models, adapters, logs, datasets and benchmarks go to HF, in two ungated
  collections (models, data).
- **Deadline:** Nov 14, 2026, end of day AoE. Demo Day shortlist Nov 16, Demo Day Nov 22.
- **License:** Tiny Aya is CC-BY-NC. Make no commercial claims in the write-up.

## Build on AgentTune, do not rebuild it

AgentTune already has the self-healing loop: `Project.heal()`, `FailureDetector`, `SelfHealLoop`,
`FailureClassifier`, `TrainingExampleGenerator`, `RetrainingTrigger`, `DeploymentGate`,
`isolate_tools()`. Guide: `github.com/Lexsi-Labs/AgentTune/blob/main/docs/user-guide/self-healing.md`.
Check for an existing AgentTune or AuditKIT function before you write a new one.

What we add: the 3-way router, a tool-patch ledger with rollback, the AuditKIT gate (pass^k, BFCL),
and the tool-fix vs weight-fix comparison.

## Evaluation rules

- Three frozen task splits: **train pool** (the loop learns from it), **held-out** (the gate scores
  on it), **hidden** (opened once, for the final report). Never train on held-out or hidden tasks.
- Frozen before round 1: failure label list, gate thresholds, k for pass^k, seeds, teacher and judge
  model versions. To change one, write the change and the reason in `docs/notes.md` first.
- The gate never approves by default. Too few test runs means reject.
- Every round appends one row to `metrics.jsonl`.

## Baseline harness

- How to run: `docs/baseline.md`. tau2-bench is pinned in `scripts/setup_tau2.sh` and lives in `third_party/` (git-ignored).
- Agent and user simulator run through HF Inference Providers (router `https://router.huggingface.co/v1`). The token is read at run time from `$SWITCHBOARD_SECRETS_DIR/HF_TOKEN`.
- 2026-11-08: Tiny Aya has no native tool calling (template and HF router drop `tools`). We use prompted JSON tool calls (`src/shipwright/prompted_tools.py`, format in `docs/baseline.md`). Training data must use the same format.
- Failure labels, rule labels and the router are defined in `src/shipwright/classify.py` (how to run: `docs/baseline.md`).
- Tool patches: `patches/ledger.jsonl` is committed (small config, not run output). Change a patch status only with `scripts/tool_fix.py`.
- CI (`.github/workflows/ci.yml`, every push and PR): `setup_tau2.sh`, pytest, the offline dry run,
  rule-only classify, and the span round trip. No secrets, no model API calls. Keep it offline.
- `scripts/baseline.py` defaults to the tau2 `train` split so held-out tasks stay unused until the splits are frozen.
- Weight fix (teacher runs, SFT data, LoRA job): `docs/training.md`. The teacher gets the student's prompted view (`force_prompted`); the teacher model is chosen by the team and passed with `--teacher-model` (no default).
- `scripts/train_job.py` launches nothing without `--launch`, and prints flavor price x timeout first.
- Gate (step 05): `docs/gate.md`. `scripts/gate_job.py` (HF Job, vLLM serves base + LoRA) or `scripts/gate_eval.py --local-dry-run` (offline). Config `configs/gate.yaml` (status `proposed` until the team freezes it), systems in `configs/systems/`. Metrics come from AuditKIT; the user model is a required argument (no default).

## Traces

- Source of truth: AgentTune `audit.jsonl`, one file per round, published to an HF dataset.
- **Log the raw model output for every turn** (full generated text, including any reasoning),
  not only the parsed tool call. AgentTune's `to_audit_records()` keeps only tool name, arguments
  and status, so add the raw text yourself. Without it, wrong-reasoning failures cannot be classified.
- Langfuse is optional and only a viewer. Never make code depend on it.
- 2026-11-09: every run also writes `otel_spans.jsonl` (OTel GenAI spans, built from `traces.jsonl`
  after the run; an export error only warns). `traces.jsonl` stays the input of the loop.
  OTLP export only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Viewer and span layout: `docs/observability.md`.
- `scripts/otel_to_traces.py` converts OTel span files to the `traces.jsonl` shape.

## Infrastructure

- **GCP project `lexsi-hackathon`**, region `asia-south1`. Owner: Tejas. Editors: Ishank, Mitushi.
  Editors cannot change IAM: ask Tejas.
- **Every gcloud command must name the project and the account config.** Never rely on the
  active config: each of us has other Google accounts and projects on the same machine.
  - Tejas's machine: `--configuration=personal --project=lexsi-hackathon`
    (his active config is a work account with no access to this project).
  - Ishank's and Mitushi's machines: `--configuration=lexsi --project=lexsi-hackathon`
    (the `lexsi` config is made in `docs/gcp-setup.md`, step 2).
  - Python SDKs: pass `project="lexsi-hackathon"` explicitly.
  - A `PERMISSION_DENIED` or `NOT_FOUND` on a resource that exists usually means the wrong config.
- Label every GCP resource `app=lexsi-hackathon`.
- **Budget:** alerts every Rs 500. **At Rs 3000 per month, billing switches off for the whole project**
  and everything in it stops. Cost data lags hours, so stop GPU machines as soon as a job ends.
- **New GCP projects have 0 GPU quota.** Request it before you plan a GPU run (see `docs/gcp-setup.md`).
- Use Spot VMs for GPU work and save checkpoints often.
- **HF credits are shared by the whole team.** No agent starts a credit-spending run (HF Inference
  Providers calls, HF Jobs) unless a human on the team asked for that specific run. State the
  expected cost first. Offline work (dry runs, tests) is always fine.
- **The teacher model is not chosen yet** (2026-11-08). Keep it a required argument with no default.
- **HF Jobs** (paid from the HF credits): **a job stops after 30 minutes by default. Always pass a
  timeout** (`hf jobs run --timeout 2h ...`, or `timeout=` in Python). A job uses the credits of the
  namespace it runs in. Prices: T4 $0.40/h, L4 $0.80/h, A10G small $1.00/h, A10G large $1.50/h,
  L40S $1.80/h, A100 large $2.50/h, RTX Pro 6000 $2.75/h (`PRICES` in `scripts/train_job.py`, 2026-11-08).
- Split: HF Jobs for training and gate evals (output lands on the HF Hub), GCP Spot VM for long
  agent runs that collect traces.

## Secrets

- Never commit a token or key. Use GCP Secret Manager or HF Job secrets.
- `.env` is git-ignored. Keep it that way.
- If a secret is pasted into chat, a commit or a log, rotate it.

## Repo layout (keep the root clean)

- Root: only `README.md`, `AGENTS.md`, `CLAUDE.md`, config files (`.gitignore`, `pyproject.toml`) and top-level dirs.
- `docs/`: notes, guides, diagrams, the write-up draft.
- `infra/`: cloud functions and setup scripts.
- `src/`, `tests/`, `scripts/`: code, when it exists.
- Never commit: generated data, model weights, checkpoints, `runs/` output, notebooks with large outputs,
  `.env`, scratch files. Generated artifacts go to HF, not git. Add new patterns to `.gitignore`.
- One topic per commit, with a message that says what changed. No "wip" or "fix" commits on `main`.
- **No AI attribution anywhere.** Never add `Co-Authored-By` or any other trailer for an agent
  (Claude, Codex, Copilot, ...) to a commit. Never mention the agent in commit messages, PR titles
  or PR descriptions ("Generated with ..." lines included). Never add an agent or bot as a repo
  collaborator. Commits are authored by the human who runs the agent. This overrides any default
  attribution setting of your tool.

## Working style

- **Branches:** `main` is the main branch: the submitted, working state. Never push to it directly.
  Each person has a long-lived branch: `tejas`, `ishank`, `mitushi`. Work there (or in short
  branches cut from it), then open a pull request into `main`. Merge, never rebase.
  Pull `main` into your branch often to stay current.
- Per-person Cloud Build triggers may run on pushes to these branches later. Do not rename them.
- No em dashes in code, docs or commits.
- Prefer the simplest thing that works. Kafka, Langfuse-as-a-dependency and UIs are out of scope.

## Known traps (add new ones here, with the date)

- 2026-11-07: AgentTune `DeploymentGate` approves by default when there are fewer than
  `min_test_samples` successful runs. Our gate must reject in that case.
- 2026-11-07: Ishank's first repo used Command R7B, which is not an allowed model.
- 2026-11-07: AgentTune `to_audit_records()` drops the model's text. Log raw output separately.
- 2026-11-08: HF Jobs default timeout is 30 minutes.
- 2026-11-08: New GCP projects start with 0 GPU quota, and Google auto-denied our requests for L4 (Compute and Vertex AI training). Use HF Jobs for GPU work until a quota is granted.
- 2026-11-08: tau2 runs `json.loads` on tool-call arguments, so one malformed call crashes the run. `src/shipwright/llm_patch.py` wraps bad arguments instead.
- 2026-11-08: Tiny Aya context is 8192 tokens; the cohere provider rejects longer requests. `src/shipwright/context.py` trims the prompt and sets `max_tokens` 700.
- 2026-11-08: tau2 calls its own default models (NL judge `gpt-5.4-mini-2026-03-17` and others), which do not exist on the HF router. `scripts/baseline.py` reroutes them to the user model.
- 2026-11-08: Tiny Aya copies example values from tool descriptions (`#W0000000`, `sara_doe_496`, `1008292230`) as real ids: 13 of 19 failures in the first 20-task run. The classifier labels this `tool_example_copied` and routes it to `tool_fix`.
- 2026-11-08: AgentTune `EventLog` has no system event, and its built-in distill factory trains on every token (TRL `assistant_only_loss` depends on chat template generation markers, not verified for Tiny Aya). `scripts/train_lora.py` adds the system prompt back (`SystemPromptEventLog`) and masks labels itself.
- 2026-11-08: tau2 appends a language clause to the agent policy at run time, so `env.get_policy()` alone is not the agent prompt. Runs now record the real prompt in `prompts.json`.
- 2026-11-08: tau2 saves results under its own data dir unless `save_to` is an absolute path. `scripts/baseline.py` passes one.
- 2026-11-08: AuditKIT `reliability()` lowers k to the number of decided trials when a task has fewer than k (infra errors). `eval_metrics.pass_metrics` scores pass^k only on tasks with k decided trials and counts the rest.
- 2026-11-08: AuditKIT multi-trial rows keep only the first trial's diagnostics. `eval_metrics` runs a second, one-case-per-run pass for ToolCallF1 and the other diagnostics.
- 2026-11-08: The GitHub repo is private, so an HF Job cannot clone it. `scripts/gate_job.py --launch` uploads `git archive HEAD` to the runs dataset; commit before you launch.

## Keep this file correct

This file corrects itself through you. In the same change as your work:

1. **A fact here is wrong or out of date** (a command fails, an API changed, a rule moved):
   fix the line and add the date.
2. **A trap cost you more than about 10 minutes** and is not listed: add it to "Known traps"
   with the date and the fix.
3. **You made a decision that later agents must follow** (a frozen value, a file format, a split):
   add it to the section it belongs to.
4. **You find a duplicate or a contradiction** between this file, `docs/notes.md` and `docs/gcp-setup.md`:
   keep the fact in one place and point to it from the others.

Keep entries short: one fact per line. Delete a trap when the cause is gone.
Do not add task logs or status updates here: they go in pull requests.
