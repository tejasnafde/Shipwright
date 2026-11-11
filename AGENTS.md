# AGENTS.md

Rules and facts for any coding agent (Claude Code, Codex, Cursor, ...) that works in this repo.
Read this file first. Keep it correct, and keep it the *only* such file (see "Keep this file
correct" at the end) - never add a second notes/decisions/journal file alongside it.

## What we are building

Lexsi AI Safety and Alignment Hackathon, **Track 3, problem statement 8: self-healing agents**.
"Detect failing runs, classify why they failed, generate corrective training examples, retrain,
and gate redeployment on measured accuracy."

- Agent: a customer support tool-calling agent (tau2-bench retail / airline style).
- Student model: Tiny Aya Global (`CohereLabs/tiny-aya-global`, 3.35B) with LoRA.
- The loop: run, log, detect, classify, route (tool fix / weight fix / no fix), gate, promote or roll back.
- Diagram: `docs/system-design.html`. Cloud setup: `docs/setup/gcp-setup.md`,
  `docs/setup/cloud-runs.md`. HF account/token setup: `docs/setup/hf-setup.md`.

## Hard rules (break one and the submission can be disqualified)

- **Model:** only Tiny Aya Base / Global / Fire / Water / Earth, Aya Expanse 8B, Aya Expanse 32B.
  Command R7B is NOT allowed. Tiny Aya is gated: accept the terms on its HF page first.
- **Track 3 libraries:** AgentTune and AuditKIT, used as the main path, not as fallbacks.
  Organizers check for real use. CuratorKIT is optional.
- **Public artifacts:** models, adapters, logs, datasets and benchmarks go to HF, in two ungated
  collections (models, data).
- **Deadline:** the official due time is Nov 15, 2026, 11:59 PM AoE (hackathon.lexsi.ai/structure, checked 2026-11-12); we plan to finish by Nov 14 AoE to keep a day spare. Shortlist (up to 20 teams) is told by Nov 16; shortlisted teams get until Nov 22, 11:59 PM AoE for a video or slides and $200 more HF credits. Judging on Nov 15 to 16 uses what is submitted by the deadline.
- **License:** Tiny Aya is CC-BY-NC. Make no commercial claims in the write-up.

## Submission and judging

- One submission counts: the last one before the deadline. Organizers recommend picking one track.
- Double-blind: more than one judge scores each team; normalized averages pick the Demo Day
  teams. Rubric: write-up clarity /10, practical usefulness /10, code/notebook quality /10,
  overall review /100. An internal team first checks validity (public artifacts, write-up,
  checklist, no duplicates), then tags each submission to a track.
- Rules page (checked 2026-11-12): before-and-after metrics on more than one metric; the write-up says which metrics and why, the approach and why these datasets, tasks and models, and how the model is meaningfully better; an HF article; an HF Space if relevant; an optional but preferred ROI comparison (closed-model token billing vs self-hosting: sample volume, GPU hours, GPU price per hour, tokens/sec). Track 3 is exempt from the 2-repos rule. Late or incomplete submissions may not be considered.
- HF article checklist: (1) problem statement and approach, (2) hackathon repos used and for
  what, (3) models used and their licenses, (4) datasets/benchmarks used and their licenses,
  (5) model artifacts as a public HF collection, (6) data artifacts as a public HF collection,
  (7) experiments run and findings, (8) with more credits, what you'd try next, (9) teammates
  tagged (judges mask names for the double-blind round). A markdown write-up in the GitHub repo
  is accepted if HF article access is a problem.
- Organizer clarifications (Discord): not publishing harmful weights is fine; tagging teammates
  in the article is fine. Docs: agenttune.lexsi.ai, audit.lexsi.ai, curatorkit.lexsi.ai/latest/.
  Office hours run daily (server Events tab); off-list scope questions go in #report_issues.

## Build on AgentTune, do not rebuild it

AgentTune already has the self-healing loop: `Project.heal()`, `FailureDetector`, `SelfHealLoop`,
`FailureClassifier`, `TrainingExampleGenerator`, `RetrainingTrigger`, `DeploymentGate`,
`isolate_tools()`. Guide: `github.com/Lexsi-Labs/AgentTune/blob/main/docs/user-guide/self-healing.md`.
Check for an existing AgentTune or AuditKIT function before you write a new one.

What we add: the 3-way router, a tool-patch ledger with rollback, the AuditKIT gate (pass^k, BFCL),
and the tool-fix vs weight-fix comparison.

Design inspiration drawn from outside Track 3 too: the tool-patch ledger + one-step rollback
follows NousResearch/hermes-agent's `skill_ledger.py` pattern (JSONL audit, content-addressed
before/after snapshots, never auto-delete). Closest Track 3 competitor found in a repo scan
(2026-11-07, 18 repos): UEddy/Kataa (Tiny Aya + AgentTune, using AgentTune's own DPO self-heal
path) - its gate rejected every repair attempt (99% escalation) because it is not a 3-way
failure-routing loop; closing that gap is this project's main bet.

2026-11-09: detect, classify, the retrain trigger and the gate call AgentTune classes
(`src/shipwright/heal/agenttune_bridge.py`, table in `docs/pipeline/baseline.md`); our rules and AuditKIT thresholds
add to them. agenttune is installed `--no-deps` (`scripts/setup_tau2.sh`): its torch pin fails on a Mac.
Traps: `FailureDetector` resumes from `<audit>.offset` (we delete it), and `agenttune.agentic` imports torch.

## Evaluation rules

- Three task splits, ids in `configs/splits.yaml` (frozen 2026-11-11, Mitushi): **train pool** (the
  whole tau2 retail `train` split, 74 tasks; the loop learns from it), **held-out** (20 of the 40
  `test` tasks; the gate scores on them), **hidden** (the other 20 `test` tasks; opened once, for
  the final report). Never train on held-out or hidden tasks. `scripts/baseline.py` refuses hidden
  ids, and a `test` run without `--task-ids`, unless `--open-hidden` is passed
  (`src/shipwright/heal/splits.py`). Why a held-out/hidden split at all: the code originally
  trained on the whole train split and gated on the whole 40-task test split, with no hidden set -
  the final numbers would otherwise come from the same tasks the gate used to pick rounds. The ids
  are listed explicitly (not re-derived) so a tau2 update can't move them.
- Gate floors are sized for 20 held-out tasks x k=3 (60 runs per system), not the full 40-task test
  split: `min_valid_runs` 54 (90% of 60), `min_complete_tasks` 18, `target_label_min_drop` 0.05
  (one of 3 runs/task), `non_target_max_increase` 0.05. `configs/gate.yaml` is frozen together with
  the splits (`status: frozen`, 2026-11-11) - a stale branch with the pre-split numbers
  (`min_valid_runs: 108`, sized for 40 tasks) would reject every gate for "too few valid runs"
  against a split it was never actually scoring on; freezing both files together is what stops
  that drift from coming back.
- Frozen before round 1: failure label list, gate thresholds, k for pass^k, seeds, teacher and judge
  model versions. To change a frozen value, add the change and the reason as a new line here.
- The gate never approves by default. Too few test runs means reject.
- 2026-11-12 (Ishank): the headline result is task accuracy on the held-out tasks (pass^1 and pass^k from AuditKIT), because problem statement 8 says to "gate redeployment on measured accuracy". Tool-error and label rates are supporting metrics (the rules ask for before-and-after numbers on more than one metric), never a replacement. A low base pass rate is fixed by improving the student and the data, not by changing the headline metric.
- Every round appends one row to `metrics.jsonl`.
- 2026-11-12 (Mitushi), frozen-value change: `classify.py` now also finds `premature_transfer` by rule (a transfer
  before any attempt to authenticate). The label list is unchanged: `premature_transfer` was already a judge label.
  Reason: it was the most common round 1 failure (37 of 59 valid runs transferred, 18 as the very first tool call)
  and no rule counted it, so no candidate could target it.

## Baseline harness

- How to run: `docs/pipeline/baseline.md`. tau2-bench is pinned in `scripts/setup_tau2.sh` and lives in `third_party/` (git-ignored).
- Tiny Aya (the agent) runs through HF Inference Providers (router `https://router.huggingface.co/v1`). The token is read at run time from `$SWITCHBOARD_SECRETS_DIR/HF_TOKEN`.
- 2026-11-09: model ids `vertex/<publisher>/<model>` go to Vertex AI (`src/shipwright/llm/vertex.py`), for the teacher, user simulator, tau2 judge, `classify.py` judge and `tool_fix.py` proposer. All other ids go to the HF router unchanged. Smoke check: `scripts/check_vertex.py`.
- 2026-11-09, team choice: teacher `vertex/deepseek-ai/deepseek-v4.1-flash-maas`; user simulator and judge `vertex/openai/gpt-oss-120b-maas`. `--teacher-model` and `--user-model` stay required with no default; `classify.py --classifier-model` (AgentTune), `--judge-model` and `tool_fix.py --model` default to the Vertex gpt-oss.
- `summary.json` `cost_estimate_usd` is per role (agent or teacher, user, judge) by provider, with cached prompt tokens at the cached price. Prices: `src/shipwright/llm/costs.py`.
- 2026-11-08: Tiny Aya has no native tool calling (template and HF router drop `tools`). We use prompted JSON tool calls (`src/shipwright/llm/prompted_tools.py`, format in `docs/pipeline/baseline.md`). Training data must use the same format.
- Failure labels, rule labels and the router are defined in `src/shipwright/heal/classify.py` (how to run: `docs/pipeline/baseline.md`).
- Tool patches: `patches/ledger.jsonl` is committed (small config, not run output). Change a patch status only with `scripts/tool_fix.py`.
- 2026-11-12 (Mitushi): tool guards, the "add validation" half of the tool fix (`src/shipwright/llm/guards.py`,
  `baseline.py --tool-guards`, system key `tool_guards: true`, candidate `configs/systems/r1-guards.yaml`). Before a
  student tool call runs, policy checks: authenticate first, no made-up ids, no transfer before trying to
  authenticate, an explicit yes before a change, no repeat of a failed call, order status fits the tool. On a
  violation the agent is asked again with the reason (at most 2 times, then the call runs as written). Rejected
  replies go to the trace (`guard_rejections`), never into the history, so training data stays clean. Off by
  default and never on for teacher runs. How it was checked: `docs/pipeline/baseline.md`, "Tool guards".
- 2026-11-12 (Mitushi): the prompted-tools instruction has a 5-step "How to handle a request" checklist
  (authenticate, look ids up, confirm before a change, read errors, transfer last). Like the few-shot change
  below, runs made before it are not comparable with new ones.
- CI (`.github/workflows/ci.yml`, every push and PR): `setup_tau2.sh`, pytest, the offline dry run,
  rule-only classify, and the span round trip. No secrets, no model API calls. Keep it offline.
- `scripts/baseline.py` defaults to the tau2 `train` split so held-out tasks stay unused until the splits are frozen.
- Weight fix (teacher runs, SFT data, LoRA job): `docs/pipeline/training.md`. The teacher gets the student's prompted view (`force_prompted`); the teacher model is passed with `--teacher-model` (no default; team choice above).
- `scripts/train_job.py` launches nothing without `--launch`, and prints flavor price x timeout first.
- 2026-11-11: Agent-R (MCTS revision data) was removed as scope creep: 1 valid revision for about $1 in round 1.
  The only weight-fix data source is teacher replay. Do not add a second data source before a full round has
  reached a gate decision.
- 2026-11-12: round 2 data: 3 of 74 passed (as in round 1), `invented_id` still 47 of 71 failures: the copied value was `#W0000000` from the tau2 tool descriptions (43 of 85 "Order not found" calls), not only our few-shot id. The student's tool list now hides real-looking examples (`prompted_tools.example_hint`); a baseline made before this is not comparable. The teacher replays `tool_fix` and `weight_fix` failures (round 2 sent only the 24 `weight_fix` ones: 21 SFT examples), `--only-solved-from` + `--teacher-trials 5` harvests more passes from tasks it has solved (about 48% pass again vs 34% overall); `--skip-unsolved-from` and `--skip-solved-from` drop tasks it never solved or already solved. A copied tool example is `tool_example_copied` (route tool_fix) only if that run's tool list showed it (`tool_list_version` 1 in `run_config.json`); from version 2 it is `invented_id` and goes to the weights. `configs/systems/r3-weight.yaml` targets every weight_fix label; `run_round.py ceiling` runs the teacher on the held-out tasks (evaluation only); the gate prints pass^1 per slice (0-1 writes, 2+ writes; context, not part of the decision); `scripts/promote.py` keeps the deployed adapter in `promotions/ledger.jsonl` (rollback, rows never deleted) and the next gate's `--current` is `configs/systems/current.yaml`. Missing: a GPU baseline of the deployed adapter on the train tasks, and the data stage pools earlier teacher runs of the same model (`--extra-teacher-runs`; 140 examples from 47 tasks on what we have, versus 21 from round 2 alone). `run_round.py` stops between stages on over 25% infra errors or under 40 SFT examples (`--force`), waits for the Cloud Build to repoint the job, and splits spend into metered and estimated. Orchestrator logs go to `runs/_meta/` (`log.run_script(meta=True)`).
- 2026-11-12: A round runs through `scripts/run_round.py` (`docs/pipeline/round.md`): one stage per command, a cost estimate and a hard cap per stage (`--max-cost-usd` on `baseline.py` and `teacher_runs.py`), a manifest `runs/round-N.json`. The baseline and the weight candidate have no tool patches, so a gain is the weights. The round 2 gate uses an HF router user simulator (`GATE_USER_MODEL` in the script) and a 90 minute limit: a Vertex token would expire mid-gate.
- Gate (step 05): `docs/pipeline/gate.md`. `scripts/gate_job.py` (HF Job, vLLM serves base + LoRA) or `scripts/gate_eval.py --local-dry-run` (offline). Config `configs/gate.yaml` (status `proposed` until the team freezes it), systems in `configs/systems/`. Metrics come from AuditKIT; the user model is a required argument (no default).

## Logs and debugging

- How to debug a run: `docs/ops/debugging.md` (`scripts/why.py`, the links, `scripts/rerun.py`).
- Log with `shipwright.observability.log` (`log.get_logger`, keywords as fields), never `logging.basicConfig` or a
  new logger setup. Every script ends with `sys.exit(log.run_script(main, step="<step>"))`.
- Every script writes `log.jsonl`, `errors.json` and `repro.json` into its run dir
  (`log.attach_run_dir`), and `status` (ok / partial / failed) into `summary.json`.
- New external call (model, HTTP, GCS, HF Hub): wrap it in `shipwright.observability.instrument.external` (or go
  through `llm_patch`), so it gets a log line and a span.
- Never print or log a token or key. `shipwright.observability.log` redacts `hf_`, `ya29.`, `Bearer`, `sk-`; do
  not rely on that for other formats.

## Traces

- Source of truth: our `traces.jsonl` per run (format in `docs/pipeline/baseline.md`), plus `otel_spans.jsonl`
  built from it. Run dirs are published to an HF dataset. AgentTune's `audit.jsonl` is only
  AgentTune's own format: convert to or from it where an AgentTune API needs it.
- **Log the raw model output for every turn** (full generated text, including any reasoning),
  not only the parsed tool call. AgentTune's `to_audit_records()` keeps only tool name, arguments
  and status, so add the raw text yourself. Without it, wrong-reasoning failures cannot be classified.
- Langfuse is optional and only a viewer. Never make code depend on it.
- 2026-11-09: every run also writes `otel_spans.jsonl` (OTel GenAI spans, built from `traces.jsonl`
  after the run; an export error only warns). `traces.jsonl` stays the input of the loop.
  OTLP export only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Viewer and span layout: `docs/ops/observability.md`.
- `scripts/otel_to_traces.py` converts OTel span files to the `traces.jsonl` shape.

## Demo: control-room (2026-11-10, Ishank)

- `src/shipwright/control_room/app.py` (Streamlit) is a read-only replay of a round's real
  artifacts, step by step: run -> detect -> classify -> route -> generate/patch -> curate/train ->
  gate -> outcome. It never calls a teacher API, a trainer or the gate itself - only reads what
  the harness scripts already wrote. Run: `streamlit run src/shipwright/control_room/app.py`.
- Because a "round" here is several independently-timestamped run dirs, not one folder
  (`runs/baseline-<ts>/`, a `build_sft.py --output-dir`, `runs/gate-round-N-<mode>-<ts>/`), the app
  asks for up to three dirs in its sidebar instead of a single round number - the same three a
  person tracks by hand when running a round for real.
- Deploys as its own, much smaller Hugging Face Space image: `infra/space/Dockerfile` (no GPU, no
  AgentTune/AuditKIT, no secrets) - see `infra/space/README.md`. Decided after reviewing a strong
  submission for a different problem statement: a live, interactive HF Space is clearly rewarded in
  this judging round.
- 2026-11-11: `scripts/demo_data.py` writes four clearly-labeled staged run dirs (a round-1
  failure, its teacher fix, a staged round-2 pass, a staged gate rejection) for the "Follow a
  Task" tab and a genuine gate-rejection walkthrough, since no real round-2 adapter has been
  trained yet. Every staged dir carries a `demo.json` marker; `render_demo_banner()` shows an
  orange "STAGED EXAMPLE" banner wherever one is loaded, so staged and real data are never
  visually confused.
- 2026-11-11 (Ishank): the sidebar's "Load demo scenario" dropdown (`SCENARIOS` in `app.py`: the
  real round 1, the staged walkthrough) fills in every run-dir picker. Baseline and teacher/SFT
  pickers stay in the sidebar; the gate picker sits in Gate & Outcome and the after-training
  picker in Follow a Task (the only tabs that read them; the header is filled after them, since
  it reads the gate dir too). A scenario is listed only when all of its dirs exist. Traces render as
  chat bubbles with a raw-JSON expander under each, so no caller may put a trace inside an
  expander (Streamlit cannot nest them).
- 2026-11-12 (Ishank): Fix & Train shows tool patches per baseline run, not per ledger `round`:
  "active" = the ids in that run's `run_config.json` `active_tool_patches`, "proposed from" =
  ledger rows whose `source_failures` are that run's simulation ids. The 35 round-1 patches were
  proposed on Nov 8 from a run no longer on disk, and `r1-baseline-train` ran with all 35 active.

## Infrastructure

- **GCP project `lexsi-hackathon`**, region `asia-south1`. Owner: Tejas. Editors: Ishank, Mitushi.
  Editors cannot change IAM: ask Tejas.
- **Every gcloud command must name the project and the account config.** Never rely on the
  active config: each of us has other Google accounts and projects on the same machine.
  - Tejas's machine: `--configuration=personal --project=lexsi-hackathon`
    (his active config is a work account with no access to this project).
  - Ishank's and Mitushi's machines: `--configuration=lexsi --project=lexsi-hackathon`
    (the `lexsi` config is made in `docs/setup/gcp-setup.md`, step 2).
  - Python SDKs: pass `project="lexsi-hackathon"` explicitly.
  - A `PERMISSION_DENIED` or `NOT_FOUND` on a resource that exists usually means the wrong config.
- Label every GCP resource `app=lexsi-hackathon`.
- **Budget:** alerts every Rs 500. **At Rs 3000 per month, billing switches off for the whole project**
  and everything in it stops. Cost data lags hours, so stop GPU machines as soon as a job ends.
- **New GCP projects have 0 GPU quota.** Request it before you plan a GPU run (see `docs/setup/gcp-setup.md`).
- Use Spot VMs for GPU work and save checkpoints often.
- **HF credits are shared by the whole team.** No agent starts a credit-spending run (HF Inference
  Providers calls, HF Jobs) unless a human on the team asked for that specific run. State the
  expected cost first. Offline work (dry runs, tests) is always fine.
- **Vertex AI (2026-11-09)** for the teacher, user simulator and judge (spends GCP budget, not HF credits):
  OpenAI-compatible endpoint `https://aiplatform.googleapis.com/v1/projects/lexsi-hackathon/locations/global/endpoints/openapi`.
  Auth is a `gcloud --configuration=$SHIPWRIGHT_GCLOUD_CONFIG auth print-access-token` token (default `personal`,
  teammates set `lexsi`), cached and refreshed after 45 minutes. Inside Cloud Run the service account token
  comes from the metadata server instead. Never ADC or litellm's `vertex_ai` provider.
  Other env vars: `SHIPWRIGHT_GCP_PROJECT`, `SHIPWRIGHT_GCLOUD`, `SHIPWRIGHT_VERTEX_LOCATION`, `SHIPWRIGHT_VERTEX_TOKEN`
  (`src/shipwright/llm/vertex.py`). Prices per 1M tokens: DeepSeek V4.1 Flash $0.30 in, $0.03 cached, $1.20 out;
  gpt-oss-120b $0.09 in, $0.36 out. Tiny Aya stays on the HF router (Vertex has no Cohere models); GPUs stay on HF Jobs.
  The same "a human asked for this run" rule applies to Vertex calls.
- **HF Jobs** (paid from the HF credits): **a job stops after 30 minutes by default. Always pass a
  timeout** (`hf jobs run --timeout 2h ...`, or `timeout=` in Python). A job uses the credits of the
  namespace it runs in. Prices: T4 $0.40/h, L4 $0.80/h, A10G small $1.00/h, A10G large $1.50/h,
  L40S $1.80/h, A100 large $2.50/h, RTX Pro 6000 $2.75/h (`PRICES` in `scripts/train_job.py`, 2026-11-08).
- Split: HF Jobs for training and GPU gate evals (output lands on the HF Hub), Cloud Run Jobs for
  agent runs that collect traces.
- **Live runs go through Cloud Run Jobs, never on a laptop (2026-11-09).** Each person has
  `shipwright-<owner>` (owner: tejas, ishank, mitushi, main), which runs its branch's latest image and
  uploads `runs/<dir>/` to `gs://lexsi-hackathon-runs/<owner>/`. How: `docs/setup/cloud-runs.md`.
  The shared-credit rule above still applies to every job execution. Build images with Cloud Build only.

## Secrets

- Never commit a token or key. Use GCP Secret Manager or HF Job secrets.
- `.env` is git-ignored. Keep it that way.
- If a secret is pasted into chat, a commit or a log, rotate it.

## Repo layout (keep the root clean)

- Root: only `README.md`, `AGENTS.md`, `CLAUDE.md`, config files (`.gitignore`, `pyproject.toml`) and top-level dirs.
- `docs/` (reorganized 2026-11-10, Ishank): `system-design.html` (diagram) at the root; everything
  else grouped by what it's for -
  `docs/setup/` (`gcp-setup.md`, `hf-setup.md`, `cloud-runs.md` - accounts, tokens, infra setup),
  `docs/pipeline/` (`baseline.md`, `gate.md`, `training.md` - how to run each loop step),
  `docs/ops/` (`debugging.md`, `observability.md` - logs, traces, debugging a run).
- `infra/`: cloud functions and setup scripts. `infra/docker/`: the harness image for Cloud Run
  Jobs (`infra/cloudbuild.yaml`). `infra/space/`: control-room's own, much smaller image, meant
  for a Hugging Face Space.
- `src/shipwright/` (reorganized 2026-11-10, Ishank): grouped by concern, not left flat -
  `heal/` (detect/classify/route/fix/gate: `agenttune_bridge.py`, `classify.py`, `gate.py`,
  `tool_fix.py`, `systems.py`, `eval_metrics.py`, `bfcl_eval.py`, `sft_data.py`, `traces.py`),
  `llm/` (model-call adapters: `vertex.py`, `prompted_tools.py`, `patches.py` - was
  `llm_patch.py`, `context.py`, `dry_run.py` - was `fake_llm.py`, `costs.py`),
  `observability/` (`log.py`, `otel_export.py` - was `otel.py`, `otel_import.py` - was
  `otel_convert.py`, `otel_live.py` - was `tracing.py`, `instrument.py`, `report.py`, `repro.py`,
  `runinfo.py`), `control_room/` (the demo UI, unchanged). Import as
  `from shipwright.heal import classify` etc; nothing imports the bare `shipwright.<module>`
  flat form that existed before this reorg.
- `tests/`: mirrors `src/shipwright/`'s subpackages (`tests/heal/`, `tests/llm/`,
  `tests/observability/`); `conftest.py` and `fixtures/` stay at `tests/` root, shared by all of
  them.
- `scripts/`: left flat on purpose, not reorganized. Several scripts import a sibling directly
  (`classify.py` imports `baseline`, `gate_job.py` imports `train_job`) via
  `sys.path.insert(ROOT / "scripts")`, and `job_entry.py`'s Cloud Run dispatch resolves a script
  by bare filename under `scripts/` - nesting would need every one of those fixed for a cosmetic
  gain on a directory that already reads fine at 16 clearly-named files. Don't nest it without
  also fixing both of those.
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
- Code style (2026-11-11, Ishank): readable to a programmer who does not know Python. Descriptive names (`output_directory`, `run_options`, not `out_dir`, `args`), plain `if/else` over clever expressions, named objects (dataclasses) over tuples and string-keyed dicts, shallow nesting, orchestration functions that read like pseudocode. No new abstractions that only wrap one call. New code follows this; `scripts/`, `src/shipwright/heal/` and `src/shipwright/llm/` were converted; `observability/` and `control_room/` are not (left as they were).
- Prefer the simplest thing that works. Kafka, Langfuse-as-a-dependency and UIs are out of scope.

## Known traps (add new ones here, with the date)

- 2026-11-07: AgentTune `DeploymentGate` approves by default when there are fewer than
  `min_test_samples` successful runs. Our gate must reject in that case. `FullClosedLoop`'s
  `GateConfig.min_test_samples` defaults to 1, also in AgentTune's examples (2026-11-09, Ishank).
- 2026-11-07: Ishank's first repo used Command R7B, which is not an allowed model.
- 2026-11-07: AgentTune `to_audit_records()` drops the model's text. Log raw output separately.
- 2026-11-08: HF Jobs default timeout is 30 minutes.
- 2026-11-08: New GCP projects start with 0 GPU quota, and Google auto-denied our requests for L4 (Compute and Vertex AI training). Use HF Jobs for GPU work until a quota is granted.
- 2026-11-08: tau2 runs `json.loads` on tool-call arguments, so one malformed call crashes the run. `src/shipwright/llm/patches.py` wraps bad arguments instead.
- 2026-11-08: Tiny Aya context is 8192 tokens; the cohere provider rejects longer requests. `src/shipwright/llm/context.py` trims the prompt and sets `max_tokens` 700.
- 2026-11-08: tau2 calls its own default models (NL judge `gpt-5.4-mini-2026-03-17` and others), which do not exist on the HF router. `scripts/baseline.py` reroutes them to the user model.
- 2026-11-12: Tiny Aya on the HF router had no retry: a 429, timeout or 5xx ended the run as an infra error.
  `patches.call_with_transient_retries` now waits and retries those (2, 5, 10, 20 s) on every non-Vertex call.
  An empty user-simulator reply (tau2: "UserMessage must have either content or tool_calls") is asked again twice.
  `context.TOKENIZER_MARGIN` is 250 (was 100): a request was rejected 18 tokens over 8192.
- 2026-11-08: Tiny Aya copies example values from tool descriptions (`#W0000000`, `sara_doe_496`, `1008292230`) as real ids: 13 of 19 failures in the first 20-task run. The classifier labels this `tool_example_copied` and routes it to `tool_fix`.
- 2026-11-08: AgentTune `EventLog` has no system event, and its built-in distill factory trains on every token (TRL `assistant_only_loss` depends on chat template generation markers, not verified for Tiny Aya). `scripts/train_lora.py` adds the system prompt back (`SystemPromptEventLog`) and masks labels itself.
- 2026-11-08: tau2 appends a language clause to the agent policy at run time, so `env.get_policy()` alone is not the agent prompt. Runs now record the real prompt in `prompts.json`.
- 2026-11-08: tau2 saves results under its own data dir unless `save_to` is an absolute path. `scripts/baseline.py` passes one.
- 2026-11-08: AuditKIT `reliability()` lowers k to the number of decided trials when a task has fewer than k (infra errors). `eval_metrics.pass_metrics` scores pass^k only on tasks with k decided trials and counts the rest.
- 2026-11-08: AuditKIT multi-trial rows keep only the first trial's diagnostics. `eval_metrics` runs a second, one-case-per-run pass for ToolCallF1 and the other diagnostics.
- 2026-11-09 (Ishank): AgentTune `TrainingExampleGenerator` returns `None` for any root cause other than `wrong_tool` or `loop_collapse`. Our weight fix uses teacher runs (`docs/pipeline/training.md`) instead.
- 2026-11-09 (Ishank): `pip install agenttune` fails on a CPU-only macOS machine (torch 2.11 + bitsandbytes force a source build). It installs on Linux x86_64 (CI, HF Jobs). Do not replace AgentTune with a local reimplementation.
- 2026-11-09: AuditKIT `reliability()` takes `independent=`, while `AgentEvalRunner` takes `reset_confirmed=` (we use the runner). Verdicts are `success` / `failure` / `unknown`, not pass/fail.
- 2026-11-09: DeepSeek V4.1 Flash on Vertex is served only on the `global` endpoint (a `us-central1` call fails). Vertex has no explicit context cache (`cachedContents`) for MaaS open models; implicit prefix caching (Preview) covers `deepseek-v4.1-flash-maas` from 4096 prompt tokens, not gpt-oss-120b. Our fixed teacher prefix (policy + tool list) is about 2.6k tokens, so hits come from the growing conversation of one run.
- 2026-11-09: Vertex gpt-oss-120b spends reply tokens on reasoning (`reasoning_content`) first. `vertex.py` raises `max_tokens` below 1024 to 1024; `reasoning_effort: low` is accepted and now set by `route_kwargs`. 2026-11-12: Vertex calls time out after 60 s (gpt-oss) or 180 s and retry twice (`TIMEOUT_RETRIES`); round 2 had calls hang up to 20 minutes. `run_round.py` defaults to `--max-concurrency 8`.
- 2026-11-09: A Vertex token lasts about 1 hour and an HF Job cannot run gcloud. `gate_job.py` passes a token minted at launch (`SHIPWRIGHT_VERTEX_TOKEN`), with no refresh: a gate job with a Vertex user model must finish within about 1 hour. 20 tasks x k=3 x 2 systems (120 tau2 runs) at `max_concurrency: 8` is the scenario this was sized for; don't raise `heldout.num_tasks` back toward 40 without also moving the user model off Vertex (an HF router model instead), or the run risks losing its user simulator partway through.
- 2026-11-11: In tests, `agenttune.agentic` cannot be imported without torch. Load `agenttune/agentic/events.py` by file
  path and register the module in `sys.modules` before `exec_module`, or its dataclasses fail
  (`'NoneType' object has no attribute '__dict__'`).
- 2026-11-11: The prompted-tools few-shot example used `#W1234567`, and the student copied it as a real order id (28 of 74 round 1 baseline traces, `invented_id` 34 of 71 failures). It now uses `<...>` placeholders (`prompted_tools.py`). Never put a real-looking id in a prompt example; any baseline or teacher run made before this change used the old prompt and is not comparable with new runs.
- 2026-11-08: The GitHub repo is private, so an HF Job cannot clone it. `scripts/gate_job.py --launch` uploads `git archive HEAD` to the runs dataset; commit before you launch.

- 2026-11-11: Vertex gpt-oss-120b (user simulator) returns 429 at 4 to 6 concurrent tau2 runs; with no retry, 15 of 74 baseline runs became infra errors. `vertex.call_with_refresh` now waits and retries a 429 up to 5 times. Some 400s from the same model remain (cause not found yet); 2026-11-12: user and judge calls retry a 400 twice (`BAD_REQUEST_WAITS`), and the NL judge is asked again (twice) when its reply is not JSON (tau2 `json.loads` crashed the run, 3 infra errors in round 1).
- 2026-11-11: `SHIPWRIGHT_HF_TOKEN` (mitushigoyal5, fine-grained) can write the Hub but HF Jobs return
  `403 ... missing permissions: job.write` on namespace mitushigoyal5. The token needs the Jobs permissions
  (read and write) before `train_job.py --launch` or `gate_job.py --launch` can work.
- 2026-11-09: Cloud Trace `batchWrite` returns 200 but stores nothing while `observability.googleapis.com` is off (read gives `_Trace bucket not found`). Tejas must enable that API once; until then use `live_spans.jsonl` and the logs (they work).
- 2026-11-10 (Ishank): confirmed by actually installing `agenttune==1.1.3 --no-deps` and
  inspecting the real classes (not just source reading) - `FailureClassifier` has no `classify()`
  method; the real API is `async def classify_batch(failures)` / `async def _classify_single(f)`,
  and `_build_prompt()` returns `list[dict]` chat messages, not a string. `DeploymentGate.score_model`
  calls `model_fn(case["input_text"])` and compares the result case-insensitively against
  `case["expected_verdict"]` - a test-set item needs exactly those two keys, not `task_id`/anything
  else. `agenttune_bridge.py`'s `ShipwrightFailureClassifier`/`agenttune_gate_check` already match
  this; this line exists so the next person who touches that file does not have to re-derive it
  from source, or worse, from an earlier draft that got it wrong (an `ig/main` prototype of the
  same classifier/gate wrapper called a `classify()` that does not exist - caught in this merge,
  never shipped).
- 2026-11-11 (Ishank): Tiny Aya's 262,144-token vocabulary makes the training-time logits tensor
  (`[seq_len, vocab]`) the real memory bottleneck for `scripts/train_lora.py`, not the base model's
  own weights or LoRA's ~30M trainable params - at our 8192-token `--max-seq-len`, one bf16 logits
  tensor alone is ~4.3 GB before gradients. Found independently by a different Track 3 team hitting
  the same wall on a 16 GB T4 for their DPO trainer (github.com/Timidan/lexsi-hack,
  `scripts/train_dpo.py`): a `logits_to_keep` window that skips projecting the loss-masked prefix
  through the lm_head. Ported as `windowed_sft_trainer()` in `train_lora.py`, adapted for our own
  `SFTTrainer` subclass (`create_distill_trainer`'s default factory builds a plain TRL
  `SFTTrainer`, not DPOTrainer - confirmed by reading `agenttune.agentic.distill_trainer` source
  directly) and computing the loss by hand on the returned logits rather than trusting any
  model-specific `labels=`+`logits_to_keep` interaction, which was not verified against Tiny Aya's
  real modeling code (no GPU here to check). Only applies when `--batch-size 1` (the default);
  falls back to the unwindowed loss otherwise, and also falls back if the loaded model rejects a
  plain int `logits_to_keep` kwarg. Confirm on the real GPU host before trusting the memory
  savings in a real training run - the *correctness* of the windowed loss itself does not depend on
  anything Tiny-Aya-specific (`logits_to_keep` as a plain int is a long-standing, architecture-
  independent `transformers` API), only the *memory savings* are unverified.
- 2026-11-11: `scripts/teacher_runs.py`'s `kept.jsonl` is a list of full trace dicts (reward == 1
  runs, same shape as `traces.jsonl` rows - `messages`, `reward`, `task_id`), not a
  `{student/rejected, teacher/chosen}` pair. Control-room used to read it expecting that pair
  shape and silently showed "(not captured)" on every real run dir - fixed the same day by
  reading `kept.jsonl` as what it actually is (a full trace) and pairing it in the UI with the
  student's own round-1 trace for the same task id (already loaded from the baseline dir), via
  `_trace_for_task`/`_render_task_trace` in `control_room/app.py`. No producer script needed to
  change. Don't reintroduce a `{student, teacher}` pair field anywhere on the strength of an old
  UI assumption; match the real trace shape instead.

## Keep this file correct

This file corrects itself through you. **`AGENTS.md` is the only file that carries facts, traps
and decisions forward between sessions or agents - never create a second one** (a `notes.md`,
a `DECISIONS.md`, a `journal.md`, anything with a similar job) to hold this kind of thing instead.
One repo history had exactly that problem (a `docs/notes.md` research log duplicating and drifting
from this file) and was folded back in and deleted on 2026-11-11 for that reason - don't recreate
it under a different name. If a fact belongs somewhere more specific (`docs/setup/gcp-setup.md` and
the like, for setup/ops detail a coding agent doesn't need on every read), it still only lives in
one place: put it there, or here, and point to it from the other - never both, and never a third
file that duplicates either.

In the same change as your work:

1. **A fact here is wrong or out of date** (a command fails, an API changed, a rule moved):
   fix the line and add the date.
2. **A trap cost you more than about 10 minutes** and is not listed: add it to "Known traps"
   with the date and the fix.
3. **You made a decision that later agents must follow** (a frozen value, a file format, a split):
   add it to the section it belongs to, in this file.
4. **You find a duplicate or a contradiction** between this file and another doc: resolve it by
   the single-source rule above, in the same change.

Keep entries short: one fact per line. Delete a trap when the cause is gone.
Do not add task logs or status updates here: they go in pull requests.
