# The self-healing loop: one command, unattended (`shipwright`)

The whole loop is one command, `shipwright` (`scripts/loop.py` before `pip install -e .`; code in `src/shipwright/loop/`).
A real run is ONE Cloud Run execution that runs every stage itself with nobody watching, and ends in one of five
statuses. The laptop only checks, plans, launches and reads.

```sh
shipwright doctor                                                   # free preflight: tokens, configs, every command the loop builds
shipwright plan --config configs/rounds/r10-fixes.yaml              # free: what it would run and may spend
shipwright run  --config configs/rounds/r10-fixes.yaml --dry-run    # free, offline: fake models, fake HF jobs (CI runs it)
shipwright run  --config configs/rounds/r10-fixes.yaml --launch     # spends: runs doctor, then starts the cloud execution
shipwright status                                                   # the round manifests (after scripts/pull_runs.py)
shipwright show | rollback | promote <gate_decision.json>           # the deployed system
shipwright stage ceiling|smoke --config X --launch                  # side stages, by hand
```

`--launch` needs a clean, pushed HEAD and waits (up to 20 minutes) for the Cloud Build to point `shipwright-<owner>` at
the image of HEAD. A real run refuses to start on a laptop. The same command resumes a dead execution: every finished
stage is in the manifest and its run dir in GCS. `--redo <stage>` runs a finished stage of the first round again.

## What a round runs

The controller (`src/shipwright/loop/controller.py`) reads why the last gate refused and lets the recipe
(`src/shipwright/loop/recipe.py`, each rule with its evidence) choose the next remedy. Then it runs only what that needs:

| Round | Stages | About |
|---|---|---|
| fresh (no refused candidate yet) | baseline, classify, teacher, data, train, bfcl_screen, gate | $5 |
| BFCL fell: more general function-calling examples | data, train, bfcl_screen, gate | $2.3 |
| looping, overflow, pass^k: turn the tool guards on | gate (the same adapter) | $1 |
| still failing with guards: corrections (on-policy) | candidate_train, classify, corrections, data, train, bfcl_screen, gate | $3.5 |
| corrections improved the gate: another round of them | the same; at most 3 rounds, each trains on all earlier corrections | $3.5 |
| corrections did not improve the gate, or 3 done | stop (`exhausted`) | |

The stages (`src/shipwright/loop/stages.py`): `baseline` (Tiny Aya on the train tasks, HF router), `candidate_train` (the
refused candidate on the train tasks the teacher can solve, HF Job with vLLM), `classify` (AgentTune FailureDetector +
FailureClassifier + our rules), `teacher` (the teacher replays the failed tasks), `corrections` (the teacher's reply at
each turn the candidate got wrong, `scripts/build_corrections.py`), `data` (`scripts/build_sft.py`, pushed to the SFT repo
at branch round-N), `train` (AgentTune's RetrainingTrigger, then LoRA on an HF Job), `bfcl_screen` (BFCL alone, about 8
minutes: the gate is paid for only if its BFCL check can pass), `gate` (the candidate against the deployed system, 8
checks, `docs/pipeline/gate.md`). A promotion is recorded in the ledger; a refusal writes `next_round.json`.

## The config (`configs/rounds/<name>.yaml`)

One file pins the round number, owner, seed, every model, trials, concurrency, the teacher pool, training, the gate, the
stop rules and a spend cap per stage and in total. `git_sha` pins the commit (the run refuses another image). Optional:

- `recipe:` what the first round starts with (`tool_guards`, `general_replay`, `boost`, `corrections_rounds`): what
  earlier rounds already learned. Without it the first round is bare (gate 3's recipe).
- `start: {from_round: 8, candidate: configs/systems/r8-weight.yaml}`: continue from a gate that refused a candidate
  instead of a fresh round. Its `gate_decision.json` is read from the runs dataset.
- `teacher:` `skip_unsolved_from`, `skip_solved_from`, `only_solved_from` + `min_teacher_rate` (the harvest: a solved
  task passes again about 48% of the time, so replaying solved tasks is the cheapest way to more passing traces).
- `gate.candidate_extra:` serving-only switches for the candidate (`state_reminders: true`).

`shipwright doctor` and `shipwright run` refuse a config whose GPU worst case (dearest flavor x timeout) passes its cap, a
gate timeout under 60 minutes, or a missing section. The manifest records the config as run, the code sha, the image,
the library versions and the sha256 of `configs/splits.yaml` and `configs/gate.yaml`. Those two are frozen: the loop only
reads them.

## What keeps it running without anyone watching

- **Stop rules** (config `stop`): a failed stage (exit code, crash, an HF job that ended in ERROR, over 25% infra errors)
  is tried once more into the same output dir (tau2 `auto_resume` keeps decided runs). Not retried: under 40 SFT
  examples, AgentTune's RetrainingTrigger says no, the teacher gave no correction, the BFCL screen cannot pass.
- **Spend**: a stage starts only if its cap plus everything spent fits in `spend.total`. API stages get their cap as
  `--max-cost-usd` and meter their own tokens; GPU stages are priced by the job's running seconds (a queued job is not
  billed), and their cap must cover price x timeout of the dearest flavor they may fall back to.
- **GPU queue**: a job waiting over 12 minutes for a GPU is cancelled and the next flavor is tried; the last one waits.
- **Nothing to commit mid-run**: the candidate's system config is written into its round dir and packed into the gate's
  code bundle (`configs/systems/generated/`). The promotion ledger is kept in the runs dataset on the Hub
  (`promotions/ledger.jsonl`) and mirrored to `promotions/ledger.jsonl` and `configs/systems/current.yaml` here, so a
  promotion made in the cloud survives the container. Commit the mirror when convenient; the Hub copy is the record.
- **Resume**: each stage is written to `runs/rN-round/round.json` (mirrored to `runs/round-N.json` for the control room)
  as it finishes, and uploaded at once.

Statuses: `promoted`, `rejected` (no promotion in the rounds asked for), `stopped` (a stage failed twice; the same
command resumes), `over_budget`, `exhausted` (no remedy left). The run's summary is `runs/chain-rN/chain.json`.

## Outputs, round N

`runs/rN-baseline` (or `rN-cand-train`), `rN-teacher`, `rN-corrections`, `rN-sft`, `gate-rN`, and `rN-round/` with
`round.json`, `rN-weight.yaml` (the candidate), `gate_decision.json`, `metrics.jsonl`, and `next_round.json` on a refusal
or `ledger.jsonl` on a promotion. In the cloud they are in `gs://lexsi-hackathon-runs/<owner>/`; `scripts/pull_runs.py`
copies them here. Read them in the control room: `streamlit run src/shipwright/control_room/app.py`.

## By hand

Each stage is a script you can still run alone to debug (`scripts/baseline.py`, `classify.py`, `teacher_runs.py`,
`build_corrections.py`, `build_sft.py`, `round_status.py`, `train_job.py`, `gate_job.py`); the loop calls the same ones.
Held-out failures are never training data; the hidden split is opened once, for the final report
(`scripts/baseline.py --open-hidden`).
