# The weight fix: teacher data and LoRA training

Failures routed `weight_fix` are fixed in the weights: a teacher shows the right behaviour, the examples are rendered
exactly as the student sees its requests, and a LoRA adapter is trained on HF Jobs with AgentTune. The loop runs these
as the `teacher`, `corrections`, `data` and `train` stages; each is also a script that runs alone.

## Three sources of examples

| Source | Script | What it teaches |
|---|---|---|
| Teacher replay | `scripts/teacher_runs.py` | whole passing conversations on the train tasks the student failed |
| On-policy corrections | `scripts/build_corrections.py` | the right next turn at each point the current candidate went wrong |
| General function calling | `scripts/build_bfcl_replay.py` | BFCL samples outside the ones the gate scores, so narrow training does not erase general tool use |

**Teacher replay.** The teacher plays the agent with the student's prompted view (`prompted_tools.force_prompted`):
the same compact tool list, reply protocol and 8,192-token prompt budget; only its reply budget is larger, because a
reasoning model spends tokens before it answers. Only passing runs (`kept.jsonl`) become data. A task the teacher
solved before passes again about half the time, so the harvest modes in the round config (`teacher.only_solved_from`,
`skip_unsolved_from`, `skip_solved_from`) aim spend at the tasks that pay. The teacher must allow training on its
outputs.

**On-policy corrections.** The refused candidate runs the train tasks with the guards on (`candidate_train`). Every
turn a guard rejected is a wrong turn; the teacher is shown the candidate's view at that point (trimmed the same way)
and gives the right next reply. A correction row trains only that final reply (`train_last_only`), never the
candidate's own earlier turns. Answers are cached (`asked.jsonl`), so a rerun does not pay twice. Each correction
round trains on all earlier rounds' corrections.

**General function calling.** 100 BFCL v3 samples per category, taken after the first 100 that the gate scores
(`tests/test_bfcl_replay.py` checks they never overlap).

## Building the dataset

`scripts/build_sft.py` renders every example with the recorded prompt (`prompts.json`) and the same conversion the live
request uses, then filters:

- train-split runs only (`run_config.json`);
- no conversation with a leaked user-simulator turn;
- no correction whose view had lost the customer's identity;
- no reply longer than the student's 512-token reply budget, no row over the context;
- every row is rendered once with Tiny Aya's real chat template before it is accepted.

A conversation longer than the window is cut into the windows the student would see. `--boost` repeats the examples
of tasks the candidate failed. The dataset is pushed to `mitushigoyal5/shipwright-sft` on branch `round-N`, with
`sft_stats.json`.

## Training

`scripts/train_job.py` launches `scripts/train_lora.py` on an HF Job (`l40sx1` by default; it prints price x timeout
before launching and always passes a timeout). The training script:

- turns each row into an AgentTune `EventLog` (user turns are observations, assistant turns are trained text) and calls
  `agenttune.create_distill_trainer`, checking that the rows AgentTune builds equal the dataset;
- passes its own trainer factory: labels are masked outside the assistant turns (and outside the last assistant turn
  of a correction row), the end-of-turn token is trained, and the loss is computed only over the trained window
  (`logits_to_keep`), because Tiny Aya's 262,144-token vocabulary makes full logits the memory bottleneck;
- compares its first-step loss with the model's own `labels=` loss and refuses to train if they differ;
- trains LoRA r 16, alpha 32 on all attention and MLP projections. The promoted recipe: a fresh adapter from the base
  model, 2 epochs, learning rate 1e-4;
- pushes the adapter, `training_meta.json`, AgentTune's lineage record `lexsi_provenance.json` and a model card to
  `mitushigoyal5/shipwright-lora` on branch `round-N`.

A sound first step on real rows is a loss of a few nats that equals the model's own; a loss near ln(262,144) = 12.5
means the labels are wrong.

## By hand

```sh
.venv/bin/python scripts/teacher_runs.py --dry-run --teacher-model vertex/deepseek-ai/deepseek-v4.1-flash-maas \
    --user-model openai/openai/gpt-oss-120b:deepinfra --task-ids 0 1 --trials 1 --output-dir runs/teacher-dry
.venv/bin/python scripts/build_sft.py --teacher runs/<teacher run> --output-dir runs/sft-dry --estimate-tokens
.venv/bin/python scripts/train_job.py --dataset-revision round-N --revision round-N            # plan and cost only
.venv/bin/python scripts/train_job.py --dataset-revision round-N --revision round-N --launch   # paid GPU
```
