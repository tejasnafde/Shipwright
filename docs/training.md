# Weight fix: teacher data and LoRA training

The weight-fix path for failures that `scripts/classify.py` routes to `weight_fix`:

1. **Teacher runs** (`scripts/teacher_runs.py`): a strong model plays the agent on train tasks.
2. **SFT data** (`scripts/build_sft.py`, `src/shipwright/sft_data.py`): its successful runs,
   rendered exactly as Tiny Aya sees them.
3. **Training** (`scripts/train_job.py` launches `scripts/train_lora.py` on HF Jobs): a LoRA on
   Tiny Aya Global with AgentTune, pushed to the Hub on a round branch.

Nothing here spends credits unless you run a command that says so (teacher runs call Vertex AI
or the HF router; `build_sft.py --push` writes to the Hub; `train_job.py --launch` starts a paid GPU job).

## The teacher sees the student's prompt

`scripts/teacher_runs.py` calls `prompted_tools.force_prompted(teacher)`, so the teacher gets the
same request as Tiny Aya: no native tools, the compact tool list and the ```json block protocol
in the system prompt, tool results as user turns, and the same 8192-token prompt budget
(`context.fit_messages`). Only its reply budget is larger (`--reply-tokens`, default 4096),
because a reasoning teacher spends tokens before it answers. Replies over 700 tokens (the
student's budget) are dropped in `build_sft.py`.

The teacher (`--teacher-model`) and the user simulator (`--user-model`) are required arguments,
with no default. The teacher must allow training on its outputs (no closed Claude or GPT models).

**Team choice (2026-11-09):** teacher `vertex/deepseek-ai/deepseek-v4.1-flash-maas`, user simulator
`vertex/openai/gpt-oss-120b-maas`, both on Vertex AI (GCP budget, not HF credits; setup and auth in
`docs/baseline.md`, "Vertex AI models"). Other ids still work, for example on the HF router
`openai/openai/gpt-oss-120b:cerebras` (pass `--reasoning-effort medium`) or
`openai/CohereLabs/aya-expanse-32b:cohere`.

The teacher's requests start with the same system message (policy plus tool list) on every call,
so Vertex implicit prefix caching can apply; `summary.json` and `teacher_summary.json` report the
cache hit rate, and the cost estimate prices cached tokens at $0.03 per 1M ("Prompt caching" in
`docs/baseline.md`).

If the teacher is the same model as the user simulator, that is fine for retail: the user
simulator has no tools, so it never gets the prompted view.

## Record of the agent prompt

Every prompted request records its un-converted system prompt and tool schemas
(`prompted_tools.recorded_prompts()`, key `prompt_sha`, also in each trace turn).
`scripts/baseline.py` writes them to `prompts.json` in the run dir. `build_sft.py` renders
training data from it with `convert_messages`, the same function the live request uses. For
older runs without `prompts.json` it rebuilds the prompt from tau2 (policy plus the English
language guidance); this rebuild was checked equal to a recorded prompt on 2026-11-08.

## SFT format

`sft.jsonl`, one example per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "assistant", "content": "Hi! ..."},
              {"role": "user", "content": "..."}, {"role": "assistant", "content": "```json\n{...}\n```"}, ...],
 "source": "teacher", "task_id": "3", "simulation_id": "...", "window": 0,
 "tokens": 4120, "reply_tokens_max": 96}
```

- Roles are only system, user and assistant (Tiny Aya's template). Tool calls are the ```json
  block from `format_tool_call`; tool results are `Tool result for <name>: <minified JSON>`.
- Every assistant message is a target. System and user messages are context (loss-masked).
- A run whose whole conversation fits the prompt budget is one example. A longer run is cut into
  windows (`--long-mode window`, default): each window is the trimmed prompt Tiny Aya would get
  before one of its turns, plus that turn; each assistant turn is a target once.
  `--long-mode drop` drops long runs instead.
- Duplicates (same messages) are removed. Examples over `--max-tokens` (8192) are dropped.
- `--replay RUN_DIR` mixes in Tiny Aya's own successful runs (`source: replay`).
- Only runs from the train split are accepted (`run_config.json`); `--allow-split` overrides.
- `sft_stats.json`: examples, per source, tasks, assistant turns, average and max tokens, drops.

## Training with AgentTune

`scripts/train_lora.py` (PEP 723 script, `agenttune==1.1.3`, which pins `trl==1.7.1` and
`transformers>=5.15`):

- Each example becomes a full-tier AgentTune `EventLog`: user turns are `OBSERVATION` events,
  assistant turns are `TEXT` events. AgentTune's `EventLog` has no system event, so a small
  subclass (`SystemPromptEventLog`) puts the system prompt back as the first, masked message of
  its SFT row.
- `agenttune.create_distill_trainer(student, trajectories=logs, trainer_factory=...)` runs
  `Project.distill`. The script checks that the rows AgentTune builds equal `sft.jsonl`.
- Our `trainer_factory` (`MaskedSFTFactory`) tokenizes each row with Tiny Aya's chat template and
  sets labels to -100 outside the assistant turns, from AgentTune's `loss_mask`. The assistant
  end-of-turn token is trained, so the model learns to stop. It returns a TRL `SFTTrainer` with
  LoRA (r 16, alpha 32, dropout 0.05, all attention and MLP projections), lr 2e-4 cosine,
  3 epochs, bf16, batch 1 x grad accum 8, max_seq_len 8192, gradient checkpointing.
  AgentTune's built-in factory trains on every token, and TRL's `assistant_only_loss` depends on
  chat template generation markers (not verified for Tiny Aya), so we pass our own factory.
- The adapter, `training_meta.json` and a model card go to `mitushigoyal5/shipwright-lora`,
  branch `--revision` (e.g. `round-1`).

## Commands

```sh
# 0. Offline checks (no network, no credits)
.venv/bin/python -m pytest -q
.venv/bin/python scripts/teacher_runs.py --dry-run --teacher-model vertex/deepseek-ai/deepseek-v4.1-flash-maas \
    --user-model vertex/openai/gpt-oss-120b-maas --task-ids 0 1 --trials 1 --output-dir runs/teacher-dry

# 1a. Vertex access check (2 tiny calls, GCP billing)
.venv/bin/python scripts/check_vertex.py

# 1b. Teacher runs (Vertex: spends GCP budget). Default tasks: the weight_fix ids of the baseline run.
#     Teammates: export SHIPWRIGHT_GCLOUD_CONFIG=lexsi first.
.venv/bin/python scripts/teacher_runs.py --teacher-model vertex/deepseek-ai/deepseek-v4.1-flash-maas \
    --user-model vertex/openai/gpt-oss-120b-maas --trials 3 --from-run runs/baseline-20261108-214337
#    add --all-train to also run every train task; outputs kept.jsonl and teacher_summary.json

# 2. Build the SFT data (downloads only the Tiny Aya tokenizer; --estimate-tokens avoids that)
.venv/bin/python scripts/build_sft.py --teacher runs/teacher-<stamp> \
    --replay runs/baseline-20261108-214337 --output-dir runs/sft-round-1

# 3. Push the dataset (writes to the Hub)
.venv/bin/python scripts/build_sft.py --teacher runs/teacher-<stamp> \
    --replay runs/baseline-20261108-214337 --output-dir runs/sft-round-1 \
    --push --revision round-1

# 4. Training job: plan and cost first, then launch (paid GPU)
.venv/bin/python scripts/train_job.py --dataset-revision round-1 --revision round-1
.venv/bin/python scripts/train_job.py --dataset-revision round-1 --revision round-1 --launch
#    fallback flavor: --flavor a10g-small; extra train args after "--", e.g. -- --epochs 2
hf jobs logs <job id>      # follow
hf jobs cancel <job id>    # stop early
```

`train_job.py` prints the flavor price times the timeout (worst case) before it launches.
Default `l4x1` at $0.80/h with a 2 h timeout: at most $1.60. The timeout is always passed:
HF Jobs stop after 30 minutes by default.

Memory risk: Tiny Aya's vocabulary is large, so the logits of an 8192-token row
take several GB. If the job runs out of memory on 24 GB, lower `--max-seq-len` (rows over it are
dropped and counted) or use a larger flavor with `--price-per-hour`.
