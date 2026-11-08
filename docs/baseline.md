# Baseline: Tiny Aya Global on tau2-bench retail

Runs Tiny Aya Global as the tool-calling agent in tau2-bench, with no local GPU.
Both the agent and the user simulator go through Hugging Face Inference Providers
(OpenAI-compatible router `https://router.huggingface.co/v1`).

- Agent: `CohereLabs/tiny-aya-global:cohere` (litellm id `openai/CohereLabs/tiny-aya-global:cohere`)
- User simulator: `openai/gpt-oss-120b:cerebras` (litellm id `openai/openai/gpt-oss-120b:cerebras`)
- tau2-bench: pinned commit in `scripts/setup_tau2.sh`, cloned to `third_party/` (git-ignored)

## Setup (once)

```sh
./scripts/setup_tau2.sh
```

## Token

The scripts read the HF token at run time from `$SWITCHBOARD_SECRETS_DIR/HF_TOKEN`,
or from `$HF_TOKEN`. The token goes to litellm through process env vars only. It is never written to disk.

## Run

```sh
# 1. Offline check of the whole pipeline (fake LLM, no network, no token)
.venv/bin/python scripts/baseline.py --dry-run --num-tasks 2 --output-dir runs/dry

# 2. Does Tiny Aya return native tool calls through the router?
.venv/bin/python scripts/check_tool_calling.py

# 3. Cost check on 5 tasks, then the 20-task baseline
.venv/bin/python scripts/baseline.py --num-tasks 5
.venv/bin/python scripts/baseline.py --num-tasks 20
```

Other flags: `--task-split train|test|base` (default `train`, so held-out tasks stay unused),
`--task-ids 0 1 2`, `--trials`, `--agent-model`, `--user-model`, `--max-concurrency` (default 4),
`--output-dir` (default `runs/baseline-<timestamp>/`).

## Output (in the run dir, git-ignored)

- `tau2/results.json`: tau2's own results file (`tau2 view` can read it).
- `traces.jsonl`: one line per task run. Task id, reward, pass, termination reason, the full message
  list, and per agent turn: raw model output text, reasoning (if any), raw tool calls as the model
  wrote them, parsed tool calls, tool results, and error flags.
- `summary.json`: pass rate, tool calls, malformed tool calls, unknown tool names, tool errors,
  tool calls written as text, average turns, token usage, a cost estimate.
- `run_config.json`: the arguments of the run.

## Malformed tool calls

tau2 runs `json.loads` on every tool-call argument string, so one bad call would crash the
simulation. `src/shipwright/llm_patch.py` wraps the litellm call: invalid arguments become
`{"__malformed_arguments__": "<original text>"}`. The environment then returns a tool error
to the agent and the run goes on. The trace shows the original text.

## Cost

The cost in `summary.json` is an estimate from token counts and assumed prices
(`PRICES` in `scripts/baseline.py`). Check HF billing for the real number.
