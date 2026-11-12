"""General function-calling examples to mix into the SFT data, so the adapter keeps its general tool-calling skill.

    .venv/bin/python scripts/build_bfcl_replay.py                       # 100 of each category to runs/bfcl-replay/

Why: after training on retail conversations only, the adapter answers in words in every task ("who are you?") even
when a function should be called. The gate's BFCL check fell from 43 to 35 (round 3) and to 34 (round 4):
simple 67 to 4, multiple 58 to 7, irrelevance 4 to 90. Training less did not change that. The standard fix is to
mix general examples into the training data.

Where the examples come from: the BFCL v3 categories the gate uses (simple, multiple, irrelevance), at the gate's
pinned revision, but ONLY samples after the first ``bfcl.limit_per_category`` of each category. The gate scores the
first 100 of each (AuditKIT ``load_bfcl(limit=100)`` takes the first N in file order), so the replay set and the
gate's set never overlap; ``tests/test_bfcl_replay.py`` checks that against the real data. BFCL has no official
train split: say so in the write-up.

Each example is rendered exactly like the gate's BFCL request (``bfcl_eval.sample_messages``: the prompted-tools
system prompt with the sample's tools), then the reply the benchmark accepts: for simple and multiple the first
accepted value of every argument as a ```json call, for irrelevance a plain-text reply that calls nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.heal import bfcl_eval  # noqa: E402
from shipwright.llm.prompted_tools import format_tool_call  # noqa: E402
from shipwright.observability import log  # noqa: E402

# Replies for a question no available function can answer. Several, so the model learns the behaviour and not a sentence.
NO_CALL_REPLIES = (
    "None of the available functions can answer that, so I will not call any of them. {names} cannot help with this request.",
    "I can't answer this with the tools I have ({names}): none of them does what you ask, so I won't call one.",
    "These functions ({names}) don't fit this request, so no function call is needed here. Could you tell me more about what you need?",
)


def accepted_call(expected_tool_calls: list) -> dict | None:
    """The first expected call with the first accepted value of every argument. Optional arguments whose accepted
    values include an empty string are left out."""
    if not expected_tool_calls or not expected_tool_calls[0]:
        return None
    first_call = expected_tool_calls[0][0]
    arguments = {}
    for argument_name, accepted_values in (first_call.get("arguments") or {}).items():
        values = accepted_values if isinstance(accepted_values, list) else [accepted_values]
        if "" in values:
            continue
        arguments[argument_name] = values[0]
    return {"name": first_call["name"], "arguments": arguments}


def replay_row(sample, position: int) -> dict | None:
    """One SFT record (the shape of scripts/build_sft.py records) for one BFCL sample, or None if it has no usable answer."""
    category = sample.metadata["bfcl_category"]
    messages = bfcl_eval.sample_messages(sample)
    if category == "irrelevance":
        tool_names = ", ".join(tool["function"]["name"] for tool in (sample.tools or []))
        reply = NO_CALL_REPLIES[position % len(NO_CALL_REPLIES)].format(names=tool_names)
    else:
        call = accepted_call(sample.expected_tool_calls)
        if call is None:
            return None
        reply = format_tool_call(call["name"], call["arguments"])
    sample_id = sample.metadata["bfcl_id"]
    return {
        "messages": [*messages, {"role": "assistant", "content": reply}],
        "source": "general_replay",
        "task_id": sample_id,
        "simulation_id": sample_id,
        "window": 0,
        "model": "bfcl-accepted-answer",
    }


def build_rows(samples_by_category: dict[str, list], skip: int, per_category: int) -> list[dict]:
    """The rows: ``per_category`` samples of each category, taken after the first ``skip`` (the gate's)."""
    rows = []
    for category, samples in samples_by_category.items():
        taken = 0
        for position, sample in enumerate(samples[skip:]):
            if taken == per_category:
                break
            row = replay_row(sample, position)
            if row is not None:
                rows.append(row)
                taken += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gate.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "bfcl-replay")
    options = parser.parse_args()

    bfcl_config = yaml.safe_load(options.config.read_text())["bfcl"]
    categories = list(bfcl_config["categories"])
    gate_samples_per_category = int(bfcl_config["limit_per_category"])
    root = bfcl_eval.download(bfcl_config["revision"], categories)

    import auditkit

    samples_by_category = {}
    for category in categories:
        questions_path, answers_path = bfcl_eval.bfcl_files(root, category)
        samples = auditkit.load_bfcl(
            str(questions_path), str(answers_path) if answers_path else None, revision=bfcl_config["revision"], limit=None)
        for sample in samples:
            sample.metadata["bfcl_category"] = category
        samples_by_category[category] = samples

    rows = build_rows(samples_by_category, gate_samples_per_category, options.per_category)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    with open(options.output_dir / "replay.jsonl", "w") as replay_file:
        for row in rows:
            replay_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "revision": bfcl_config["revision"], "rows": len(rows),
        "per_category": {category: sum(1 for row in rows if row["task_id"].startswith(category)) for category in categories},
        "skipped_first_per_category": gate_samples_per_category,
        "ids": [row["task_id"] for row in rows],
    }
    (options.output_dir / "replay_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "ids"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="sft", meta=True))
