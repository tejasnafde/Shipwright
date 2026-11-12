"""Corrective training examples: the teacher's next action at the points where the candidate went wrong.

    .venv/bin/python scripts/build_corrections.py --failures runs/r4-cand-train --max-cost-usd 0.8

The problem statement is "generate corrective training examples". Teacher replays (scripts/teacher_runs.py) show how a
strong model solves a task from the start. They do not show the student what to do in the situations it gets itself
into, and those are where it fails: round 5's adapter repeated one lookup 8 times and exchanged a bookshelf for a camera.
So here, for every failed run of the candidate:

  1. Find the turns that were wrong, deterministically: the tool calls the guards (src/shipwright/llm/guards.py) would
     reject. The training data and the harness then agree on what "wrong" means, and no LLM judge is in the loop.
  2. Show the teacher the student's own conversation up to that turn (in the student's prompted format) and ask for its
     next reply.
  3. Keep the reply only if it passes the same guards. A text reply is always kept.

Each row is the student's conversation plus the teacher's reply, marked ``train_last_only``: the earlier assistant turns are
the student's mistakes and are context, never a target (scripts/train_lora.py masks them).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.heal.sft_data import trace_to_openai  # noqa: E402
from shipwright.llm import costs, guards, vertex  # noqa: E402
from shipwright.llm.context import CONTEXT_LIMIT, count_messages  # noqa: E402
from shipwright.llm.prompted_tools import convert_messages, format_tool_call, parse_tool_call  # noqa: E402
from shipwright.observability import log  # noqa: E402

TEACHER_REPLY_TOKENS = 4096
TEACHER_TEMPERATURE = 0.3


@dataclass
class WrongTurn:
    task_id: str
    simulation_id: str
    conversation_before: list[dict]  # OpenAI-format messages, up to but not including the wrong turn
    rule: str


def find_wrong_turns(trace: dict, per_rule_limit: int = 2) -> list[WrongTurn]:
    """The agent tool calls of a run that the guards would have rejected, at most ``per_rule_limit`` per rule (a loop
    breaks the same rule 8 times: the first few turns teach it, the rest repeat it)."""
    messages = trace_to_openai(trace["messages"])
    wrong_turns: list[WrongTurn] = []
    count_by_rule: dict[str, int] = {}
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        call = guards.tool_call_from_message(message["tool_calls"][0])
        violation = guards.first_violation(messages[:index], call)
        if violation is None or count_by_rule.get(violation.rule, 0) >= per_rule_limit:
            continue
        count_by_rule[violation.rule] = count_by_rule.get(violation.rule, 0) + 1
        wrong_turns.append(WrongTurn(str(trace.get("task_id")), str(trace.get("simulation_id")), messages[:index], violation.rule))
    return wrong_turns


def reply_is_acceptable(conversation_before: list[dict], reply_text: str) -> bool:
    """A text reply is kept. A tool call is kept only if the guards accept it in that same conversation."""
    if not reply_text.strip():
        return False
    parsed = parse_tool_call(reply_text)
    if parsed is None:
        return True
    return guards.first_violation(conversation_before, guards.ToolCall(name=parsed[0], arguments=parsed[1])) is None


def canonical_reply(reply_text: str) -> str:
    """A tool call as exactly one ```json block (the protocol: "reply with exactly one block and nothing else"): the teacher
    sometimes writes a sentence before it. A text reply is kept as written."""
    parsed = parse_tool_call(reply_text)
    if parsed is None:
        return reply_text.strip()
    return format_tool_call(parsed[0], parsed[1])


def correction_row(wrong_turn: WrongTurn, prompt: dict, reply_text: str, teacher_model: str) -> dict:
    reply_text = canonical_reply(reply_text)
    student_view = convert_messages([{"role": "system", "content": prompt["system"]}, *wrong_turn.conversation_before],
                                    prompt["tools"])
    return {
        "messages": [*student_view, {"role": "assistant", "content": reply_text.strip()}],
        "source": "correction",
        "task_id": wrong_turn.task_id,
        "simulation_id": wrong_turn.simulation_id,
        "window": 0,
        "model": teacher_model,
        "rule": wrong_turn.rule,
        "train_last_only": True,
    }


def ask_vertex_teacher(model: str, messages: list[dict]) -> tuple[str, dict]:
    """(the teacher's reply text, its token usage) for the student's prompted conversation."""
    import litellm

    response = vertex.call_with_refresh(
        litellm.completion,
        {"model": model, "messages": messages, "temperature": TEACHER_TEMPERATURE, "max_tokens": TEACHER_REPLY_TOKENS},
    )
    usage = getattr(response, "usage", None)
    return (response.choices[0].message.content or ""), {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0, "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}


@dataclass
class CorrectionStats:
    wrong_turns_found: dict[str, int] = field(default_factory=dict)
    accepted: dict[str, int] = field(default_factory=dict)
    rejected: int = 0
    too_long: int = 0
    cost_usd: float = 0.0
    stopped_on_budget: bool = False


def call_cost(model: str, usage: dict) -> float:
    price, _ = costs.price_for(model, "vertex")
    return (usage["prompt_tokens"] * price[0] + usage["completion_tokens"] * price[2]) / 1_000_000


def build_corrections(
    traces: list[dict],
    prompt_for_trace: Callable[[dict], dict],
    ask_teacher: Callable[[str, list[dict]], tuple[str, dict]],
    teacher_model: str,
    max_cost_usd: float,
    per_rule_limit: int = 2,
) -> tuple[list[dict], CorrectionStats]:
    rows: list[dict] = []
    stats = CorrectionStats()
    for trace in traces:
        if trace.get("passed") or trace.get("termination_reason") == "infrastructure_error":
            continue
        prompt = prompt_for_trace(trace)
        for wrong_turn in find_wrong_turns(trace, per_rule_limit):
            stats.wrong_turns_found[wrong_turn.rule] = stats.wrong_turns_found.get(wrong_turn.rule, 0) + 1
            if stats.cost_usd >= max_cost_usd:
                stats.stopped_on_budget = True
                continue
            student_view = convert_messages(
                [{"role": "system", "content": prompt["system"]}, *wrong_turn.conversation_before], prompt["tools"])
            reply_text, usage = ask_teacher(teacher_model, student_view)
            stats.cost_usd += call_cost(teacher_model, usage)
            if not reply_is_acceptable(wrong_turn.conversation_before, reply_text):
                stats.rejected += 1
                continue
            row = correction_row(wrong_turn, prompt, reply_text, teacher_model)
            if count_messages(row["messages"]) > CONTEXT_LIMIT:
                stats.too_long += 1
                continue
            rows.append(row)
            stats.accepted[wrong_turn.rule] = stats.accepted.get(wrong_turn.rule, 0) + 1
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--failures", type=Path, required=True, help="a run dir of the candidate on train tasks (traces.jsonl, prompts.json)")
    parser.add_argument("--teacher-model", default=vertex.TEACHER_MODEL)
    parser.add_argument("--max-cost-usd", type=float, required=True, help="stop asking the teacher past this (a hard cap)")
    parser.add_argument("--per-rule-limit", type=int, default=2, help="wrong turns used per rule and run")
    parser.add_argument("--output-dir", type=Path, default=None)
    options = parser.parse_args()

    traces = [json.loads(line) for line in (options.failures / "traces.jsonl").read_text().splitlines() if line.strip()]
    prompts = json.loads((options.failures / "prompts.json").read_text())
    only_prompt = next(iter(prompts.values()))

    def prompt_for_trace(trace: dict) -> dict:
        shas = {turn.get("prompt_sha") for turn in trace.get("agent_turns") or [] if turn.get("prompt_sha")}
        return prompts[next(iter(shas))] if len(shas) == 1 and next(iter(shas)) in prompts else only_prompt

    rows, stats = build_corrections(traces, prompt_for_trace, ask_vertex_teacher, options.teacher_model,
                                    options.max_cost_usd, options.per_rule_limit)
    output_directory = options.output_dir or (ROOT / "runs" / f"corrections-{options.failures.name}")
    output_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / "corrections.jsonl", "w") as corrections_file:
        for row in rows:
            corrections_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "failures_run": str(options.failures), "teacher_model": options.teacher_model, "rows": len(rows),
        "wrong_turns_found": stats.wrong_turns_found, "accepted": stats.accepted, "rejected_by_the_guards": stats.rejected,
        "too_long": stats.too_long, "cost_usd": round(stats.cost_usd, 4), "max_cost_usd": options.max_cost_usd,
        "stopped_on_budget": stats.stopped_on_budget,
    }
    (output_directory / "corrections_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="teacher", meta=True))
