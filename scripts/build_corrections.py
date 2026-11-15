"""Corrective training examples: the teacher's next action at the points where the candidate went wrong.

    .venv/bin/python scripts/build_corrections.py --failures runs/r4-cand-train --max-cost-usd 0.8

The problem statement is "generate corrective training examples". Teacher replays (scripts/teacher_runs.py) show how a
strong model solves a task from the start. They do not show the student what to do in the situations it gets itself
into, and those are where it fails: round 5's adapter repeated one lookup 8 times and exchanged a bookshelf for a camera.
So here, for every failed run of the candidate:

  1. Find the turns that were wrong, deterministically: the tool calls the guards (src/shipwright/llm/guards.py) would
     reject, and, in a run with the guards on, the replies they did reject (``guard_rejections`` in the trace). The training data and the harness then agree on what "wrong" means, and no LLM judge is in the loop.
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
from shipwright.llm.context import CONTEXT_LIMIT, count_messages, fit_messages  # noqa: E402
from shipwright.llm.prompted_tools import convert_messages, format_tool_call, parse_tool_call  # noqa: E402
from shipwright.observability import log  # noqa: E402

logger = log.get_logger("build_corrections")

TEACHER_REPLY_TOKENS = 4096
TEACHER_TEMPERATURE = 0.3


@dataclass
class WrongTurn:
    task_id: str
    simulation_id: str
    conversation_before: list[dict]  # OpenAI-format messages, up to but not including the wrong turn
    rule: str


def wrong_turn_positions(trace: dict) -> list[tuple[int, str]]:
    """(index in the tau2 messages, guard rule) of every wrong agent turn of a run, in conversation order.

    Two kinds. A run without the guards keeps its wrong tool calls in the history: the guards are asked about each one.
    A run with the guards on never has them in the history: the guards rejected them and the reply that was rejected is
    in that turn's ``guard_rejections`` (round 9: 8 wrong calls left in the history, 307 rejected ones). Those are the
    candidate's own mistakes at its own states, so they are the corrections that matter most."""
    tau2_messages = trace["messages"]
    positions: dict[int, str] = {}
    for turn in trace.get("agent_turns") or []:
        rejections = turn.get("guard_rejections") or []
        if rejections and turn.get("message_index") is not None:
            positions[int(turn["message_index"])] = rejections[0]["rule"]
    for index, message in enumerate(tau2_messages):
        if index in positions or message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        history = trace_to_openai(tau2_messages[:index])
        call = guards.tool_call_from_message(trace_to_openai([message])[0]["tool_calls"][0])
        violation = guards.first_violation(history, call)
        if violation is not None:
            positions[index] = violation.rule
    return sorted(positions.items())


def find_wrong_turns(trace: dict, per_rule_limit: int = 2) -> list[WrongTurn]:
    """The agent turns of a run that the guards rejected or would have rejected, at most ``per_rule_limit`` per rule (a
    loop breaks the same rule 8 times: the first few turns teach it, the rest repeat it)."""
    wrong_turns: list[WrongTurn] = []
    count_by_rule: dict[str, int] = {}
    for index, rule in wrong_turn_positions(trace):
        if count_by_rule.get(rule, 0) >= per_rule_limit:
            continue
        count_by_rule[rule] = count_by_rule.get(rule, 0) + 1
        conversation_before = trace_to_openai(trace["messages"][:index])
        wrong_turns.append(WrongTurn(str(trace.get("task_id")), str(trace.get("simulation_id")), conversation_before, rule))
    return wrong_turns


def student_view(prompt: dict, conversation_before: list[dict]) -> list[dict]:
    """What the student was sent before the wrong turn: its prompted format, trimmed to the 8192-token budget the same
    way the live request is (``context.fit_messages``). Without the trim, a long looping run (the runs that need a
    correction most) gives a row over the limit, and the row is dropped."""
    converted = convert_messages([{"role": "system", "content": prompt["system"]}, *conversation_before], prompt["tools"])
    fitted, _ = fit_messages(converted)
    return fitted


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
    return {
        "messages": [*student_view(prompt, wrong_turn.conversation_before), {"role": "assistant", "content": reply_text.strip()}],
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
    teacher_errors: int = 0  # calls that still failed after vertex.call_with_refresh's retries: skipped, not fatal
    answers_reused: int = 0  # answers paid for by an earlier, interrupted run (asked.jsonl)
    cost_usd: float = 0.0
    stopped_on_budget: bool = False


def call_cost(model: str, usage: dict) -> float:
    price, _ = costs.price_for(model, "vertex")
    return (usage["prompt_tokens"] * price[0] + usage["completion_tokens"] * price[2]) / 1_000_000


def answer_key(wrong_turn: WrongTurn) -> str:
    """Names one wrong turn across runs of this script: the run and how far into it the turn is."""
    return f"{wrong_turn.simulation_id}:{len(wrong_turn.conversation_before)}"


@dataclass
class SavedAnswer:
    reply_text: str
    usage: dict


def build_corrections(
    traces: list[dict],
    prompt_for_trace: Callable[[dict], dict],
    ask_teacher: Callable[[str, list[dict]], tuple[str, dict]],
    teacher_model: str,
    max_cost_usd: float,
    per_rule_limit: int = 2,
    saved_answers: dict[str, SavedAnswer] | None = None,
    save_answer: Callable[[str, SavedAnswer], None] | None = None,
) -> tuple[list[dict], CorrectionStats]:
    """``saved_answers`` are the teacher's answers from an earlier run that stopped midway (round 9: a 429 after 30 minutes
    lost every answer): they are used again, not paid for again. ``save_answer`` is called right after each paid answer."""
    saved_answers = saved_answers if saved_answers is not None else {}
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
            key = answer_key(wrong_turn)
            if key in saved_answers:
                answer = saved_answers[key]
                stats.answers_reused += 1
            else:
                try:
                    reply_text, usage = ask_teacher(teacher_model, student_view(prompt, wrong_turn.conversation_before))
                except Exception as call_error:  # noqa: BLE001 - one failed call must not lose the answers already paid for
                    stats.teacher_errors += 1
                    logger.warning("teacher call failed, wrong turn skipped", key=key, error=str(call_error)[:200])
                    continue
                answer = SavedAnswer(reply_text, usage)
                if save_answer is not None:
                    save_answer(key, answer)
            reply_text = answer.reply_text
            stats.cost_usd += call_cost(teacher_model, answer.usage)
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
    # The student's view is rebuilt the way the candidate saw it: a candidate that served the short policy is corrected
    # on the short policy (prompts.json keeps tau2's own text, converted at request time).
    run_config_path = options.failures / "run_config.json"
    if run_config_path.exists() and json.loads(run_config_path.read_text()).get("condensed_policy"):
        from shipwright.llm import prompted_tools

        prompted_tools.set_condensed_policy(True)
    if run_config_path.exists() and json.loads(run_config_path.read_text()).get("identity_note"):
        from shipwright.llm import context

        context.set_identity_note(True)

    traces =[json.loads(line) for line in (options.failures / "traces.jsonl").read_text().splitlines() if line.strip()]
    prompts = json.loads((options.failures / "prompts.json").read_text())
    only_prompt = next(iter(prompts.values()))

    def prompt_for_trace(trace: dict) -> dict:
        shas = {turn.get("prompt_sha") for turn in trace.get("agent_turns") or [] if turn.get("prompt_sha")}
        return prompts[next(iter(shas))] if len(shas) == 1 and next(iter(shas)) in prompts else only_prompt

    output_directory = options.output_dir or (ROOT / "runs" / f"corrections-{options.failures.name}")
    output_directory.mkdir(parents=True, exist_ok=True)
    asked_path = output_directory / "asked.jsonl"
    saved_answers: dict[str, SavedAnswer] = {}
    if asked_path.exists():
        for line in asked_path.read_text().splitlines():
            if line.strip():
                saved = json.loads(line)
                saved_answers[saved["key"]] = SavedAnswer(saved["reply_text"], saved["usage"])

    def save_answer(key: str, answer: SavedAnswer) -> None:
        with open(asked_path, "a") as asked_file:
            asked_file.write(json.dumps({"key": key, "reply_text": answer.reply_text, "usage": answer.usage}, ensure_ascii=False) + "\n")

    rows, stats = build_corrections(traces, prompt_for_trace, ask_vertex_teacher, options.teacher_model,
                                    options.max_cost_usd, options.per_rule_limit, saved_answers, save_answer)
    with open(output_directory / "corrections.jsonl", "w") as corrections_file:
        for row in rows:
            corrections_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "failures_run": str(options.failures), "teacher_model": options.teacher_model, "rows": len(rows),
        "wrong_turns_found": stats.wrong_turns_found, "accepted": stats.accepted, "rejected_by_the_guards": stats.rejected,
        "too_long": stats.too_long, "teacher_errors": stats.teacher_errors, "answers_reused": stats.answers_reused,
        "cost_usd": round(stats.cost_usd, 4), "max_cost_usd": options.max_cost_usd,
        "stopped_on_budget": stats.stopped_on_budget,
    }
    (output_directory / "corrections_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="teacher", meta=True))
