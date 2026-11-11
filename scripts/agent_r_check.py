"""Check Agent-R revision runs (scripts/agent_r_revise.py output) before they become SFT data.

  .venv/bin/python scripts/agent_r_check.py runs/agent-r-revise-<ts>

Offline: no model calls, no network (tau2 replays the tool calls on a fresh DB). Per revision run:

- oracle: tau2's own environment evaluator (DB state and env assertions) on the spliced
  conversation, as if it were a live run. ``strict`` replays every tool call and fails when a
  recorded tool result differs from what the real DB returns (the splice-mismatch risk in
  docs/pipeline/agent-r.md, "Known limits"); ``lenient`` scores the final DB only. The
  action check (expected tool calls) is reported too.
- format: rendered as the student sees it (sft_data.revision_records); only system, user and
  assistant roles; every trained assistant turn that tries a tool call parses with the live
  prompted-tools parser.
- masks: system and user turns are masked; the masked assistant turns of the full
  conversation are exactly the shared prefix plus the bad steps (n_context_assistant); the
  first trained turn starts with the revision thought; every target turn is trained once
  across windows.
- tokens: every record fits the 8192-token context.

Writes check.jsonl (one row per revision) and check_summary.json into the revise dir.
scripts/build_sft.py --agent-r then keeps only the revisions marked valid there.
Exit 0 when at least one revision is valid on every check, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from shipwright.observability import log  # noqa: E402

logger = log.get_logger("agent_r_check")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("revise_dir", type=Path, help="output dir of scripts/agent_r_revise.py")
    p.add_argument("--estimate-tokens", action="store_true", help="3.5 chars per token, no tokenizer download")
    return p.parse_args(argv)


def oracle(rev: dict, task, domain: str) -> dict:
    """tau2 ENV (strict and lenient replay) and ACTION rewards of the spliced messages."""
    from datetime import datetime

    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    by_role = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    msgs = [by_role[m["role"]].model_validate(m) for m in rev["messages"] if m.get("role") in by_role]
    now = datetime.now().isoformat()
    sim = SimulationRun(id=f"check-{rev.get('bad_simulation_id')}", task_id=str(task.id), timestamp=now,
                        start_time=now, end_time=now, duration=0.0,
                        termination_reason=TerminationReason.USER_STOP, messages=msgs)
    scoring = CommunicateScoring(scorer=CommunicateScorer.SUBSTRING)
    out: dict = {}
    for name, etype, strict in (("env_strict", EvaluationType.ENV, True), ("env_lenient", EvaluationType.ENV, False),
                                ("action", EvaluationType.ACTION, False)):
        try:
            info = evaluate_simulation(sim, task, etype, solo_mode=False, domain=domain, strict_replay=strict,
                                       communicate_scoring=scoring)
            out[name] = float(info.reward)
        except Exception as e:  # noqa: BLE001 - a strict replay mismatch raises
            out[name] = 0.0
            out[f"{name}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def format_and_masks(rev: dict, prompt: dict) -> dict:
    """Checks on the SFT records of one revision run."""
    from shipwright.heal import sft_data
    from shipwright.llm.prompted_tools import looks_like_tool_attempt, parse_tool_call

    conv = sft_data.render_conversation(prompt["system"], prompt["tools"], sft_data.trace_to_openai(rev["messages"]))
    full_mask = sft_data.revision_mask(conv, rev["n_context_assistant"])
    n_assistant = sum(1 for m in conv if m["role"] == "assistant")
    masked_assistant = sum(1 for m, k in zip(conv, full_mask, strict=True) if m["role"] == "assistant" and k)
    first_target = next((m for m, k in zip(conv, full_mask, strict=True) if m["role"] == "assistant" and not k), None)
    records = sft_data.revision_records(rev, prompt)
    problems: list[str] = []
    trained_texts: list[str] = []
    for r in records:
        if r["messages"][0]["role"] != "system":
            problems.append("record does not start with the system prompt")
        for m, k in zip(r["messages"], r["loss_mask"], strict=True):
            if m["role"] not in ("system", "user", "assistant"):
                problems.append(f"role {m['role']}")
            if m["role"] != "assistant" and not k:
                problems.append(f"{m['role']} turn not masked")
            if m["role"] == "assistant" and not k:
                trained_texts.append(m["content"])
                if looks_like_tool_attempt(m["content"]) and parse_tool_call(m["content"]) is None:
                    problems.append("trained tool call does not parse")
    if masked_assistant != rev["n_context_assistant"]:
        problems.append(f"masked assistant turns {masked_assistant} != n_context_assistant {rev['n_context_assistant']}")
    thought = (rev.get("thought") or "").strip()
    if rev.get("format", "inline") == "inline" and (first_target is None or not first_target["content"].startswith(thought)):
        problems.append("first trained turn does not start with the revision thought")
    targets = n_assistant - masked_assistant
    if len(trained_texts) != targets:
        problems.append(f"trained turns across windows {len(trained_texts)} != targets {targets}")
    return {"records": len(records), "assistant_turns": n_assistant, "masked_assistant": masked_assistant,
            "target_turns": targets, "tokens_max": max((r["tokens"] for r in records), default=0),
            "reply_tokens_max": max((r["reply_tokens_max"] for r in records), default=0),
            "problems": sorted(set(problems))}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    d = args.revise_dir.resolve()
    log.attach_run_dir(d)
    from shipwright.llm import context

    import baseline

    if not args.estimate_tokens:
        context.load_tokenizer(context.STUDENT_REPO, baseline.load_token())
    from build_sft import reconstruct_prompt
    from tau2.registry import registry

    cfg = json.loads((d / "run_config.json").read_text())
    if cfg.get("task_split") != "train":
        raise SystemExit(f"{d}: task_split is {cfg.get('task_split')!r}, not 'train'.")
    domain = cfg.get("domain", "retail")
    tasks = {str(t.id): t for t in registry.get_tasks_loader(domain)(task_split_name="train")}
    prompts = json.loads((d / "prompts.json").read_text()) if (d / "prompts.json").exists() else {}
    fallback = next(iter(prompts.values())) if prompts else reconstruct_prompt()
    revs = [json.loads(x) for x in (d / "revisions.jsonl").read_text().splitlines() if x.strip()]
    rows = []
    for rev in revs:
        prompt = prompts.get(rev.get("prompt_sha")) or fallback
        row = {"task_id": rev["task_id"], "bad_simulation_id": rev.get("bad_simulation_id"),
               "good_simulation_id": rev.get("good_simulation_id"),
               "t": rev.get("t"), "t_prime": rev.get("t_prime"), "critic_found_bad": rev.get("critic_found_bad")}
        row.update(oracle(rev, tasks[str(rev["task_id"])], domain))
        row.update(format_and_masks(rev, prompt))
        row["fits_context"] = row["tokens_max"] <= context.CONTEXT_LIMIT
        row["valid"] = bool(row["env_strict"] == 1.0 and not row["problems"] and row["fits_context"])
        rows.append(row)
        logger.info("revision checked", **{k: v for k, v in row.items() if k != "problems"}, problems=row["problems"])
    with open(d / "check.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "revisions": len(rows),
        "valid": sum(r["valid"] for r in rows),
        "oracle_env_strict_pass": sum(r["env_strict"] == 1.0 for r in rows),
        "oracle_env_lenient_pass": sum(r["env_lenient"] == 1.0 for r in rows),
        "oracle_action_pass": sum(r["action"] == 1.0 for r in rows),
        "format_or_mask_problems": sum(bool(r["problems"]) for r in rows),
        "over_context": sum(not r["fits_context"] for r in rows),
        "problems": sorted({p for r in rows for p in r["problems"]}),
        "token_count_method": context.count_method(),
    }
    (d / "check_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="agent_r_check"))
