"""Agent-R step 2: revision trajectories from an MCTS search run (arxiv 2501.11425).

Reads the traces of scripts/agent_r_search.py and, per task:

1. Pairs (Eq. 4): every failed leaf (reward at least --beta below a good one) gets the good
   leaf (reward >= --alpha) with the longest shared prefix. They diverge at step t.
2. Transition point t' (--transition):
     model       the actor judges each bad step from t on, Good / Bad / Uncertain, and t'
                 is the first Bad one (Agent-R's model-guided critique; default)
     direct      t' is the last bad step (the paper's Direct-Revision baseline)
     divergence  t' = t (no model calls; a cheap ablation of ours)
   The critic is the actor itself (--critic-model, default the search's actor, on the
   same endpoint), "within its current capability" as in the paper.
3. Revision trajectory (Eq. 3): shared prefix + bad steps t..t' + a revision thought +
   good steps t..end. The prefix and the bad steps are context (loss-masked); the thought
   and the good suffix are trained (Eq. 6). --revision-format inline (default) puts the
   thought in front of the first good action; turn uses the paper's extra turn pair.

Writes OUT_DIR/
  revisions.jsonl        one revision run per pair (tau2 messages + n_context_assistant)
  good.jsonl             the distinct successful leaves (Eq. 6 trains on them too)
  critic.jsonl           every critic judgment (raw text kept)
  revise_summary.json    pairs, t' found by the critic, fallbacks, per task
  run_config.json, prompts.json   copied from the search run (build_sft checks the split)
Then: scripts/build_sft.py --agent-r OUT_DIR.

--transition model calls the critic (HF router credits, or the vLLM endpoint): at most one
call per bad step after the divergence point. --dry-run uses a rule-based fake critic.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from shipwright.heal import agent_r  # noqa: E402
from shipwright.observability import log  # noqa: E402

logger = log.get_logger("agent_r_revise")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("search_dir", type=Path, help="run dir of scripts/agent_r_search.py")

    # Which leaves count as good and bad.
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        choices=sorted(agent_r.ALPHA_BY_ITERATION),
        help="sets --alpha to the paper's schedule (0.5, 0.7, 1.0)",
    )
    parser.add_argument("--alpha", type=float, default=None, help="good leaf: reward >= alpha (tau2 rewards are 0 or 1)")
    parser.add_argument("--beta", type=float, default=agent_r.BETA, help="good reward - bad reward >= beta")
    parser.add_argument("--max-pairs-per-task", type=int, default=10)
    parser.add_argument("--max-good-per-task", type=int, default=3)

    # How the first bad step is found, and how the revision is written.
    parser.add_argument("--transition", choices=["model", "direct", "divergence"], default="model")
    parser.add_argument("--revision-format", choices=["inline", "turn"], default="inline")
    parser.add_argument("--critic-model", default=None, help="default: the search's actor model")
    parser.add_argument("--critic-api-base", default=None, help="default: the search's --agent-api-base")

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="rule-based fake critic, no network")
    return parser.parse_args(argv)


def load_search(search_directory: Path) -> tuple[dict[str, list[dict]], dict, dict]:
    """Traces by task (only rollouts with an mcts path), run_config, prompts."""
    traces_by_task_id: dict[str, list[dict]] = defaultdict(list)
    with open(search_directory / "traces.jsonl") as traces_file:
        for line in traces_file:
            trace = json.loads(line)
            if trace.get("mcts") and trace.get("reward") is not None:
                traces_by_task_id[str(trace["task_id"])].append(trace)

    run_config = json.loads((search_directory / "run_config.json").read_text())

    prompts_path = search_directory / "prompts.json"
    if prompts_path.exists():
        prompts_by_sha = json.loads(prompts_path.read_text())
    else:
        prompts_by_sha = {}
    return traces_by_task_id, run_config, prompts_by_sha


def policy_text(trace: dict, prompts_by_sha: dict) -> str:
    """The actor's own system prompt (the policy), for the critic."""
    prompt_shas = [turn.get("prompt_sha") for turn in trace.get("agent_turns", []) if turn.get("prompt_sha")]
    if prompt_shas and prompt_shas[-1] in prompts_by_sha:
        return prompts_by_sha[prompt_shas[-1]]["system"]
    if len(prompts_by_sha) == 1:
        return next(iter(prompts_by_sha.values()))["system"]
    return ""


def make_critic_chat(model: str, api_base: str | None):
    """chat(messages) -> text for the actor as critic: vLLM endpoint, Vertex, or the HF router."""
    from baseline import HF_ROUTER, load_token
    from shipwright.llm import vertex
    from shipwright.observability import instrument

    request_model = model.removeprefix("openai/")
    if api_base:
        url = f"{api_base.rstrip('/')}/chat/completions"

        def make_headers() -> dict:
            return {}
    else:
        if vertex.is_vertex_model(model):
            endpoint_model = model
        else:
            endpoint_model = request_model
        url, request_model, make_headers = vertex.endpoint(endpoint_model, load_token(), HF_ROUTER)
    http_client = httpx.Client(timeout=120)

    def chat(messages: list[dict]) -> str:
        # 400 reply tokens: the critic reasons briefly, then the Judgment line (Tiny Aya's
        # context is 8192 and critic_messages keeps the prompt near 5k tokens).
        request_body = {"model": request_model, "messages": messages, "temperature": 0, "max_tokens": 400}
        with instrument.external("llm_critic", model, role="critic") as call_record:
            response = http_client.post(url, headers=make_headers(), json=request_body)
            call_record["status"] = response.status_code
            response.raise_for_status()
            response_body = response.json()
            call_record.update(instrument.usage_fields(response_body))
            return response_body["choices"][0]["message"].get("content") or ""

    return chat


DRY_BAD_MARKERS = ("#W0000000", "sara_doe_496", "transfer_to_human_agents")


def dry_critic(messages: list[dict]) -> str:
    """The --dry-run critic: judges an action Bad if it contains a known bad marker."""
    current_action = messages[-1]["content"].split("Current action:")[-1].split("What happened after it:")[0]
    action_is_bad = any(marker in current_action for marker in DRY_BAD_MARKERS)
    verdict = "Bad" if action_is_bad else "Good"
    return f"Dry run critic. Judgment: {verdict}"


def revise_task(
    task_id: str,
    traces: list[dict],
    revise_options: argparse.Namespace,
    prompts_by_sha: dict,
    chat,
    critic_rows: list,
) -> tuple[list, list, dict]:
    """Revisions and good traces of one task. Appends every critic call to critic_rows."""
    trace_by_simulation_id = {trace["simulation_id"]: trace for trace in traces}
    leaves = [
        {"path": trace["mcts"]["path"], "reward": float(trace["reward"]), "simulation_id": trace["simulation_id"]}
        for trace in traces
    ]
    pairs = agent_r.select_pairs(
        leaves, revise_options.alpha, revise_options.beta, revise_options.max_pairs_per_task, revise_options.seed
    )
    good_leaves = agent_r.select_good(leaves, revise_options.alpha, revise_options.max_good_per_task)
    random_generator = random.Random(f"{revise_options.seed}-{task_id}")

    revisions = []
    outcome_counts = defaultdict(int)
    for pair in pairs:
        bad_trace = trace_by_simulation_id[pair["bad"]["simulation_id"]]
        good_trace = trace_by_simulation_id[pair["good"]["simulation_id"]]
        bad_steps = agent_r.split_steps(bad_trace["messages"])
        good_steps = agent_r.split_steps(good_trace["messages"])
        steps_match_paths = (
            len(bad_steps) == len(pair["bad"]["path"]) and len(good_steps) == len(pair["good"]["path"])
        )
        if not steps_match_paths:
            outcome_counts["skipped_step_mismatch"] += 1
            continue

        divergence_step = pair["t"]
        policy = policy_text(bad_trace, prompts_by_sha)

        def judge_step(history, step, judged_trace=bad_trace, policy=policy):
            raw_reply = chat(agent_r.critic_messages(policy, history, step))
            critic_rows.append({
                "task_id": task_id,
                "bad_simulation_id": judged_trace["simulation_id"],
                "step": len(history),
                "action": agent_r.step_summary(step),
                "verdict": agent_r.parse_judgment(raw_reply),
                "raw": raw_reply,
            })
            return raw_reply

        if revise_options.transition == "model":
            critic_judge = judge_step
        else:
            critic_judge = None
        try:
            transition = agent_r.find_transition(bad_steps, divergence_step, revise_options.transition, critic_judge)
        except httpx.HTTPError as critic_error:
            logger.warning(f"critic call failed, pair skipped: {critic_error}", task_id=task_id)
            outcome_counts["skipped_critic_error"] += 1
            continue

        if transition["found"]:
            outcome_counts["found"] += 1
        else:
            outcome_counts["fallback_last_step"] += 1

        revision_thought = random_generator.choice(agent_r.REVISION_THOUGHTS)
        revision = agent_r.build_revision(
            bad_steps, good_steps, divergence_step, transition["t_prime"], revision_thought,
            revise_options.revision_format,
        )
        bad_prompt_sha = next(
            (turn.get("prompt_sha") for turn in bad_trace.get("agent_turns", []) if turn.get("prompt_sha")),
            None,
        )
        revisions.append({
            "task_id": task_id,
            "source": "revision",
            **revision,
            "t": divergence_step,
            "t_prime": transition["t_prime"],
            "transition": revise_options.transition,
            "critic_found_bad": transition["found"],
            "bad_steps_judged": len(transition["judgments"]),
            "thought": revision_thought,
            "format": revise_options.revision_format,
            "bad_simulation_id": bad_trace["simulation_id"],
            "good_simulation_id": good_trace["simulation_id"],
            "bad_reward": bad_trace["reward"],
            "good_reward": good_trace["reward"],
            "prompt_sha": bad_prompt_sha,
        })

    good_traces = [trace_by_simulation_id[leaf["simulation_id"]] for leaf in good_leaves]
    task_row = {
        "leaves": len(leaves),
        "pairs": len(pairs),
        "revisions": len(revisions),
        "good": len(good_traces),
        **outcome_counts,
    }
    return revisions, good_traces, task_row


def make_critic(revise_options: argparse.Namespace, search_run_config: dict):
    """The chat function that judges steps, or None when --transition is not "model"."""
    if revise_options.transition != "model":
        return None
    if revise_options.dry_run:
        return dry_critic

    critic_model = revise_options.critic_model or search_run_config["agent_model"]
    if revise_options.critic_api_base:
        critic_api_base = revise_options.critic_api_base
    elif revise_options.critic_model:
        critic_api_base = None  # a different critic model does not share the search's endpoint
    else:
        critic_api_base = search_run_config.get("agent_api_base")
    return make_critic_chat(critic_model, critic_api_base)


def main(argv: list[str] | None = None) -> int:
    revise_options = parse_args(argv)
    if revise_options.alpha is None:
        revise_options.alpha = agent_r.ALPHA_BY_ITERATION[revise_options.iteration]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if revise_options.output_dir:
        output_directory = revise_options.output_dir.resolve()
    else:
        output_directory = (ROOT / "runs" / f"agent-r-revise-{timestamp}").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    log.attach_run_dir(output_directory)

    traces_by_task_id, run_config, prompts_by_sha = load_search(revise_options.search_dir)
    if run_config.get("task_split") != "train":
        raise SystemExit(f"{revise_options.search_dir}: task_split is {run_config.get('task_split')!r}, not 'train'.")
    chat = make_critic(revise_options, run_config)

    from shipwright.heal.traces import write_jsonl

    all_revisions: list[dict] = []
    all_good_traces: list[dict] = []
    critic_rows: list[dict] = []
    row_by_task_id: dict[str, dict] = {}
    for task_id in sorted(traces_by_task_id, key=lambda task_id: (len(task_id), task_id)):
        revisions, good_traces, task_row = revise_task(
            task_id, traces_by_task_id[task_id], revise_options, prompts_by_sha, chat, critic_rows
        )
        all_revisions += revisions
        all_good_traces += good_traces
        row_by_task_id[task_id] = task_row
        logger.info("task revised", task_id=task_id, **task_row)

    write_jsonl(output_directory / "revisions.jsonl", all_revisions)
    write_jsonl(output_directory / "good.jsonl", all_good_traces)
    write_jsonl(output_directory / "critic.jsonl", critic_rows)
    search_prompts_path = revise_options.search_dir / "prompts.json"
    if search_prompts_path.exists():
        shutil.copy(search_prompts_path, output_directory / "prompts.json")

    run_config["agent_r_revise"] = {
        option_name: (str(option_value) if isinstance(option_value, Path) else option_value)
        for option_name, option_value in vars(revise_options).items()
    }
    (output_directory / "run_config.json").write_text(json.dumps(run_config, indent=2))

    verdict_counts: dict[str, int] = defaultdict(int)
    for critic_row in critic_rows:
        verdict_counts[critic_row["verdict"]] += 1
    if all_revisions:
        bad_steps_kept = [revision["t_prime"] - revision["t"] + 1 for revision in all_revisions]
        mean_bad_steps_kept = round(sum(bad_steps_kept) / len(all_revisions), 2)
    else:
        mean_bad_steps_kept = None
    summary = {
        "search_dir": str(revise_options.search_dir),
        "alpha": revise_options.alpha,
        "beta": revise_options.beta,
        "transition": revise_options.transition,
        "revision_format": revise_options.revision_format,
        "tasks": len(traces_by_task_id),
        "tasks_with_revisions": sum(1 for task_row in row_by_task_id.values() if task_row["revisions"]),
        "revisions": len(all_revisions),
        "good": len(all_good_traces),
        "critic_calls": len(critic_rows),
        "critic_verdicts": dict(verdict_counts),
        "critic_found_bad": sum(
            1 for revision in all_revisions if revision["critic_found_bad"] and revision["transition"] == "model"
        ),
        "mean_bad_steps_kept": mean_bad_steps_kept,
        "per_task": row_by_task_id,
    }
    (output_directory / "revise_summary.json").write_text(json.dumps(summary, indent=2))
    printable_summary = {key: value for key, value in summary.items() if key != "per_task"}
    print(json.dumps(printable_summary, indent=2))
    print(f"\nWrote {output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="agent_r_revise"))
