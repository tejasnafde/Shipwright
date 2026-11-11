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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("search_dir", type=Path, help="run dir of scripts/agent_r_search.py")
    p.add_argument("--iteration", type=int, default=1, choices=sorted(agent_r.ALPHA_BY_ITERATION),
                   help="sets --alpha to the paper's schedule (0.5, 0.7, 1.0)")
    p.add_argument("--alpha", type=float, default=None, help="good leaf: reward >= alpha (tau2 rewards are 0 or 1)")
    p.add_argument("--beta", type=float, default=agent_r.BETA, help="good reward - bad reward >= beta")
    p.add_argument("--transition", choices=["model", "direct", "divergence"], default="model")
    p.add_argument("--revision-format", choices=["inline", "turn"], default="inline")
    p.add_argument("--critic-model", default=None, help="default: the search's actor model")
    p.add_argument("--critic-api-base", default=None, help="default: the search's --agent-api-base")
    p.add_argument("--max-pairs-per-task", type=int, default=10)
    p.add_argument("--max-good-per-task", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="rule-based fake critic, no network")
    return p.parse_args(argv)


def load_search(search_dir: Path) -> tuple[dict[str, list[dict]], dict, dict]:
    """Traces by task (only rollouts with an mcts path), run_config, prompts."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    with open(search_dir / "traces.jsonl") as f:
        for line in f:
            t = json.loads(line)
            if t.get("mcts") and t.get("reward") is not None:
                by_task[str(t["task_id"])].append(t)
    cfg = json.loads((search_dir / "run_config.json").read_text())
    prompts_path = search_dir / "prompts.json"
    prompts = json.loads(prompts_path.read_text()) if prompts_path.exists() else {}
    return by_task, cfg, prompts


def policy_text(trace: dict, prompts: dict) -> str:
    """The actor's own system prompt (the policy), for the critic."""
    shas = [t.get("prompt_sha") for t in trace.get("agent_turns", []) if t.get("prompt_sha")]
    if shas and shas[-1] in prompts:
        return prompts[shas[-1]]["system"]
    if len(prompts) == 1:
        return next(iter(prompts.values()))["system"]
    return ""


def make_critic_chat(model: str, api_base: str | None):
    """chat(messages) -> text for the actor as critic: vLLM endpoint, Vertex, or the HF router."""
    from baseline import HF_ROUTER, load_token
    from shipwright.llm import vertex
    from shipwright.observability import instrument

    body_model = model.removeprefix("openai/")
    if api_base:
        url, headers = f"{api_base.rstrip('/')}/chat/completions", (lambda: {})
    else:
        url, body_model, headers = vertex.endpoint(model if vertex.is_vertex_model(model) else body_model,
                                                   load_token(), HF_ROUTER)
    client = httpx.Client(timeout=120)

    def chat(messages: list[dict]) -> str:
        # 400 reply tokens: the critic reasons briefly, then the Judgment line (Tiny Aya's
        # context is 8192 and critic_messages keeps the prompt near 5k tokens).
        with instrument.external("llm_critic", model, role="critic") as call:
            r = client.post(url, headers=headers(),
                            json={"model": body_model, "messages": messages, "temperature": 0, "max_tokens": 400})
            call["status"] = r.status_code
            r.raise_for_status()
            body = r.json()
            call.update(instrument.usage_fields(body))
            return body["choices"][0]["message"].get("content") or ""

    return chat


DRY_BAD_MARKERS = ("#W0000000", "sara_doe_496", "transfer_to_human_agents")


def dry_critic(messages: list[dict]) -> str:
    action = messages[-1]["content"].split("Current action:")[-1].split("What happened after it:")[0]
    bad = any(m in action for m in DRY_BAD_MARKERS)
    return f"Dry run critic. Judgment: {'Bad' if bad else 'Good'}"


def revise_task(task_id: str, traces: list[dict], args, prompts: dict, chat, critic_rows: list) -> tuple[list, list, dict]:
    by_sim = {t["simulation_id"]: t for t in traces}
    leaves = [{"path": t["mcts"]["path"], "reward": float(t["reward"]), "simulation_id": t["simulation_id"]}
              for t in traces]
    pairs = agent_r.select_pairs(leaves, args.alpha, args.beta, args.max_pairs_per_task, args.seed)
    goods = agent_r.select_good(leaves, args.alpha, args.max_good_per_task)
    rng = random.Random(f"{args.seed}-{task_id}")
    revisions, stats = [], defaultdict(int)
    for pair in pairs:
        bad, good = by_sim[pair["bad"]["simulation_id"]], by_sim[pair["good"]["simulation_id"]]
        bad_steps, good_steps = agent_r.split_steps(bad["messages"]), agent_r.split_steps(good["messages"])
        if len(bad_steps) != len(pair["bad"]["path"]) or len(good_steps) != len(pair["good"]["path"]):
            stats["skipped_step_mismatch"] += 1
            continue
        t = pair["t"]
        policy = policy_text(bad, prompts)

        def judge(history, step, _bad=bad):
            raw = chat(agent_r.critic_messages(policy, history, step))
            critic_rows.append({"task_id": task_id, "bad_simulation_id": _bad["simulation_id"],
                                "step": len(history), "action": agent_r.step_summary(step),
                                "verdict": agent_r.parse_judgment(raw), "raw": raw})
            return raw

        try:
            tr = agent_r.find_transition(bad_steps, t, args.transition, judge if args.transition == "model" else None)
        except httpx.HTTPError as e:
            logger.warning(f"critic call failed, pair skipped: {e}", task_id=task_id)
            stats["skipped_critic_error"] += 1
            continue
        stats["found" if tr["found"] else "fallback_last_step"] += 1
        thought = rng.choice(agent_r.REVISION_THOUGHTS)
        rev = agent_r.build_revision(bad_steps, good_steps, t, tr["t_prime"], thought, args.revision_format)
        sha = next((x.get("prompt_sha") for x in bad.get("agent_turns", []) if x.get("prompt_sha")), None)
        revisions.append({
            "task_id": task_id, "source": "revision", **rev,
            "t": t, "t_prime": tr["t_prime"], "transition": args.transition, "critic_found_bad": tr["found"],
            "bad_steps_judged": len(tr["judgments"]), "thought": thought, "format": args.revision_format,
            "bad_simulation_id": bad["simulation_id"], "good_simulation_id": good["simulation_id"],
            "bad_reward": bad["reward"], "good_reward": good["reward"], "prompt_sha": sha,
        })
    good_traces = [by_sim[g["simulation_id"]] for g in goods]
    row = {"leaves": len(leaves), "pairs": len(pairs), "revisions": len(revisions), "good": len(good_traces),
           **stats}
    return revisions, good_traces, row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.alpha is None:
        args.alpha = agent_r.ALPHA_BY_ITERATION[args.iteration]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (args.output_dir or ROOT / "runs" / f"agent-r-revise-{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.attach_run_dir(out_dir)

    by_task, cfg, prompts = load_search(args.search_dir)
    if cfg.get("task_split") != "train":
        raise SystemExit(f"{args.search_dir}: task_split is {cfg.get('task_split')!r}, not 'train'.")
    chat = None
    if args.transition == "model":
        if args.dry_run:
            chat = dry_critic
        else:
            model = args.critic_model or cfg["agent_model"]
            chat = make_critic_chat(model, args.critic_api_base or (None if args.critic_model else cfg.get("agent_api_base")))

    from shipwright.heal.traces import write_jsonl

    revisions, goods, critic_rows, per_task = [], [], [], {}
    for task_id in sorted(by_task, key=lambda x: (len(x), x)):
        r, g, row = revise_task(task_id, by_task[task_id], args, prompts, chat, critic_rows)
        revisions += r
        goods += g
        per_task[task_id] = row
        logger.info("task revised", task_id=task_id, **row)

    write_jsonl(out_dir / "revisions.jsonl", revisions)
    write_jsonl(out_dir / "good.jsonl", goods)
    write_jsonl(out_dir / "critic.jsonl", critic_rows)
    for name in ("prompts.json",):
        if (args.search_dir / name).exists():
            shutil.copy(args.search_dir / name, out_dir / name)
    cfg["agent_r_revise"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    verdicts: dict[str, int] = defaultdict(int)
    for c in critic_rows:
        verdicts[c["verdict"]] += 1
    summary = {
        "search_dir": str(args.search_dir),
        "alpha": args.alpha, "beta": args.beta, "transition": args.transition,
        "revision_format": args.revision_format,
        "tasks": len(by_task), "tasks_with_revisions": sum(1 for r in per_task.values() if r["revisions"]),
        "revisions": len(revisions), "good": len(goods),
        "critic_calls": len(critic_rows), "critic_verdicts": dict(verdicts),
        "critic_found_bad": sum(1 for r in revisions if r["critic_found_bad"] and r["transition"] == "model"),
        "mean_bad_steps_kept": round(sum(r["t_prime"] - r["t"] + 1 for r in revisions) / len(revisions), 2)
        if revisions else None,
        "per_task": per_task,
    }
    (out_dir / "revise_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="agent_r_revise"))
