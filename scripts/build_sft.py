"""Build the SFT dataset for the weight-fix path from successful runs.

  --teacher RUN_DIR   teacher runs (scripts/teacher_runs.py); repeatable
  --replay RUN_DIR    Tiny Aya's own runs; their successes are mixed in (replay); repeatable
  --agent-r DIR       Agent-R output (scripts/agent_r_revise.py): revision runs, trained only on
                      the revision thought and the good suffix (records carry ``loss_mask``),
                      plus the good MCTS leaves; repeatable
  --general FILE      general chat data (jsonl, messages or ShareGPT), mixed so that agent data
                      is --eta of the total (Agent-R Eq. 6, eta = 0.2)

Only runs with reward == 1 are used, and only from train-split runs (run_config.json).
Every example is rendered as Tiny Aya sees it (see shipwright.heal.sft_data), deduplicated,
and dropped if it has more than --max-tokens tokens or a reply over 700 tokens.

Writes OUT_DIR/sft.jsonl (one {"messages": [...], "source", "task_id", ...} per line)
and OUT_DIR/sft_stats.json. --push uploads both to an HF dataset repo (needs the token).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from shipwright.llm import context  # noqa: E402
from shipwright.heal import sft_data  # noqa: E402

DEFAULT_REPO = "mitushigoyal5/shipwright-sft"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", type=Path, action="append", default=[], help="teacher run dir")
    p.add_argument("--replay", type=Path, action="append", default=[], help="Tiny Aya run dir")
    p.add_argument("--max-replay", type=int, default=None, help="cap on replay examples")
    p.add_argument("--agent-r", type=Path, action="append", default=[], help="agent_r_revise.py output dir")
    p.add_argument("--general", type=Path, default=None, help="general chat jsonl for the eta mix")
    p.add_argument("--eta", type=float, default=0.2, help="agent share of the data when --general is set")
    p.add_argument("--long-mode", choices=["window", "drop"], default="window",
                   help="runs longer than the context: cut into windows, or drop")
    p.add_argument("--max-tokens", type=int, default=context.CONTEXT_LIMIT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--estimate-tokens", action="store_true",
                   help="count tokens with the chars estimate (no tokenizer download)")
    p.add_argument("--tool-patches", type=Path, default=None, metavar="LEDGER",
                   help="render every example with the ledger's applied tool patches (the gate's tool list)")
    p.add_argument("--allow-split", action="store_true", help="accept runs not on the train split")
    p.add_argument("--push", action="store_true", help="upload to the HF dataset repo")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--revision", default="main", help="dataset branch, e.g. round-1")
    return p.parse_args(argv)


def load_run(run_dir: Path, allow_split: bool) -> tuple[list[dict], dict]:
    """Successful traces and prompts of one run dir."""
    cfg_path = run_dir / "run_config.json"
    split = json.loads(cfg_path.read_text()).get("task_split") if cfg_path.exists() else None
    if split != "train" and not allow_split:
        raise SystemExit(f"{run_dir}: task_split is {split!r}, not 'train'. Never train on held-out tasks.")
    traces = []
    with open(run_dir / "traces.jsonl") as f:
        for line in f:
            t = json.loads(line)
            if t.get("passed"):
                traces.append(t)
    prompts_path = run_dir / "prompts.json"
    prompts = json.loads(prompts_path.read_text()) if prompts_path.exists() else {}
    return traces, prompts


def reconstruct_prompt(domain: str = "retail", seed: int = 300) -> dict:
    """The agent prompt for runs made before prompts.json existed.

    Rebuilt the way tau2 builds it (LLMAgent system prompt, domain policy plus the
    language guidance of the default English run). tests/heal/test_sft_data.py checks it
    against a recorded prompt.
    """
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
    from tau2.data_model.simulation import TextRunConfig
    from tau2.multilingual.runtime import agent_language_guidance, task_context
    from tau2.registry import registry
    from tau2.runner.build import agent_domain_policy

    cfg = TextRunConfig(domain=domain, agent="llm_agent", llm_agent="agent", user="user_simulator",
                        llm_user="user", task_split_name="train", seed=seed)
    task = registry.get_tasks_loader(domain)(task_split_name="train")[0]
    guidance = agent_language_guidance(task_context(cfg, task))
    env = registry.get_env_constructor(domain)()
    policy = agent_domain_policy(env, guidance)
    system = SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION)
    return {"system": system, "tools": [t.openai_schema for t in env.get_tools()]}


def patch_prompt(prompt: dict | None, rows: list[dict] | None) -> dict | None:
    """``prompt`` with every applied patch in ``rows``; fields already patched stay as they are.

    Teacher runs used the original tool descriptions, the gate uses the patched ones: render
    the training data with the tool list the gated system sees.
    """
    if not prompt or not rows:
        return prompt
    from shipwright.heal import tool_fix

    tools = tool_fix.tool_map(prompt["tools"])
    todo = []
    for r in rows:
        if r.get("status") != "applied":
            continue
        current = tool_fix.sha256(tool_fix.get_field(tools[r["tool"]], r["field"]))
        if current == r["sha256_before"]:
            todo.append(r)
        elif current != r["sha256_after"]:
            raise ValueError(f"patch {r['id']}: {r['tool']}.{r['field']} matches neither its before nor after text")
    patched, _ = tool_fix.apply_patches(prompt["tools"], todo)
    return {**prompt, "tools": patched}


def prompt_for(trace: dict, prompts: dict, fallback: dict | None) -> dict | None:
    shas = {t.get("prompt_sha") for t in trace["agent_turns"] if t.get("prompt_sha")}
    if len(shas) == 1 and next(iter(shas)) in prompts:
        return prompts[next(iter(shas))]
    if len(prompts) == 1 and not shas:
        return next(iter(prompts.values()))
    return fallback


def collect(run_dirs: list[Path], source: str, args: argparse.Namespace, notes: list[str]) -> list[dict]:
    records: list[dict] = []
    fallback = None
    for run_dir in run_dirs:
        traces, prompts = load_run(run_dir, args.allow_split)
        cfg_path = run_dir / "run_config.json"
        model = json.loads(cfg_path.read_text()).get("agent_model") if cfg_path.exists() else None
        for t in traces:
            prompt = prompt_for(t, prompts, None)
            if prompt is None:
                if fallback is None:
                    fallback = reconstruct_prompt()
                    notes.append("some runs had no prompts.json: agent prompt rebuilt from tau2")
                prompt = fallback
            recs = sft_data.trace_records(t, patch_prompt(prompt, args.patch_rows), source, args.long_mode)
            for r in recs:  # which model wrote the targets (teacher id, or Tiny Aya for replay)
                r["model"] = model
            records += recs
        notes.append(f"{run_dir.name}: {len(traces)} successful {source} runs")
    return records


def collect_agent_r(dirs: list[Path], args: argparse.Namespace, notes: list[str]) -> list[dict]:
    """Revision records (masked) and good-leaf records from agent_r_revise.py output dirs."""
    records: list[dict] = []
    for d in dirs:
        cfg = json.loads((d / "run_config.json").read_text())
        if cfg.get("task_split") != "train" and not args.allow_split:
            raise SystemExit(f"{d}: task_split is {cfg.get('task_split')!r}, not 'train'. Never train on held-out tasks.")
        prompts = json.loads((d / "prompts.json").read_text()) if (d / "prompts.json").exists() else {}
        fallback = next(iter(prompts.values())) if prompts else reconstruct_prompt()
        n_rev = n_good = n_invalid = 0
        valid = None
        if (d / "check.jsonl").exists():  # scripts/agent_r_check.py: keep only revisions that passed it
            valid = {(r["bad_simulation_id"], r.get("good_simulation_id")) for r in
                     map(json.loads, (d / "check.jsonl").read_text().splitlines()) if r.get("valid")}
        for line in (d / "revisions.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rev = json.loads(line)
            if valid is not None and (rev.get("bad_simulation_id"), rev.get("good_simulation_id")) not in valid:
                n_invalid += 1
                continue
            prompt = patch_prompt(prompts.get(rev.get("prompt_sha")) or fallback, args.patch_rows)
            records += sft_data.revision_records(rev, prompt, args.long_mode)
            n_rev += 1
        for line in (d / "good.jsonl").read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                records += sft_data.trace_records(t, patch_prompt(prompt_for(t, prompts, fallback), args.patch_rows),
                                                  "good", args.long_mode)
                n_good += 1
        checked = "unchecked" if valid is None else f"{n_invalid} dropped by agent_r_check"
        notes.append(f"{d.name}: {n_rev} Agent-R revision runs ({checked}), {n_good} good MCTS leaves")
    return records


def push(out_dir: Path, repo: str, revision: str) -> None:
    from huggingface_hub import HfApi

    import baseline

    token = baseline.load_token()
    if not token:
        raise SystemExit("No HF token for --push.")
    from shipwright.observability import instrument, retry

    api = HfApi(token=token)
    retry.call_with_retry(api.create_repo, repo, repo_type="dataset", exist_ok=True, private=False,
                          what="hf_hub create_repo")
    if revision != "main":
        retry.call_with_retry(api.create_branch, repo, repo_type="dataset", branch=revision, exist_ok=True,
                              what="hf_hub create_branch")

    for name in ("sft.jsonl", "sft_stats.json"):
        with instrument.external("hf_hub_upload", f"datasets/{repo}@{revision}/{name}"):
            retry.call_with_retry(api.upload_file, path_or_fileobj=str(out_dir / name), path_in_repo=name,
                                  repo_id=repo, repo_type="dataset", revision=revision,
                                  commit_message=f"SFT data ({out_dir.name})", what="hf_hub_upload")
    print(f"Pushed to https://huggingface.co/datasets/{repo}/tree/{revision}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.teacher and not args.replay and not args.agent_r:
        raise SystemExit("Pass at least one --teacher, --replay or --agent-r dir.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (args.output_dir or ROOT / "runs" / f"sft-{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    from shipwright.observability import log

    log.attach_run_dir(out_dir)

    import baseline

    method = "estimate"
    if not args.estimate_tokens and context.load_tokenizer(context.STUDENT_REPO, baseline.load_token()):
        method = "tokenizer"
    notes: list[str] = [f"token counting: {method}"]
    args.patch_rows = None
    if args.tool_patches:
        args.patch_rows = [json.loads(x) for x in args.tool_patches.read_text().splitlines() if x.strip()]
        applied = sum(r.get("status") == "applied" for r in args.patch_rows)
        notes.append(f"tool list: {applied} applied patches from {args.tool_patches}")

    teacher = collect(args.teacher, "teacher", args, notes)
    replay = collect(args.replay, "replay", args, notes)
    if args.max_replay is not None:
        replay = replay[: args.max_replay]
    agent_r = collect_agent_r(args.agent_r, args, notes)
    before = len(teacher) + len(replay) + len(agent_r)
    records = sft_data.dedupe(agent_r + teacher + replay)
    kept, dropped = sft_data.filter_records(records, args.max_tokens)
    dropped["duplicates"] = before - len(records)
    mix = None
    if args.general:
        rows = [json.loads(x) for x in args.general.read_text().splitlines() if x.strip()]
        general, _ = sft_data.filter_records(sft_data.general_records(rows), args.max_tokens)
        kept, mix = sft_data.eta_mix(kept, general, args.eta)
        mix["general_file"] = str(args.general)

    with open(out_dir / "sft.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = sft_data.stats(kept, dropped, {
        "token_count_method": method,
        "long_mode": args.long_mode,
        "max_tokens_limit": args.max_tokens,
        "teacher_runs": [str(p) for p in args.teacher],
        "replay_runs": [str(p) for p in args.replay],
        "agent_r_dirs": [str(p) for p in args.agent_r],
        "eta_mix": mix,
        "notes": notes,
    })
    (out_dir / "sft_stats.json").write_text(json.dumps(st, indent=2))
    print(json.dumps(st, indent=2))
    print(f"\nWrote {out_dir}")
    if args.push:
        push(out_dir, args.repo, args.revision)
    return 0


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="sft"))
