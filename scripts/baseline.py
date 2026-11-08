"""Baseline: Tiny Aya Global as a tau2-bench tool-calling agent via HF Inference Providers.

Writes to the output dir:
  tau2/results.json  tau2's own results file
  traces.jsonl       one line per task run, with raw model output for every agent turn
  summary.json       pass rate, tool-call quality counts, steps, tokens, cost estimate
  run_config.json    the arguments of this run (no secrets)

The HF token is read at run time from $SWITCHBOARD_SECRETS_DIR/HF_TOKEN (or $HF_TOKEN).
It is passed to litellm only through process environment variables, never written to disk.

--dry-run uses a scripted fake LLM: no network, no token, exercises the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HF_ROUTER = "https://router.huggingface.co/v1"
DEFAULT_AGENT = "openai/CohereLabs/tiny-aya-global:cohere"
DEFAULT_USER = "openai/openai/gpt-oss-120b:cerebras"

# USD per 1M tokens (input, output). Estimates only; check HF billing for real cost.
PRICES = {
    "agent": (0.50, 1.50),  # Tiny Aya on cohere: price not published, conservative guess
    "user": (0.25, 0.69),  # gpt-oss-120b on cerebras
}


def load_token() -> str | None:
    secrets_dir = os.environ.get("SWITCHBOARD_SECRETS_DIR")
    if secrets_dir:
        path = Path(secrets_dir) / "HF_TOKEN"
        try:
            token = path.read_text().strip()
            if token:
                return token
        except OSError:
            pass
    return os.environ.get("HF_TOKEN") or None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", default="retail")
    p.add_argument("--num-tasks", type=int, default=20)
    p.add_argument("--task-split", default="train", help="tau2 split: train, test or base")
    p.add_argument("--task-ids", nargs="*", default=None, help="explicit task ids (overrides --num-tasks)")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--agent-model", default=DEFAULT_AGENT)
    p.add_argument("--user-model", default=DEFAULT_USER)
    p.add_argument("--agent-temperature", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="fake LLM, no network")
    return p.parse_args()


def estimate_cost(summary: dict) -> dict:
    out = {}
    for who in ("agent", "user"):
        toks = summary.get(f"{who}_tokens") or {}
        pin, pout = PRICES[who]
        out[who] = round(
            toks.get("prompt_tokens", 0) / 1e6 * pin + toks.get("completion_tokens", 0) / 1e6 * pout, 4
        )
    out["total"] = round(out["agent"] + out["user"], 4)
    out["note"] = "estimate from token counts and assumed prices in scripts/baseline.py"
    return out


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (args.output_dir or ROOT / "runs" / f"baseline-{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        os.environ.setdefault("OPENAI_API_KEY", "dry-run")
    else:
        token = load_token()
        if not token:
            print("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.", file=sys.stderr)
            return 2
        # litellm's openai provider reads these; nothing token-bearing goes into tau2 config.
        os.environ["OPENAI_API_KEY"] = token
        os.environ["OPENAI_API_BASE"] = HF_ROUTER
        os.environ["OPENAI_BASE_URL"] = HF_ROUTER

    from shipwright import llm_patch
    from shipwright.traces import build_trace, summarize, write_jsonl

    if args.dry_run:
        from shipwright.fake_llm import make_fake_completion

        llm_patch.install(make_fake_completion(args.agent_model))
    else:
        llm_patch.install()

    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.runner import run_domain

    tool_names = {t.name for t in registry.get_env_constructor(args.domain)().get_tools()}

    run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_config["output_dir"] = str(out_dir)
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    config = TextRunConfig(
        domain=args.domain,
        agent="llm_agent",
        llm_agent=args.agent_model,
        llm_args_agent={"temperature": args.agent_temperature},
        user="user_simulator",
        llm_user=args.user_model,
        llm_args_user={"temperature": 0.0},
        task_split_name=args.task_split,
        task_ids=args.task_ids or None,
        num_tasks=None if args.task_ids else args.num_tasks,
        num_trials=args.trials,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        save_to=str(out_dir / "tau2"),  # absolute path: tau2 saves here, not under its data dir
        max_retries=0,
    )
    results = run_domain(config)

    sims = [s.model_dump(mode="json") for s in results.simulations]
    traces = [build_trace(s, tool_names) for s in sims]
    write_jsonl(out_dir / "traces.jsonl", traces)
    summary = summarize(traces)
    summary["cost_estimate_usd"] = estimate_cost(summary)
    summary["agent_model"] = args.agent_model
    summary["user_model"] = args.user_model
    summary["domain"] = args.domain
    summary["task_split"] = args.task_split
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
