"""Baseline: Tiny Aya Global as a tau2-bench tool-calling agent via HF Inference Providers.

Writes to the output dir:
  tau2/results.json  tau2's own results file
  traces.jsonl       one line per task run, with raw model output for every agent turn
  summary.json       pass rate, tool-call quality counts, steps, tokens, cost estimate
  run_config.json    the arguments of this run (no secrets)
  prompts.json       the agent's un-converted system prompt and tool schemas, by prompt_sha
                     (prompted models only; scripts/build_sft.py renders training data from it)

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", default="retail")
    p.add_argument("--num-tasks", type=int, default=20, help="0 means all tasks of the split")
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
    p.add_argument("--tool-patches", type=Path, default=None,
                   help="ledger (patches/ledger.jsonl); its 'applied' patches change the Tiny Aya tool prompt")
    p.add_argument("--dry-run", action="store_true", help="fake LLM, no network")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def estimate_cost(summary: dict, prices: dict | None = None) -> dict:
    prices = prices or PRICES
    out = {}
    for who in ("agent", "user"):
        toks = summary.get(f"{who}_tokens") or {}
        pin, pout = prices[who]
        out[who] = round(
            toks.get("prompt_tokens", 0) / 1e6 * pin + toks.get("completion_tokens", 0) / 1e6 * pout, 4
        )
    out["total"] = round(out["agent"] + out["user"], 4)
    out["note"] = "estimate from token counts and assumed prices in scripts/baseline.py"
    return out


def run(
    args: argparse.Namespace,
    *,
    prefix: str = "baseline",
    prompted_models: dict[str, int] | None = None,
    agent_llm_args: dict | None = None,
    prices: dict | None = None,
) -> tuple[Path, list[dict], dict] | None:
    """Run tau2 with the given args. Returns (run dir, traces, summary), or None without a token.

    ``prompted_models`` maps extra model names to their reply-token budget: they get the
    same prompted-tools view as Tiny Aya (used for the teacher). ``agent_llm_args`` is
    merged into the agent's litellm args (e.g. reasoning_effort).
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (args.output_dir or ROOT / "runs" / f"{prefix}-{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    token = None
    if args.dry_run:
        os.environ.setdefault("OPENAI_API_KEY", "dry-run")
    else:
        token = load_token()
        if not token:
            print("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.", file=sys.stderr)
            return None
        # litellm's openai provider reads these; nothing token-bearing goes into tau2 config.
        os.environ["OPENAI_API_KEY"] = token
        os.environ["OPENAI_API_BASE"] = HF_ROUTER
        os.environ["OPENAI_BASE_URL"] = HF_ROUTER

    from shipwright import llm_patch, prompted_tools
    from shipwright.traces import build_trace, summarize, write_jsonl

    for model, reply_tokens in (prompted_models or {}).items():
        prompted_tools.force_prompted(model, reply_tokens)

    # tau2 has built-in default models (NL-assertion judge gpt-5.4-mini, env interface,
    # reviewers). Send every model other than agent and user to the user model.
    rerouter = llm_patch.ModelRerouter({args.agent_model, args.user_model}, args.user_model)
    if args.dry_run:
        from shipwright.fake_llm import make_fake_completion

        llm_patch.install(make_fake_completion(args.agent_model), model_map=rerouter)
    else:
        llm_patch.install(model_map=rerouter)
        from shipwright import context

        if prompted_tools.is_prompted_model(args.agent_model):
            # Count tokens with the student's tokenizer: the teacher gets the student's budget.
            repo = args.agent_model.removeprefix("openai/").split(":")[0]
            repo = repo if "tiny-aya" in repo.lower() else context.STUDENT_REPO
            print(f"token counting: {'tokenizer' if context.load_tokenizer(repo, token) else 'estimate'}")

    import tau2.evaluator.evaluator_nl_assertions as nl_assertions

    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = args.user_model

    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.runner import run_domain

    domain_tools = registry.get_env_constructor(args.domain)().get_tools()
    tool_names = {t.name for t in domain_tools}

    active_patches: list[str] = []
    if args.tool_patches:
        from shipwright.prompted_tools import set_tool_patches
        from shipwright.tool_fix import apply_patches, load_ledger

        ledger = load_ledger(args.tool_patches)
        # Fail now, not mid-run, if an applied patch does not fit the current tools.
        _, active_patches = apply_patches([t.openai_schema for t in domain_tools], ledger)
        set_tool_patches(ledger)
        print(f"tool patches active: {active_patches or 'none'}")

    run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_config["output_dir"] = str(out_dir)
    run_config["active_tool_patches"] = active_patches
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    config = TextRunConfig(
        domain=args.domain,
        agent="llm_agent",
        llm_agent=args.agent_model,
        llm_args_agent={"temperature": args.agent_temperature, **(agent_llm_args or {})},
        user="user_simulator",
        llm_user=args.user_model,
        llm_args_user={"temperature": 0.0},
        task_split_name=args.task_split,
        task_ids=args.task_ids or None,
        num_tasks=None if args.task_ids or not args.num_tasks else args.num_tasks,
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
    summary["cost_estimate_usd"] = estimate_cost(summary, prices)
    summary["agent_model"] = args.agent_model
    summary["user_model"] = args.user_model
    summary["domain"] = args.domain
    summary["task_split"] = args.task_split
    summary["rerouted_models"] = rerouter.counts
    summary["active_tool_patches"] = active_patches
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    prompts = prompted_tools.recorded_prompts()
    if prompts:
        (out_dir / "prompts.json").write_text(json.dumps(prompts, indent=1))
    return out_dir, traces, summary


def main() -> int:
    result = run(parse_args())
    if result is None:
        return 2
    out_dir, _, summary = result
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
