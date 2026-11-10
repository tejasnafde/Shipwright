"""Baseline: Tiny Aya Global as a tau2-bench tool-calling agent via HF Inference Providers.

Writes to the output dir:
  tau2/results.json  tau2's own results file
  traces.jsonl       one line per task run, with raw model output for every agent turn
  summary.json       pass rate, tool-call quality counts, steps, tokens, cost estimate
  run_config.json    the arguments of this run (no secrets)
  prompts.json       the agent's un-converted system prompt and tool schemas, by prompt_sha
                     (prompted models only; scripts/build_sft.py renders training data from it)
  otel_spans.jsonl   OpenTelemetry spans built from traces.jsonl (docs/ops/observability.md)

The HF token is read at run time from $SWITCHBOARD_SECRETS_DIR/HF_TOKEN (or $HF_TOKEN).
It is passed to litellm only through process environment variables, never written to disk.
Model ids that start with vertex/ (e.g. vertex/openai/gpt-oss-120b-maas) go to Vertex AI with a
gcloud token instead (src/shipwright/llm/vertex.py); a run with only vertex/ models needs no HF token.

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

from shipwright.observability import log, repro  # noqa: E402

logger = log.get_logger("baseline")

HF_ROUTER = "https://router.huggingface.co/v1"
DEFAULT_AGENT = "openai/CohereLabs/tiny-aya-global:cohere"
DEFAULT_USER = "openai/openai/gpt-oss-120b:cerebras"



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
    p.add_argument("--user-model", default=DEFAULT_USER,
                   help="user simulator (and tau2's judge): an HF router litellm id or vertex/<publisher>/<model>")
    p.add_argument("--agent-temperature", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--tool-patches", type=Path, default=None,
                   help="ledger (patches/ledger.jsonl); its 'applied' patches change the Tiny Aya tool prompt")
    p.add_argument("--tool-patch-ids", nargs="*", default=None,
                   help="with --tool-patches: use exactly these patch ids, whatever their ledger status (gate)")
    p.add_argument("--agent-api-base", default=None,
                   help="OpenAI-compatible endpoint for the agent only, e.g. http://localhost:8000/v1 (vLLM)."
                        " The user simulator still goes to the HF router.")
    p.add_argument("--dry-run", action="store_true", help="fake LLM, no network")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def cost_estimate(meter, args: argparse.Namespace, role_names: dict[str, str] | None = None) -> dict:
    """Per-role cost from the usage meter, priced by provider (src/shipwright/llm/costs.py)."""
    from shipwright.llm.costs import estimate, provider_of

    providers = {args.user_model: provider_of(args.user_model, dry_run=args.dry_run),
                 args.agent_model: provider_of(args.agent_model, args.agent_api_base, args.dry_run)}
    return estimate(meter.rows, providers, role_names)


def run(
    args: argparse.Namespace,
    *,
    prefix: str = "baseline",
    prompted_models: dict[str, int] | None = None,
    agent_llm_args: dict | None = None,
    role_names: dict[str, str] | None = None,
) -> tuple[Path, list[dict], dict] | None:
    """Run tau2 with the given args. Returns (run dir, traces, summary), or None without a token.

    ``prompted_models`` maps extra model names to their reply-token budget: they get the
    same prompted-tools view as Tiny Aya (used for the teacher). ``agent_llm_args`` is
    merged into the agent's litellm args (e.g. reasoning_effort). ``role_names`` renames
    roles in the cost estimate (teacher runs: agent -> teacher).
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (args.output_dir or ROOT / "runs" / f"{prefix}-{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.attach_run_dir(out_dir)
    log.note_repro(**repro.models_from_args(args), dry_run=args.dry_run)
    logger.info("run start", run_dir=str(out_dir), agent_model=args.agent_model, user_model=args.user_model,
                task_split=args.task_split, dry_run=args.dry_run)

    from shipwright.llm import vertex

    token = None
    needs_hf = not (vertex.is_vertex_model(args.user_model)
                    and (vertex.is_vertex_model(args.agent_model) or args.agent_api_base))
    if args.dry_run:
        os.environ.setdefault("OPENAI_API_KEY", "dry-run")
    else:
        token = load_token()
        if needs_hf and not token:
            logger.error("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.")
            return None
        if token:
            # litellm's openai provider reads these; nothing token-bearing goes into tau2 config.
            # vertex/ calls pass their own api_base and api_key per call.
            os.environ["OPENAI_API_KEY"] = token
            os.environ["OPENAI_API_BASE"] = HF_ROUTER
            os.environ["OPENAI_BASE_URL"] = HF_ROUTER
        if vertex.is_vertex_model(args.agent_model) or vertex.is_vertex_model(args.user_model):
            try:
                vertex.provider().get()  # fail now, not mid-run, if gcloud cannot mint a token
            except RuntimeError:
                logger.error("Vertex token failed", exc_info=True)
                return None
            logger.info("vertex endpoint", project=vertex.project(), location=vertex.location())

    from shipwright.llm import patches, prompted_tools
    from shipwright.llm.costs import UsageMeter
    from shipwright.heal.traces import build_trace, summarize, write_jsonl

    for model, reply_tokens in (prompted_models or {}).items():
        prompted_tools.force_prompted(model, reply_tokens)

    # tau2 has built-in default models (NL-assertion judge gpt-5.4-mini, env interface,
    # reviewers). Send every model other than agent and user to the user model.
    rerouter = patches.ModelRerouter({args.agent_model, args.user_model}, args.user_model)
    meter = UsageMeter()
    roles = {"meter": meter, "agent_model": args.agent_model, "user_model": args.user_model}
    if args.dry_run:
        from shipwright.llm.dry_run import make_fake_completion

        patches.install(make_fake_completion(args.agent_model), model_map=rerouter, **roles)
    else:
        patches.install(model_map=rerouter, **roles)
        from shipwright.llm import context

        if prompted_tools.is_prompted_model(args.agent_model):
            # Count tokens with the student's tokenizer: the teacher gets the student's budget.
            # A vLLM served name (e.g. a LoRA module) is not a repo id: use the student repo.
            repo = args.agent_model.removeprefix("openai/").split(":")[0]
            ok = "tiny-aya" in repo.lower() and "/" in repo and not args.agent_api_base
            repo = repo if ok else context.STUDENT_REPO
            counting = "tokenizer" if context.load_tokenizer(repo, token) else "estimate"
            logger.info(f"token counting: {counting}", repo=repo)

    import tau2.evaluator.evaluator_nl_assertions as nl_assertions

    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = args.user_model

    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.runner import run_domain

    domain_tools = registry.get_env_constructor(args.domain)().get_tools()
    tool_names = {t.name for t in domain_tools}

    active_patches: list[str] = []
    if args.tool_patch_ids is not None and not args.tool_patches:
        raise SystemExit("--tool-patch-ids needs --tool-patches")
    if args.tool_patches:
        from shipwright.llm.prompted_tools import set_tool_patches
        from shipwright.heal.tool_fix import apply_patches, load_ledger, select_patches

        ledger = load_ledger(args.tool_patches)
        if args.tool_patch_ids is not None:
            ledger = select_patches(ledger, args.tool_patch_ids)
        # Fail now, not mid-run, if an applied patch does not fit the current tools.
        _, active_patches = apply_patches([t.openai_schema for t in domain_tools], ledger)
        set_tool_patches(ledger)
        logger.info(f"tool patches active: {active_patches or 'none'}", patches=active_patches)
    log.note_repro(tool_patch_ids_active=active_patches)

    run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_config["output_dir"] = str(out_dir)
    run_config["active_tool_patches"] = active_patches
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    agent_args: dict = {"temperature": args.agent_temperature, **(agent_llm_args or {})}
    if args.agent_api_base and not args.dry_run:
        # Per-call endpoint for the agent; vLLM ignores the key (no --api-key on the server).
        agent_args.update(api_base=args.agent_api_base, api_key="EMPTY")  # priced 0: our GPU, per hour

    config = TextRunConfig(
        domain=args.domain,
        agent="llm_agent",
        llm_agent=args.agent_model,
        llm_args_agent=agent_args,
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
    log.fail_point("summary")

    sims = [s.model_dump(mode="json") for s in results.simulations]
    traces = [build_trace(s, tool_names) for s in sims]
    write_jsonl(out_dir / "traces.jsonl", traces)
    summary = summarize(traces)
    summary["usage_by_role"] = meter.by_role()
    summary["cost_estimate_usd"] = cost_estimate(meter, args, role_names)
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
    export_spans(traces, out_dir, args, active_patches)
    log.note_repro(task_ids_run=sorted({str(t["task_id"]) for t in traces}, key=lambda x: (len(x), x)))
    logger.info("run done", run_dir=str(out_dir), runs=summary["num_runs"], passed=summary["num_passed"],
                infra_errors=summary["num_infra_errors"])
    return out_dir, traces, summary


def export_spans(traces: list[dict], out_dir: Path, args: argparse.Namespace, active_patches: list[str]) -> None:
    """OTel spans next to traces.jsonl (docs/ops/observability.md). Never raises."""
    try:
        from shipwright.observability.otel_export import export_run
    except Exception as e:  # noqa: BLE001 - e.g. opentelemetry not installed
        logger.warning(f"OTel spans skipped ({e})")
        return
    provider = "dry_run" if args.dry_run else (
        "openai" if args.agent_api_base else ("vertex" if args.agent_model.startswith("vertex/") else "huggingface"))
    export_run(traces, out_dir, {
        "agent_model": args.agent_model, "user_model": args.user_model, "domain": args.domain,
        "task_split": args.task_split, "provider": provider, "active_tool_patches": active_patches,
    })


def main() -> int:
    result = run(parse_args())
    if result is None:
        return 2
    out_dir, _, summary = result
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="run"))
