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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.observability import log  # noqa: E402

logger = log.get_logger("baseline")

HF_ROUTER = "https://router.huggingface.co/v1"
DEFAULT_AGENT = "openai/CohereLabs/tiny-aya-global:cohere"
DEFAULT_USER = "openai/openai/gpt-oss-120b:cerebras"


@dataclass
class RunSetup:
    """What prepare() builds before tau2 runs, and finish_run() reads afterwards."""

    usage_meter: object  # shipwright.llm.costs.UsageMeter
    model_rerouter: object  # shipwright.llm.patches.ModelRerouter
    tool_names: set[str]
    active_patches: list[str]


@dataclass
class RunResult:
    run_directory: Path
    traces: list[dict]
    summary: dict


def load_token() -> str | None:
    secrets_directory = os.environ.get("SWITCHBOARD_SECRETS_DIR")
    if secrets_directory:
        token_path = Path(secrets_directory) / "HF_TOKEN"
        try:
            token = token_path.read_text().strip()
            if token:
                return token
        except OSError:
            pass
    return os.environ.get("HF_TOKEN") or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Which tasks to run.
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--num-tasks", type=int, default=20, help="0 means all tasks of the split")
    parser.add_argument("--task-split", default="train", help="tau2 split: train, test or base")
    parser.add_argument("--task-ids", nargs="*", default=None, help="explicit task ids (overrides --num-tasks)")
    parser.add_argument("--trials", type=int, default=1)

    # Which models play the agent and the user.
    parser.add_argument("--agent-model", default=DEFAULT_AGENT)
    parser.add_argument(
        "--user-model",
        default=DEFAULT_USER,
        help="user simulator (and tau2's judge): an HF router litellm id or vertex/<publisher>/<model>",
    )
    parser.add_argument("--agent-temperature", type=float, default=0.0)
    parser.add_argument(
        "--agent-api-base",
        default=None,
        help="OpenAI-compatible endpoint for the agent only, e.g. http://localhost:8000/v1 (vLLM)."
        " The user simulator still goes to the HF router.",
    )

    # How the run executes.
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="fake LLM, no network")
    parser.add_argument(
        "--open-hidden",
        action="store_true",
        help="allow hidden test tasks (configs/splits.yaml): only for the one final-report run",
    )

    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="stop the run once its metered cost passes this (the remaining tasks end as infra errors)",
    )

    # Tool patches (the tool fix).
    parser.add_argument(
        "--tool-patches",
        type=Path,
        default=None,
        help="ledger (patches/ledger.jsonl); its 'applied' patches change the Tiny Aya tool prompt",
    )
    parser.add_argument(
        "--tool-patch-ids",
        nargs="*",
        default=None,
        help="with --tool-patches: use exactly these patch ids, whatever their ledger status (gate)",
    )
    parser.add_argument(
        "--tool-guards",
        action="store_true",
        help="policy checks on the agent's tool calls before they run (src/shipwright/llm/guards.py)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def cost_estimate(
    usage_meter,
    run_options: argparse.Namespace,
    role_names: dict[str, str] | None = None,
) -> dict:
    """Per-role cost from the usage meter, priced by provider (src/shipwright/llm/costs.py)."""
    from shipwright.llm.costs import estimate, provider_of

    provider_by_model = {
        run_options.user_model: provider_of(run_options.user_model, dry_run=run_options.dry_run),
        run_options.agent_model: provider_of(
            run_options.agent_model, run_options.agent_api_base, run_options.dry_run
        ),
    }
    return estimate(usage_meter.rows, provider_by_model, role_names)


def run(
    run_options: argparse.Namespace,
    *,
    prefix: str = "baseline",
    prompted_models: dict[str, int] | None = None,
    agent_llm_args: dict | None = None,
    role_names: dict[str, str] | None = None,
) -> RunResult | None:
    """Run tau2 with the given options. Returns None without a token.

    ``prompted_models`` maps extra model names to their reply-token budget: they get the
    same prompted-tools view as Tiny Aya (used for the teacher). ``agent_llm_args`` is
    merged into the agent's litellm args (e.g. reasoning_effort). ``role_names`` renames
    roles in the cost estimate (teacher runs: agent -> teacher).
    """
    run_directory = make_run_dir(run_options, prefix)
    run_setup = prepare(run_options, run_directory, prompted_models=prompted_models)
    if run_setup is None:
        return None

    from tau2.runner import run_domain

    tau2_config = text_run_config(run_options, run_directory, agent_llm_args)
    tau2_results = run_domain(tau2_config)
    log.fail_point("summary")
    simulations = [simulation.model_dump(mode="json") for simulation in tau2_results.simulations]
    return finish_run(run_options, run_directory, simulations, run_setup, role_names)


def make_run_dir(run_options: argparse.Namespace, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_options.output_dir:
        run_directory = run_options.output_dir.resolve()
    else:
        run_directory = (ROOT / "runs" / f"{prefix}-{timestamp}").resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    log.attach_run_dir(run_directory)
    return run_directory


def check_task_split(run_options: argparse.Namespace) -> None:
    """Refuse hidden tasks unless --open-hidden was passed (src/shipwright/heal/splits.py)."""
    from shipwright.heal import splits

    open_hidden = getattr(run_options, "open_hidden", False)
    try:
        splits.check_run(run_options.task_split, run_options.task_ids, open_hidden)
    except ValueError as split_error:
        raise SystemExit(str(split_error)) from None
    if open_hidden:
        logger.warning("hidden split opened: this run is for the final report only", task_ids=run_options.task_ids)


def configure_credentials(run_options: argparse.Namespace) -> tuple[bool, str | None]:
    """Set the litellm environment for the run.

    Returns (ready, hf_token). ready is False when a required token is missing or
    Vertex cannot mint one.
    """
    from shipwright.llm import vertex

    agent_uses_vertex = vertex.is_vertex_model(run_options.agent_model)
    user_uses_vertex = vertex.is_vertex_model(run_options.user_model)

    if run_options.dry_run:
        os.environ.setdefault("OPENAI_API_KEY", "dry-run")
        return True, None

    agent_is_not_on_hf_router = agent_uses_vertex or bool(run_options.agent_api_base)
    needs_hf_token = not (user_uses_vertex and agent_is_not_on_hf_router)
    hf_token = load_token()
    if needs_hf_token and not hf_token:
        logger.error("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.")
        return False, None

    if hf_token:
        # litellm's openai provider reads these; nothing token-bearing goes into tau2 config.
        # vertex/ calls pass their own api_base and api_key per call.
        os.environ["OPENAI_API_KEY"] = hf_token
        os.environ["OPENAI_API_BASE"] = HF_ROUTER
        os.environ["OPENAI_BASE_URL"] = HF_ROUTER

    if agent_uses_vertex or user_uses_vertex:
        try:
            vertex.provider().get()  # fail now, not mid-run, if gcloud cannot mint a token
        except RuntimeError:
            logger.error("Vertex token failed", exc_info=True)
            return False, None
        logger.info("vertex endpoint", project=vertex.project(), location=vertex.location())
    return True, hf_token


def install_llm_patches(
    run_options: argparse.Namespace,
    hf_token: str | None,
    prompted_models: dict[str, int] | None,
):
    """Patch litellm for tau2. Returns (usage_meter, model_rerouter)."""
    from shipwright.llm import patches, prompted_tools
    from shipwright.llm.costs import SpendGuard, UsageMeter

    for model_name, reply_tokens in (prompted_models or {}).items():
        prompted_tools.force_prompted(model_name, reply_tokens)

    # tau2 has built-in default models (NL-assertion judge gpt-5.4-mini, env interface,
    # reviewers). Send every model other than agent and user to the user model.
    model_rerouter = patches.ModelRerouter(
        {run_options.agent_model, run_options.user_model}, run_options.user_model
    )
    usage_meter = UsageMeter()
    patch_roles = {
        "meter": usage_meter,
        "agent_model": run_options.agent_model,
        "user_model": run_options.user_model,
    }
    if run_options.max_cost_usd is not None:
        patch_roles["spend_guard"] = SpendGuard(
            run_options.max_cost_usd,
            lambda: cost_estimate(usage_meter, run_options)["total"],
        )

    if run_options.dry_run:
        from shipwright.llm.dry_run import make_fake_completion

        completion = make_fake_completion(run_options.agent_model)
        patches.install(completion, model_map=model_rerouter, **patch_roles)
    else:
        patches.install(model_map=model_rerouter, **patch_roles)
        if prompted_tools.is_prompted_model(run_options.agent_model):
            log_token_counting_method(run_options, hf_token)
    return usage_meter, model_rerouter


def log_token_counting_method(run_options: argparse.Namespace, hf_token: str | None) -> None:
    """Count tokens with the student's tokenizer: the teacher gets the student's budget."""
    from shipwright.llm import context

    # A vLLM served name (e.g. a LoRA module) is not a repo id: use the student repo.
    repo_id = run_options.agent_model.removeprefix("openai/").split(":")[0]
    is_student_repo = "tiny-aya" in repo_id.lower() and "/" in repo_id and not run_options.agent_api_base
    if not is_student_repo:
        repo_id = context.STUDENT_REPO
    if context.load_tokenizer(repo_id, hf_token):
        counting_method = "tokenizer"
    else:
        counting_method = "estimate"
    logger.info(f"token counting: {counting_method}", repo=repo_id)


def activate_tool_patches(run_options: argparse.Namespace, domain_tools) -> list[str]:
    """Load the patch ledger into the prompt builder. Returns the ids of the active patches."""
    if run_options.tool_patch_ids is not None and not run_options.tool_patches:
        raise SystemExit("--tool-patch-ids needs --tool-patches")
    if not run_options.tool_patches:
        return []

    from shipwright.heal.tool_fix import apply_patches, load_ledger, select_patches
    from shipwright.llm.prompted_tools import set_tool_patches

    ledger = load_ledger(run_options.tool_patches)
    if run_options.tool_patch_ids is not None:
        ledger = select_patches(ledger, run_options.tool_patch_ids)
    # Fail now, not mid-run, if an applied patch does not fit the current tools.
    _, active_patches = apply_patches([tool.openai_schema for tool in domain_tools], ledger)
    set_tool_patches(ledger)
    logger.info(f"tool patches active: {active_patches or 'none'}", patches=active_patches)
    return active_patches


def write_run_config(
    run_options: argparse.Namespace,
    run_directory: Path,
    active_patches: list[str],
) -> None:
    """run_config.json: the options of this run, with Paths as strings (no secrets)."""
    run_config = {}
    for option_name, option_value in vars(run_options).items():
        if isinstance(option_value, Path):
            run_config[option_name] = str(option_value)
        else:
            run_config[option_name] = option_value
    run_config["output_dir"] = str(run_directory)
    run_config["active_tool_patches"] = active_patches
    from shipwright.llm import prompted_tools

    run_config["tool_list_version"] = prompted_tools.TOOL_LIST_VERSION
    (run_directory / "run_config.json").write_text(json.dumps(run_config, indent=2))


def prepare(
    run_options: argparse.Namespace,
    run_directory: Path,
    *,
    prompted_models: dict[str, int] | None = None,
) -> RunSetup | None:
    """Everything before tau2 runs: tokens, litellm patches, tool patches, run_config.json.

    Returns None without a token.
    """
    from shipwright.observability import repro

    check_task_split(run_options)
    log.note_repro(**repro.models_from_args(run_options), dry_run=run_options.dry_run)
    logger.info(
        "run start",
        run_dir=str(run_directory),
        agent_model=run_options.agent_model,
        user_model=run_options.user_model,
        task_split=run_options.task_split,
        dry_run=run_options.dry_run,
    )

    credentials_ready, hf_token = configure_credentials(run_options)
    if not credentials_ready:
        return None

    usage_meter, model_rerouter = install_llm_patches(
        run_options, hf_token, prompted_models
    )

    import tau2.evaluator.evaluator_nl_assertions as nl_assertions
    from tau2.registry import registry

    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = run_options.user_model
    domain_tools = registry.get_env_constructor(run_options.domain)().get_tools()
    tool_names = {tool.name for tool in domain_tools}

    active_patches = activate_tool_patches(run_options, domain_tools)
    from shipwright.llm import guards

    guards.set_enabled(bool(getattr(run_options, "tool_guards", False)))
    if guards.is_enabled():
        logger.info("tool guards active", max_retries=guards.MAX_GUARD_RETRIES)
    log.note_repro(tool_patch_ids_active=active_patches)
    write_run_config(run_options, run_directory, active_patches)
    return RunSetup(
        usage_meter=usage_meter,
        model_rerouter=model_rerouter,
        tool_names=tool_names,
        active_patches=active_patches,
    )


def text_run_config(
    run_options: argparse.Namespace,
    run_directory: Path,
    agent_llm_args: dict | None = None,
):
    """The tau2 TextRunConfig of a run."""
    from tau2.data_model.simulation import TextRunConfig

    agent_args: dict = {"temperature": run_options.agent_temperature}
    if agent_llm_args:
        agent_args.update(agent_llm_args)
    if run_options.agent_api_base and not run_options.dry_run:
        # Per-call endpoint for the agent; vLLM ignores the key (no --api-key on the server).
        agent_args.update(api_base=run_options.agent_api_base, api_key="EMPTY")  # priced 0: our GPU, per hour

    if run_options.task_ids:
        task_ids = run_options.task_ids
        num_tasks = None
    else:
        task_ids = None
        num_tasks = run_options.num_tasks if run_options.num_tasks else None

    return TextRunConfig(
        domain=run_options.domain,
        agent="llm_agent",
        llm_agent=run_options.agent_model,
        llm_args_agent=agent_args,
        user="user_simulator",
        llm_user=run_options.user_model,
        llm_args_user={"temperature": 0.0},
        task_split_name=run_options.task_split,
        task_ids=task_ids,
        num_tasks=num_tasks,
        num_trials=run_options.trials,
        max_steps=run_options.max_steps,
        max_concurrency=run_options.max_concurrency,
        seed=run_options.seed,
        save_to=str(run_directory / "tau2"),  # absolute path: tau2 saves here, not under its data dir
        max_retries=0,
        auto_resume=True,  # a rerun into the same dir skips decided runs and redoes infra errors
    )


def finish_run(
    run_options: argparse.Namespace,
    run_directory: Path,
    simulations: list[dict],
    run_setup: RunSetup,
    role_names: dict[str, str] | None = None,
    extra_trace_fields: dict | None = None,
) -> RunResult:
    """traces.jsonl, summary.json, prompts.json and spans from finished simulations.

    ``extra_trace_fields`` maps a simulation id to fields merged into its trace.
    """
    from shipwright.heal.traces import build_trace, summarize, write_jsonl
    from shipwright.llm import prompted_tools

    traces = [build_trace(simulation, run_setup.tool_names) for simulation in simulations]
    for trace, simulation in zip(traces, simulations, strict=True):
        if extra_trace_fields:
            trace.update(extra_trace_fields.get(simulation.get("id"), {}))
    write_jsonl(run_directory / "traces.jsonl", traces)

    summary = summarize(traces)
    summary["usage_by_role"] = run_setup.usage_meter.by_role()
    summary["cost_estimate_usd"] = cost_estimate(run_setup.usage_meter, run_options, role_names)
    summary["max_cost_usd"] = run_options.max_cost_usd
    summary["cost_cap_hit"] = any("max-cost-usd" in str(trace.get("error") or "") for trace in traces)
    summary["agent_model"] = run_options.agent_model
    summary["user_model"] = run_options.user_model
    summary["domain"] = run_options.domain
    summary["task_split"] = run_options.task_split
    summary["rerouted_models"] = run_setup.model_rerouter.counts
    summary["active_tool_patches"] = run_setup.active_patches
    summary["tool_guards"] = bool(getattr(run_options, "tool_guards", False))
    (run_directory / "summary.json").write_text(json.dumps(summary, indent=2))

    recorded_prompts = prompted_tools.recorded_prompts()
    if recorded_prompts:
        (run_directory / "prompts.json").write_text(json.dumps(recorded_prompts, indent=1))

    export_spans(traces, run_directory, run_options, run_setup.active_patches)
    task_ids_run = sorted({str(trace["task_id"]) for trace in traces}, key=lambda task_id: (len(task_id), task_id))
    log.note_repro(task_ids_run=task_ids_run)
    logger.info(
        "run done",
        run_dir=str(run_directory),
        runs=summary["num_runs"],
        passed=summary["num_passed"],
        infra_errors=summary["num_infra_errors"],
    )
    return RunResult(run_directory=run_directory, traces=traces, summary=summary)


def provider_name(run_options: argparse.Namespace) -> str:
    """The agent's provider, as recorded on the spans."""
    if run_options.dry_run:
        return "dry_run"
    if run_options.agent_api_base:
        return "openai"
    if run_options.agent_model.startswith("vertex/"):
        return "vertex"
    return "huggingface"


def export_spans(
    traces: list[dict],
    run_directory: Path,
    run_options: argparse.Namespace,
    active_patches: list[str],
) -> None:
    """OTel spans next to traces.jsonl (docs/ops/observability.md). Never raises."""
    try:
        from shipwright.observability.otel_export import export_run
    except Exception as import_error:  # noqa: BLE001 - e.g. opentelemetry not installed
        logger.warning(f"OTel spans skipped ({import_error})")
        return
    export_run(
        traces,
        run_directory,
        {
            "agent_model": run_options.agent_model,
            "user_model": run_options.user_model,
            "domain": run_options.domain,
            "task_split": run_options.task_split,
            "provider": provider_name(run_options),
            "active_tool_patches": active_patches,
        },
    )


def main() -> int:
    result = run(parse_args())
    if result is None:
        return 2
    print(json.dumps(result.summary, indent=2))
    print(f"\nWrote {result.run_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="run"))
