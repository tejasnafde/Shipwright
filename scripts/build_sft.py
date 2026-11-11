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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Where the training examples come from.
    parser.add_argument("--teacher", type=Path, action="append", default=[], help="teacher run dir")
    parser.add_argument("--replay", type=Path, action="append", default=[], help="Tiny Aya run dir")
    parser.add_argument("--max-replay", type=int, default=None, help="cap on replay examples")
    parser.add_argument("--agent-r", type=Path, action="append", default=[], help="agent_r_revise.py output dir")
    parser.add_argument("--allow-split", action="store_true", help="accept runs not on the train split")

    # Mixing in general chat data.
    parser.add_argument("--general", type=Path, default=None, help="general chat jsonl for the eta mix")
    parser.add_argument("--eta", type=float, default=0.2, help="agent share of the data when --general is set")

    # Length limits.
    parser.add_argument(
        "--long-mode",
        choices=["window", "drop"],
        default="window",
        help="runs longer than the context: cut into windows, or drop",
    )
    parser.add_argument("--max-tokens", type=int, default=context.CONTEXT_LIMIT)
    parser.add_argument(
        "--estimate-tokens",
        action="store_true",
        help="count tokens with the chars estimate (no tokenizer download)",
    )

    # Output.
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--push", action="store_true", help="upload to the HF dataset repo")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main", help="dataset branch, e.g. round-1")
    return parser.parse_args(argv)


def read_json_lines(path: Path) -> list[dict]:
    """Every non-empty line of a .jsonl file, parsed."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_train_split(run_directory: Path, task_split: str | None, allow_split: bool) -> None:
    if task_split != "train" and not allow_split:
        raise SystemExit(
            f"{run_directory}: task_split is {task_split!r}, not 'train'. Never train on held-out tasks."
        )


def load_run(run_directory: Path, allow_split: bool) -> tuple[list[dict], dict]:
    """Successful traces and prompts of one run dir."""
    run_config_path = run_directory / "run_config.json"
    if run_config_path.exists():
        task_split = json.loads(run_config_path.read_text()).get("task_split")
    else:
        task_split = None
    check_train_split(run_directory, task_split, allow_split)

    successful_traces = [trace for trace in read_json_lines(run_directory / "traces.jsonl") if trace.get("passed")]

    prompts_path = run_directory / "prompts.json"
    if prompts_path.exists():
        prompts_by_sha = json.loads(prompts_path.read_text())
    else:
        prompts_by_sha = {}
    return successful_traces, prompts_by_sha


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

    tau2_config = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        llm_agent="agent",
        user="user_simulator",
        llm_user="user",
        task_split_name="train",
        seed=seed,
    )
    first_task = registry.get_tasks_loader(domain)(task_split_name="train")[0]
    language_guidance = agent_language_guidance(task_context(tau2_config, first_task))
    environment = registry.get_env_constructor(domain)()
    domain_policy = agent_domain_policy(environment, language_guidance)
    system_prompt = SYSTEM_PROMPT.format(domain_policy=domain_policy, agent_instruction=AGENT_INSTRUCTION)
    return {"system": system_prompt, "tools": [tool.openai_schema for tool in environment.get_tools()]}


def prompt_for(trace: dict, prompts_by_sha: dict, fallback_prompt: dict | None) -> dict | None:
    """The recorded prompt of a trace, or fallback_prompt if it cannot be identified."""
    prompt_shas = {turn.get("prompt_sha") for turn in trace["agent_turns"] if turn.get("prompt_sha")}
    if len(prompt_shas) == 1:
        only_sha = next(iter(prompt_shas))
        if only_sha in prompts_by_sha:
            return prompts_by_sha[only_sha]
    if len(prompts_by_sha) == 1 and not prompt_shas:
        return next(iter(prompts_by_sha.values()))
    return fallback_prompt


def collect(
    run_directories: list[Path],
    source: str,
    build_options: argparse.Namespace,
    notes: list[str],
) -> list[dict]:
    records: list[dict] = []
    rebuilt_prompt = None
    for run_directory in run_directories:
        traces, prompts_by_sha = load_run(run_directory, build_options.allow_split)
        for trace in traces:
            prompt = prompt_for(trace, prompts_by_sha, None)
            if prompt is None:
                if rebuilt_prompt is None:
                    rebuilt_prompt = reconstruct_prompt()
                    notes.append("some runs had no prompts.json: agent prompt rebuilt from tau2")
                prompt = rebuilt_prompt
            records += sft_data.trace_records(trace, prompt, source, build_options.long_mode)
        notes.append(f"{run_directory.name}: {len(traces)} successful {source} runs")
    return records


def collect_agent_r(
    run_directories: list[Path],
    build_options: argparse.Namespace,
    notes: list[str],
) -> list[dict]:
    """Revision records (masked) and good-leaf records from agent_r_revise.py output dirs."""
    records: list[dict] = []
    for run_directory in run_directories:
        run_config = json.loads((run_directory / "run_config.json").read_text())
        check_train_split(run_directory, run_config.get("task_split"), build_options.allow_split)

        prompts_path = run_directory / "prompts.json"
        if prompts_path.exists():
            prompts_by_sha = json.loads(prompts_path.read_text())
        else:
            prompts_by_sha = {}
        if prompts_by_sha:
            fallback_prompt = next(iter(prompts_by_sha.values()))
        else:
            fallback_prompt = reconstruct_prompt()

        revisions = read_json_lines(run_directory / "revisions.jsonl")
        for revision in revisions:
            prompt = prompts_by_sha.get(revision.get("prompt_sha")) or fallback_prompt
            records += sft_data.revision_records(revision, prompt, build_options.long_mode)

        good_traces = read_json_lines(run_directory / "good.jsonl")
        for trace in good_traces:
            prompt = prompt_for(trace, prompts_by_sha, fallback_prompt)
            records += sft_data.trace_records(trace, prompt, "good", build_options.long_mode)

        notes.append(
            f"{run_directory.name}: {len(revisions)} Agent-R revision runs, {len(good_traces)} good MCTS leaves"
        )
    return records


def push(output_directory: Path, repo: str, revision: str) -> None:
    from huggingface_hub import HfApi

    import baseline

    token = baseline.load_token()
    if not token:
        raise SystemExit("No HF token for --push.")
    hf_api = HfApi(token=token)
    hf_api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    if revision != "main":
        hf_api.create_branch(repo, repo_type="dataset", branch=revision, exist_ok=True)

    from shipwright.observability import instrument

    for file_name in ("sft.jsonl", "sft_stats.json"):
        with instrument.external("hf_hub_upload", f"datasets/{repo}@{revision}/{file_name}"):
            hf_api.upload_file(
                path_or_fileobj=str(output_directory / file_name),
                path_in_repo=file_name,
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                commit_message=f"SFT data ({output_directory.name})",
            )
    print(f"Pushed to https://huggingface.co/datasets/{repo}/tree/{revision}")


def choose_token_counting_method(build_options: argparse.Namespace) -> str:
    """"tokenizer" if the student's tokenizer loads, else the chars "estimate"."""
    import baseline

    if build_options.estimate_tokens:
        return "estimate"
    if context.load_tokenizer(context.STUDENT_REPO, baseline.load_token()):
        return "tokenizer"
    return "estimate"


def main(argv: list[str] | None = None) -> int:
    build_options = parse_args(argv)
    if not build_options.teacher and not build_options.replay and not build_options.agent_r:
        raise SystemExit("Pass at least one --teacher, --replay or --agent-r dir.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if build_options.output_dir:
        output_directory = build_options.output_dir.resolve()
    else:
        output_directory = (ROOT / "runs" / f"sft-{timestamp}").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    from shipwright.observability import log

    log.attach_run_dir(output_directory)

    token_counting_method = choose_token_counting_method(build_options)
    notes: list[str] = [f"token counting: {token_counting_method}"]

    teacher_records = collect(build_options.teacher, "teacher", build_options, notes)
    replay_records = collect(build_options.replay, "replay", build_options, notes)
    if build_options.max_replay is not None:
        replay_records = replay_records[: build_options.max_replay]
    agent_r_records = collect_agent_r(build_options.agent_r, build_options, notes)

    record_count_before_dedupe = len(teacher_records) + len(replay_records) + len(agent_r_records)
    deduplicated_records = sft_data.dedupe(agent_r_records + teacher_records + replay_records)
    kept_records, dropped_counts = sft_data.filter_records(deduplicated_records, build_options.max_tokens)
    dropped_counts["duplicates"] = record_count_before_dedupe - len(deduplicated_records)

    eta_mix_report = None
    if build_options.general:
        general_rows = read_json_lines(build_options.general)
        general_records, _ = sft_data.filter_records(
            sft_data.general_records(general_rows), build_options.max_tokens
        )
        kept_records, eta_mix_report = sft_data.eta_mix(kept_records, general_records, build_options.eta)
        eta_mix_report["general_file"] = str(build_options.general)

    with open(output_directory / "sft.jsonl", "w") as sft_file:
        for record in kept_records:
            sft_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    statistics = sft_data.stats(
        kept_records,
        dropped_counts,
        {
            "token_count_method": token_counting_method,
            "long_mode": build_options.long_mode,
            "max_tokens_limit": build_options.max_tokens,
            "teacher_runs": [str(path) for path in build_options.teacher],
            "replay_runs": [str(path) for path in build_options.replay],
            "agent_r_dirs": [str(path) for path in build_options.agent_r],
            "eta_mix": eta_mix_report,
            "notes": notes,
        },
    )
    (output_directory / "sft_stats.json").write_text(json.dumps(statistics, indent=2))
    print(json.dumps(statistics, indent=2))
    print(f"\nWrote {output_directory}")
    if build_options.push:
        push(output_directory, build_options.repo, build_options.revision)
    return 0


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="sft"))
