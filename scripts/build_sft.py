"""Build the SFT dataset for the weight-fix path from successful runs.

  --teacher RUN_DIR   teacher runs (scripts/teacher_runs.py); repeatable
  --replay RUN_DIR    Tiny Aya's own runs; their successes are mixed in (replay); repeatable

Only runs with reward == 1 are used, and only from train-split runs (run_config.json).
Every example is rendered as Tiny Aya sees it (see shipwright.heal.sft_data), deduplicated,
and dropped if it has more than --max-tokens tokens or a reply over REPLY_TOKENS (context.py, 512 since 2026-11-14).

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
from shipwright.heal.traces import has_simulator_leak  # noqa: E402

DEFAULT_REPO = "mitushigoyal5/shipwright-sft"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Where the training examples come from.
    parser.add_argument("--teacher", type=Path, action="append", default=[], help="teacher run dir")
    parser.add_argument("--replay", type=Path, action="append", default=[], help="Tiny Aya run dir")
    parser.add_argument(
        "--only-model", default=None,
        help="use only teacher run dirs whose agent_model is exactly this (pooling runs of other teachers,"
        " whose licenses we have not checked, would put their outputs in the training data)",
    )
    parser.add_argument(
        "--boost-from", type=Path, default=None,
        help="a classified run of the candidate on train tasks (failures.jsonl): the teacher examples of the tasks it"
        " failed are repeated --boost times in the data, so the next adapter learns from its own failures",
    )
    parser.add_argument("--boost", type=int, default=1, help="with --boost-from: copies of each failed task's examples")
    parser.add_argument(
        "--general-replay", type=Path, default=None,
        help="a jsonl of general function-calling examples (scripts/build_bfcl_replay.py) added to the data as they are:"
        " they keep the adapter from answering in words where a function should be called (BFCL fell 43 to 34 without them)",
    )
    parser.add_argument(
        "--corrections", type=Path, default=None,
        help="a jsonl of corrective examples (scripts/build_corrections.py): the candidate's conversation up to a wrong "
        "turn, and the teacher's reply. Only that reply is trained (train_last_only)",
    )
    parser.add_argument("--max-replay", type=int, default=None, help="cap on replay examples")
    parser.add_argument("--allow-split", action="store_true", help="accept runs not on the train split")
    parser.add_argument(
        "--tool-patches", type=Path, default=None, metavar="LEDGER",
        help="render every example with the ledger's applied tool patches (the gate's tool list)",
    )

    # Length limits.
    parser.add_argument(
        "--long-mode",
        choices=["window", "drop"],
        default="window",
        help="runs longer than the context: cut into windows, or drop",
    )
    parser.add_argument("--max-tokens", type=int, default=context.CONTEXT_LIMIT)
    parser.add_argument(
        "--condensed-policy",
        action="store_true",
        help="render every row with the short retail policy (prompted_tools.CONDENSED_POLICY): for a candidate that "
        "serves with condensed_policy: true",
    )
    parser.add_argument(
        "--identity-note",
        action="store_true",
        help="trimmed training windows keep the user id and orders (context.still_true_note): for a candidate that "
        "serves with identity_note: true",
    )
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
    # "synthetic" tasks are generated from the train split and checked to share nothing with held-out or hidden tasks
    if task_split not in ("train", "synthetic") and not allow_split:
        raise SystemExit(
            f"{run_directory}: task_split is {task_split!r}, not 'train' or 'synthetic'. Never train on held-out tasks."
        )



def load_run(run_directory: Path, allow_split: bool) -> tuple[list[dict], dict]:
    """Successful traces and prompts of one run dir."""
    run_config_path = run_directory / "run_config.json"
    if run_config_path.exists():
        task_split = json.loads(run_config_path.read_text()).get("task_split")
    else:
        task_split = None
    check_train_split(run_directory, task_split, allow_split)

    # A passing run whose user simulator wrote past its turn (patches.cut_leaked_turn) has chat-format tokens in a
    # customer message: never train on it (rounds 1 to 3 on Vertex gpt-oss, 2026-11-14).
    successful_traces = [trace for trace in read_json_lines(run_directory / "traces.jsonl")
                         if trace.get("passed") and not has_simulator_leak(trace)]

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
        run_config_path = run_directory / "run_config.json"
        writer_model = None
        if run_config_path.exists():
            writer_model = json.loads(run_config_path.read_text()).get("agent_model")
        if build_options.only_model and writer_model != build_options.only_model:
            notes.append(f"{run_directory.name}: skipped, written by {writer_model}, not {build_options.only_model}")
            continue
        traces, prompts_by_sha = load_run(run_directory, build_options.allow_split)
        for trace in traces:
            prompt = prompt_for(trace, prompts_by_sha, None)
            if prompt is None:
                if rebuilt_prompt is None:
                    rebuilt_prompt = reconstruct_prompt()
                    notes.append("some runs had no prompts.json: agent prompt rebuilt from tau2")
                prompt = rebuilt_prompt
            prompt = patch_prompt(prompt, build_options.patch_rows)
            trace_records = sft_data.trace_records(trace, prompt, source, build_options.long_mode)
            for record in trace_records:  # which model wrote the targets (teacher id, or Tiny Aya for replay)
                record["model"] = writer_model
            records += trace_records
        notes.append(f"{run_directory.name}: {len(traces)} successful {source} runs")
    return records


def load_general_replay(path: Path, max_tokens: int, condensed_policy: bool = False) -> tuple[list[dict], dict]:
    """Pre-rendered rows (general function-calling examples, corrections) with the token counts the data statistics and
    the length filter need. ``condensed_policy`` rewrites a retail system prompt to the short policy (a BFCL row has no
    policy block and stays as it is)."""
    records = read_json_lines(path)
    for record in records:
        if condensed_policy:
            record["messages"] = sft_data.with_condensed_policy(record["messages"])
        record["tokens"] = sft_data.count_messages(record["messages"])
        record["reply_tokens_max"] = sft_data.longest_reply_tokens(record["messages"])
    return sft_data.filter_records(records, max_tokens)


def failed_task_ids(candidate_run: Path) -> list[str]:
    """Task ids of the failed runs in a classified run (failures.jsonl), the ones with a fix (not no_fix)."""
    task_ids: list[str] = []
    for failure in read_json_lines(candidate_run / "failures.jsonl"):
        task_id = str(failure["task_id"])
        if failure.get("route") != "no_fix" and task_id not in task_ids:
            task_ids.append(task_id)
    return task_ids


def push(output_directory: Path, repo: str, revision: str) -> None:
    from huggingface_hub import HfApi

    from shipwright import cloud

    token = cloud.hf_token()
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
    from shipwright import cloud

    if build_options.estimate_tokens:
        return "estimate"
    if context.load_tokenizer(context.STUDENT_REPO, cloud.hf_token()):
        return "tokenizer"
    return "estimate"


def main(argv: list[str] | None = None) -> int:
    build_options = parse_args(argv)
    if not build_options.teacher and not build_options.replay:
        raise SystemExit("Pass at least one --teacher or --replay dir.")

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
    from shipwright.llm import prompted_tools

    prompted_tools.set_condensed_policy(build_options.condensed_policy)
    context.set_identity_note(build_options.identity_note)
    if build_options.condensed_policy:
        notes.append("policy: condensed (prompted_tools.CONDENSED_POLICY)")
    build_options.patch_rows = None
    if build_options.tool_patches:
        patch_text = build_options.tool_patches.read_text()
        build_options.patch_rows = [json.loads(line) for line in patch_text.splitlines() if line.strip()]
        applied_count = sum(row.get("status") == "applied" for row in build_options.patch_rows)
        notes.append(f"tool list: {applied_count} applied patches from {build_options.tool_patches}")

    teacher_records = collect(build_options.teacher, "teacher", build_options, notes)
    replay_records = collect(build_options.replay, "replay", build_options, notes)
    if build_options.max_replay is not None:
        replay_records = replay_records[: build_options.max_replay]

    record_count_before_dedupe = len(teacher_records) + len(replay_records)
    deduplicated_records = sft_data.dedupe(teacher_records + replay_records)
    kept_records, dropped_counts = sft_data.filter_records(deduplicated_records, build_options.max_tokens)
    dropped_counts["duplicates"] = record_count_before_dedupe - len(deduplicated_records)

    boosted_task_ids: list[str] = []
    examples_before_boost = len(kept_records)
    if build_options.boost_from and build_options.boost > 1:
        boosted_task_ids = failed_task_ids(build_options.boost_from)
        boosted_records = [record for record in kept_records if str(record.get("task_id")) in boosted_task_ids]
        kept_records = kept_records + boosted_records * (build_options.boost - 1)
        notes.append(
            f"boost x{build_options.boost}: {len(boosted_task_ids)} tasks the candidate failed,"
            f" {len(boosted_records)} examples repeated"
        )

    general_replay_count = 0
    if build_options.general_replay:
        general_records, general_dropped = load_general_replay(build_options.general_replay, build_options.max_tokens,
                                                               build_options.condensed_policy)
        kept_records = kept_records + general_records
        general_replay_count = len(general_records)
        for reason, count in general_dropped.items():
            dropped_counts[reason] = dropped_counts.get(reason, 0) + count
        notes.append(f"general replay: {general_replay_count} examples from {build_options.general_replay}")

    corrections_count = 0
    if build_options.corrections:
        correction_records, correction_dropped = load_general_replay(build_options.corrections, build_options.max_tokens,
                                                                     build_options.condensed_policy)
        lost = [record for record in correction_records if sft_data.lost_authentication(record["messages"])]
        correction_records = [record for record in correction_records if not sft_data.lost_authentication(record["messages"])]
        correction_dropped["lost_authentication"] = len(lost)
        kept_records = kept_records + correction_records
        corrections_count = len(correction_records)
        for reason, count in correction_dropped.items():
            dropped_counts[reason] = dropped_counts.get(reason, 0) + count
        notes.append(f"corrections: {corrections_count} examples from {build_options.corrections}")

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
            "notes": notes,
            "boost": build_options.boost if boosted_task_ids else 1,
            "boosted_tasks": boosted_task_ids,
            "examples_before_boost": examples_before_boost,
            "general_replay": general_replay_count,
            "corrections": corrections_count,
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
