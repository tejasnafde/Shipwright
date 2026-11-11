"""Classify and route the failed runs of a baseline run dir.

Usage:
  .venv/bin/python scripts/classify.py runs/baseline-<timestamp> [--classifier-model <id>] [--judge]
  .venv/bin/python scripts/classify.py runs/baseline-<timestamp> --no-llm

Reads traces.jsonl (and tau2/results.json for the task instructions).
Writes audit.jsonl (AgentTune audit records), failures.jsonl and failure_summary.json into the
same dir. DETECT is AgentTune's FailureDetector plus the outcome (reward < 1); CLASSIFY is
AgentTune's FailureClassifier plus our rule labels. --no-llm: detector and rules only, no
network (CI). --judge: also run our own judge.

--classifier-model and --judge-model default to vertex/openai/gpt-oss-120b-maas on Vertex AI
(gcloud token, src/shipwright/llm/vertex.py; the team's choice, 2026-11-09). Any other litellm id
also works for the classifier; a raw HF router id such as openai/gpt-oss-120b:cerebras works
for the judge (HF token read like baseline.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baseline import HF_ROUTER, load_token  # noqa: E402
from shipwright.llm import vertex  # noqa: E402
from shipwright.heal.agenttune_bridge import classify_traces, make_classifier  # noqa: E402
from shipwright.heal.classify import summarize_failures, tool_example_values  # noqa: E402
from shipwright.llm.vertex import JUDGE_MODEL  # noqa: E402
from shipwright.heal.traces import write_jsonl  # noqa: E402


def load_instructions(run_directory: Path) -> dict[str, str]:
    """Task id -> the user instructions of that task, from tau2/results.json."""
    results_path = run_directory / "tau2" / "results.json"
    if not results_path.exists():
        return {}

    instructions_by_task_id = {}
    for task in json.loads(results_path.read_text()).get("tasks", []):
        instructions = (task.get("user_scenario") or {}).get("instructions")
        if isinstance(instructions, dict):
            instructions_text = json.dumps(instructions)
        else:
            instructions_text = str(instructions)
        instructions_by_task_id[str(task.get("id"))] = instructions_text
    return instructions_by_task_id


def load_tools(run_directory: Path) -> list[dict]:
    """The domain's tool schemas (from tau2), for the tool_example_copied rule."""
    run_config_path = run_directory / "run_config.json"
    if run_config_path.exists():
        domain = json.loads(run_config_path.read_text()).get("domain", "retail")
    else:
        domain = "retail"
    try:
        from tau2.registry import registry
    except ImportError:
        return []
    return [tool.openai_schema for tool in registry.get_env_constructor(domain)().get_tools()]


def make_chat(token: str | None, model: str):
    """A chat function for the judge or the patch proposer: HF router, or Vertex for vertex/ ids."""
    from shipwright.llm import vertex

    url, request_model, make_headers = vertex.endpoint(model, token, HF_ROUTER)
    http_client = httpx.Client(timeout=120)

    def post_chat_request(messages: list[dict]) -> httpx.Response:
        request_body = {
            "model": request_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2000,
            "reasoning_effort": "low",
        }
        return http_client.post(url, headers=make_headers(), json=request_body)

    def chat(messages: list[dict]) -> str:
        from shipwright.observability import instrument

        with instrument.external("llm_judge", model, role="judge") as call_record:
            response = post_chat_request(messages)
            call_record["retries"] = 0
            if response.status_code == 401 and vertex.is_vertex_model(model):
                vertex.provider().invalidate()  # expired token: get a new one, retry once
                response = post_chat_request(messages)
                call_record["retries"] = 1
            call_record["status"] = response.status_code
            response.raise_for_status()
            response_body = response.json()
            call_record.update(instrument.usage_fields(response_body))
            return response_body["choices"][0]["message"].get("content") or ""

    return chat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-llm", action="store_true", help="rule labels only, no network")

    # AgentTune's classifier.
    parser.add_argument(
        "--classifier-model",
        default=JUDGE_MODEL,
        help="AgentTune FailureClassifier model: vertex/<publisher>/<model> or any litellm id"
        " (default: the team's choice, Vertex gpt-oss-120b)",
    )
    parser.add_argument("--classifier-api-base", default=None, help="api_base for a non-vertex classifier model")

    # Our own judge.
    parser.add_argument("--judge", action="store_true", help="also run our own judge on runs no rule explains")
    parser.add_argument("--judge-all", action="store_true", help="with --judge: also judge runs that rules label")
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        help="raw HF router id (e.g. openai/gpt-oss-120b:cerebras) or vertex/<publisher>/<model>"
        " (default: the team's choice, Vertex gpt-oss-120b)",
    )
    return parser


def main() -> int:
    options = build_parser().parse_args()

    from shipwright.observability import log, repro

    log.attach_run_dir(options.run_dir)
    log.note_repro(**repro.models_from_args(options))
    traces = [json.loads(line) for line in (options.run_dir / "traces.jsonl").read_text().splitlines() if line]
    instructions = load_instructions(options.run_dir)

    judge_chat = None
    if options.judge and not options.no_llm:
        try:
            judge_chat = make_chat(load_token(), options.judge_model)
        except RuntimeError as token_error:
            print(f"{token_error} Use --no-llm, or provide the token.", file=sys.stderr)
            return 2

    tool_example_strings = tool_example_values(load_tools(options.run_dir))

    classifier = None
    if not options.no_llm:
        if vertex.is_vertex_model(options.classifier_model):
            try:
                vertex.provider().get()  # fail now if gcloud cannot mint a token
            except RuntimeError as token_error:
                print(f"{token_error} Use --no-llm, or fix the gcloud config.", file=sys.stderr)
                return 2
        classifier = make_classifier(options.classifier_model, options.classifier_api_base)

    failure_rows = classify_traces(
        traces,
        options.run_dir,
        instructions=instructions,
        chat=judge_chat,
        judge_all=options.judge_all,
        example_values=tool_example_strings,
        classifier=classifier,
    )
    for row in failure_rows:
        detected_by = "+".join(row["detected_by"])
        print(
            f"task {row['task_id']}: {row['route']:10} {','.join(row['labels'])}"
            f"  [{detected_by}; agenttune: {row['agenttune_root_cause']}]"
        )
    write_jsonl(options.run_dir / "failures.jsonl", failure_rows)

    summary = summarize_failures(failure_rows, len(traces))
    summary["judge_model"] = options.judge_model if judge_chat else None
    summary["classifier_model"] = options.classifier_model if classifier else None
    (options.run_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="classify"))
