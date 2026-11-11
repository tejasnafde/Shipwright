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


def load_instructions(run_dir: Path) -> dict[str, str]:
    path = run_dir / "tau2" / "results.json"
    if not path.exists():
        return {}
    out = {}
    for task in json.loads(path.read_text()).get("tasks", []):
        instr = (task.get("user_scenario") or {}).get("instructions")
        out[str(task.get("id"))] = json.dumps(instr) if isinstance(instr, dict) else str(instr)
    return out


def load_tools(run_dir: Path) -> list[dict]:
    """The domain's tool schemas (from tau2), for the tool_example_copied rule."""
    cfg = run_dir / "run_config.json"
    domain = json.loads(cfg.read_text()).get("domain", "retail") if cfg.exists() else "retail"
    try:
        from tau2.registry import registry
    except ImportError:
        return []
    return [t.openai_schema for t in registry.get_env_constructor(domain)().get_tools()]


def make_chat(token: str | None, model: str):
    """A chat function for the judge or the patch proposer: HF router, or Vertex for vertex/ ids."""
    from shipwright.llm import vertex

    url, body_model, headers = vertex.endpoint(model, token, HF_ROUTER)
    client = httpx.Client(timeout=120)

    def post(messages: list[dict]) -> httpx.Response:
        return client.post(url, headers=headers(),
                           json={"model": body_model, "messages": messages, "temperature": 0,
                                 "max_tokens": 2000, "reasoning_effort": "low"})

    def once(messages: list[dict]) -> str:
        from shipwright.observability import instrument

        with instrument.external("llm_judge", model, role="judge") as call:
            r = post(messages)
            call["retries"] = 0
            if r.status_code == 401 and vertex.is_vertex_model(model):
                vertex.provider().invalidate()  # expired token: get a new one, retry once
                r = post(messages)
                call["retries"] = 1
            call["status"] = r.status_code
            r.raise_for_status()
            body = r.json()
            call.update(instrument.usage_fields(body))
            return body["choices"][0]["message"].get("content") or ""

    def chat(messages: list[dict]) -> str:
        from shipwright.observability import retry

        return retry.call_with_retry(once, messages, policy=retry.LLM_USER, what=f"llm_judge {model}")

    return chat


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--no-llm", action="store_true", help="rule labels only, no network")
    p.add_argument("--classifier-model", default=JUDGE_MODEL,
                   help="AgentTune FailureClassifier model: vertex/<publisher>/<model> or any litellm id"
                        " (default: the team's choice, Vertex gpt-oss-120b)")
    p.add_argument("--classifier-api-base", default=None, help="api_base for a non-vertex classifier model")
    p.add_argument("--judge", action="store_true", help="also run our own judge on runs no rule explains")
    p.add_argument("--judge-all", action="store_true", help="with --judge: also judge runs that rules label")
    p.add_argument("--judge-model", default=JUDGE_MODEL,
                   help="raw HF router id (e.g. openai/gpt-oss-120b:cerebras) or vertex/<publisher>/<model>"
                        " (default: the team's choice, Vertex gpt-oss-120b)")
    args = p.parse_args()

    from shipwright.observability import log, repro

    log.attach_run_dir(args.run_dir)
    log.note_repro(**repro.models_from_args(args))
    traces = [json.loads(line) for line in (args.run_dir / "traces.jsonl").read_text().splitlines() if line]
    instructions = load_instructions(args.run_dir)
    chat = None
    if args.judge and not args.no_llm:
        try:
            chat = make_chat(load_token(), args.judge_model)
        except RuntimeError as e:
            print(f"{e} Use --no-llm, or provide the token.", file=sys.stderr)
            return 2

    examples = tool_example_values(load_tools(args.run_dir))
    classifier = None
    if not args.no_llm:
        if vertex.is_vertex_model(args.classifier_model):
            try:
                vertex.provider().get()  # fail now if gcloud cannot mint a token
            except RuntimeError as e:
                print(f"{e} Use --no-llm, or fix the gcloud config.", file=sys.stderr)
                return 2
        classifier = make_classifier(args.classifier_model, args.classifier_api_base)
    rows = classify_traces(traces, args.run_dir, instructions=instructions, chat=chat,
                           judge_all=args.judge_all, example_values=examples, classifier=classifier)
    for row in rows:
        print(f"task {row['task_id']}: {row['route']:10} {','.join(row['labels'])}"
              f"  [{'+'.join(row['detected_by'])}; agenttune: {row['agenttune_root_cause']}]")
    write_jsonl(args.run_dir / "failures.jsonl", rows)
    summary = summarize_failures(rows, len(traces))
    summary["judge_model"] = args.judge_model if chat else None
    summary["classifier_model"] = args.classifier_model if classifier else None
    (args.run_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="classify"))
