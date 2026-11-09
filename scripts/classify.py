"""Classify and route the failed runs of a baseline run dir.

Usage:
  .venv/bin/python scripts/classify.py runs/baseline-<timestamp> [--no-llm] [--judge-all]

Reads traces.jsonl (and tau2/results.json for the task instructions).
Writes failures.jsonl and failure_summary.json into the same dir.
The judge (--judge-model) defaults to vertex/openai/gpt-oss-120b-maas on Vertex AI (gcloud token,
src/shipwright/vertex.py). A raw HF router id such as openai/gpt-oss-120b:cerebras also works
(HF token read like baseline.py).
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
from shipwright.vertex import JUDGE_MODEL  # noqa: E402
from shipwright.classify import classify_run, summarize_failures, tool_example_values  # noqa: E402
from shipwright.traces import write_jsonl  # noqa: E402


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
    from shipwright import vertex

    url, body_model, headers = vertex.endpoint(model, token, HF_ROUTER)
    client = httpx.Client(timeout=120)

    def post(messages: list[dict]) -> httpx.Response:
        return client.post(url, headers=headers(),
                           json={"model": body_model, "messages": messages, "temperature": 0,
                                 "max_tokens": 2000, "reasoning_effort": "low"})

    def chat(messages: list[dict]) -> str:
        r = post(messages)
        if r.status_code == 401 and vertex.is_vertex_model(model):
            vertex.provider().invalidate()  # expired token: get a new one, retry once
            r = post(messages)
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""

    return chat


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--no-llm", action="store_true", help="rule labels only, no network")
    p.add_argument("--judge-all", action="store_true", help="also judge runs that rules already label")
    p.add_argument("--judge-model", default=JUDGE_MODEL,
                   help="raw HF router id (e.g. openai/gpt-oss-120b:cerebras) or vertex/<publisher>/<model>"
                        " (default: the team's choice, Vertex gpt-oss-120b)")
    args = p.parse_args()

    traces = [json.loads(line) for line in (args.run_dir / "traces.jsonl").read_text().splitlines() if line]
    instructions = load_instructions(args.run_dir)
    chat = None
    if not args.no_llm:
        try:
            chat = make_chat(load_token(), args.judge_model)
        except RuntimeError as e:
            print(f"{e} Use --no-llm, or provide the token.", file=sys.stderr)
            return 2

    examples = tool_example_values(load_tools(args.run_dir))
    rows = []
    for t in traces:
        row = classify_run(t, instructions.get(str(t.get("task_id"))), chat, args.judge_all, examples)
        if row:
            rows.append(row)
            print(f"task {row['task_id']}: {row['route']:10} {','.join(row['labels'])}")
    write_jsonl(args.run_dir / "failures.jsonl", rows)
    summary = summarize_failures(rows, len(traces))
    summary["judge_model"] = None if args.no_llm else args.judge_model
    (args.run_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
