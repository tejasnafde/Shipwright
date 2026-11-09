"""Classify and route the failed runs of a baseline run dir.

Usage:
  .venv/bin/python scripts/classify.py runs/baseline-<timestamp> --classifier-model <litellm id>
  .venv/bin/python scripts/classify.py runs/baseline-<timestamp> --no-llm

Reads traces.jsonl (and tau2/results.json for the task instructions).
Writes audit.jsonl (AgentTune audit records), failures.jsonl and failure_summary.json into the
same dir. DETECT is AgentTune's FailureDetector plus the outcome (reward < 1); CLASSIFY is
AgentTune's FailureClassifier (--classifier-model, a litellm model id) plus our rule labels.
--no-llm: detector and rules only, no network (CI). --judge: also run our own judge.
Our own judge (--judge) is openai/gpt-oss-120b:cerebras via the HF router. The token is read like baseline.py.
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
from shipwright.agenttune_bridge import classify_traces, make_classifier  # noqa: E402
from shipwright.classify import summarize_failures, tool_example_values  # noqa: E402
from shipwright.traces import write_jsonl  # noqa: E402

JUDGE_MODEL = "openai/gpt-oss-120b:cerebras"


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


def make_chat(token: str, model: str):
    client = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {token}"})

    def chat(messages: list[dict]) -> str:
        r = client.post(f"{HF_ROUTER}/chat/completions",
                        json={"model": model, "messages": messages, "temperature": 0,
                              "max_tokens": 2000, "reasoning_effort": "low"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""

    return chat


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--no-llm", action="store_true", help="rule labels only, no network")
    p.add_argument("--classifier-model", default=None,
                   help="litellm model id for AgentTune's FailureClassifier (required unless --no-llm)")
    p.add_argument("--classifier-api-base", default=None, help="optional api_base for the classifier model")
    p.add_argument("--judge", action="store_true", help="also run our own judge on runs no rule explains")
    p.add_argument("--judge-all", action="store_true", help="with --judge: also judge runs that rules label")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    args = p.parse_args()
    if not args.no_llm and not args.classifier_model:
        p.error("--classifier-model is required (no default), or use --no-llm")

    traces = [json.loads(line) for line in (args.run_dir / "traces.jsonl").read_text().splitlines() if line]
    instructions = load_instructions(args.run_dir)
    chat = None
    if args.judge and not args.no_llm:
        token = load_token()
        if not token:
            print("No HF token; use --no-llm or provide the token.", file=sys.stderr)
            return 2
        chat = make_chat(token, args.judge_model)

    examples = tool_example_values(load_tools(args.run_dir))
    classifier = None if args.no_llm else make_classifier(args.classifier_model, args.classifier_api_base)
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
    sys.exit(main())
