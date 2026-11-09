"""Tool-fix branch: propose patches to tool descriptions, apply them, roll them back.

Usage:
  .venv/bin/python scripts/tool_fix.py runs/baseline-<ts>            propose (rules + LLM), status proposed
  .venv/bin/python scripts/tool_fix.py runs/baseline-<ts> --no-llm   propose with rules only
  .venv/bin/python scripts/tool_fix.py --apply tp-xxxx [tp-yyyy ...]
  .venv/bin/python scripts/tool_fix.py --apply-proposed              apply every proposed patch
  .venv/bin/python scripts/tool_fix.py --rollback tp-xxxx
  .venv/bin/python scripts/tool_fix.py --list

Propose reads failures.jsonl (from scripts/classify.py) and traces.jsonl in the run dir.
The ledger is patches/ledger.jsonl (committed). baseline.py --tool-patches uses its applied rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shipwright import tool_fix as tf  # noqa: E402
from shipwright.vertex import JUDGE_MODEL  # noqa: E402

DEFAULT_LEDGER = ROOT / "patches" / "ledger.jsonl"


def domain_tools(domain: str) -> list[dict]:
    from tau2.registry import registry

    return [t.openai_schema for t in registry.get_env_constructor(domain)().get_tools()]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, nargs="?")
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--domain", default="retail")
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--model", default=JUDGE_MODEL,
                   help="raw HF router id (e.g. openai/gpt-oss-120b:cerebras) or vertex/<publisher>/<model>"
                        " (default: the team's choice, Vertex gpt-oss-120b)")
    p.add_argument("--apply", nargs="+", metavar="PATCH_ID")
    p.add_argument("--apply-proposed", action="store_true")
    p.add_argument("--rollback", metavar="PATCH_ID")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    ledger = tf.load_ledger(args.ledger)
    tools = domain_tools(args.domain)

    if args.list:
        for r in tf.patch_summary(ledger):
            print(json.dumps(r))
        return 0

    if args.apply or args.apply_proposed or args.rollback:
        ids = args.apply or []
        if args.apply_proposed:
            ids = [r["id"] for r in ledger if r["status"] == "proposed"]
        changes = [(i, "applied") for i in ids]
        if args.rollback:
            changes.append((args.rollback, "rolled_back"))
        for pid, status in changes:
            ledger = tf.set_status(ledger, pid, status, tools)  # raises on a conflict
            print(f"{pid}: {status}")
        tf.save_ledger(args.ledger, ledger)
        _, active = tf.apply_patches(tools, ledger)
        print(f"active patches: {active}")
        return 0

    if not args.run_dir:
        p.error("give a run dir to propose patches, or --apply / --rollback / --list")
    failures = read_jsonl(args.run_dir / "failures.jsonl")
    traces = {str(t.get("simulation_id")): t for t in read_jsonl(args.run_dir / "traces.jsonl")}
    chat = None
    if not args.no_llm:
        from baseline import load_token
        from classify import make_chat

        try:
            chat = make_chat(load_token(), args.model)
        except RuntimeError as e:
            print(f"{e} Use --no-llm, or provide the token.", file=sys.stderr)
            return 2

    rows, notes = tf.propose(failures, traces, tools, ledger, chat, args.round)
    for n in notes:
        print(f"rejected: {n}")
    for r in rows:
        print(f"{r['id']} [{r['source']}] {r['tool']}.{r['field']}\n  before: {r['before'][:160]}\n  after:  {r['after'][:160]}")
    tf.save_ledger(args.ledger, ledger + rows)
    print(f"\n{len(rows)} patches proposed in {args.ledger}. Review, then --apply <id> or --apply-proposed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
