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

from shipwright.heal import tool_fix as tool_fix_module  # noqa: E402
from shipwright.llm.vertex import JUDGE_MODEL  # noqa: E402

DEFAULT_LEDGER = ROOT / "patches" / "ledger.jsonl"


def domain_tools(domain: str) -> list[dict]:
    from tau2.registry import registry

    return [tool.openai_schema for tool in registry.get_env_constructor(domain)().get_tools()]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Propose patches from a run.
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--model",
        default=JUDGE_MODEL,
        help="raw HF router id (e.g. openai/gpt-oss-120b:cerebras) or vertex/<publisher>/<model>"
        " (default: the team's choice, Vertex gpt-oss-120b)",
    )

    # Change the status of patches already in the ledger.
    parser.add_argument("--apply", nargs="+", metavar="PATCH_ID")
    parser.add_argument("--apply-proposed", action="store_true")
    parser.add_argument("--rollback", metavar="PATCH_ID")
    parser.add_argument("--list", action="store_true")
    return parser


def change_patch_statuses(options: argparse.Namespace, ledger: list[dict], tools: list[dict]) -> int:
    """--apply, --apply-proposed and --rollback: set statuses, save the ledger."""
    patch_ids_to_apply = options.apply or []
    if options.apply_proposed:
        patch_ids_to_apply = [row["id"] for row in ledger if row["status"] == "proposed"]

    status_changes = [(patch_id, "applied") for patch_id in patch_ids_to_apply]
    if options.rollback:
        status_changes.append((options.rollback, "rolled_back"))

    for patch_id, new_status in status_changes:
        ledger = tool_fix_module.set_status(ledger, patch_id, new_status, tools)  # raises on a conflict
        print(f"{patch_id}: {new_status}")
    tool_fix_module.save_ledger(options.ledger, ledger)

    _, active_patches = tool_fix_module.apply_patches(tools, ledger)
    print(f"active patches: {active_patches}")
    return 0


def propose_patches(options: argparse.Namespace, ledger: list[dict], tools: list[dict]) -> int:
    """Propose new patches from the failures of a run dir (rules, plus an LLM unless --no-llm)."""
    failures = read_jsonl(options.run_dir / "failures.jsonl")
    trace_by_simulation_id = {
        str(trace.get("simulation_id")): trace for trace in read_jsonl(options.run_dir / "traces.jsonl")
    }

    chat = None
    if not options.no_llm:
        from baseline import load_token
        from classify import make_chat

        try:
            chat = make_chat(load_token(), options.model)
        except RuntimeError as token_error:
            print(f"{token_error} Use --no-llm, or provide the token.", file=sys.stderr)
            return 2

    new_rows, rejection_notes = tool_fix_module.propose(failures, trace_by_simulation_id, tools, ledger, chat, options.round)
    for note in rejection_notes:
        print(f"rejected: {note}")
    for row in new_rows:
        print(
            f"{row['id']} [{row['source']}] {row['tool']}.{row['field']}\n"
            f"  before: {row['before'][:160]}\n"
            f"  after:  {row['after'][:160]}"
        )
    tool_fix_module.save_ledger(options.ledger, ledger + new_rows)
    print(f"\n{len(new_rows)} patches proposed in {options.ledger}. Review, then --apply <id> or --apply-proposed.")
    return 0


def main() -> int:
    parser = build_parser()
    options = parser.parse_args()

    ledger = tool_fix_module.load_ledger(options.ledger)
    tools = domain_tools(options.domain)

    if options.list:
        for summary_row in tool_fix_module.patch_summary(ledger):
            print(json.dumps(summary_row))
        return 0

    if options.apply or options.apply_proposed or options.rollback:
        return change_patch_statuses(options, ledger, tools)

    if not options.run_dir:
        parser.error("give a run dir to propose patches, or --apply / --rollback / --list")
    return propose_patches(options, ledger, tools)


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="tool_fix"))
