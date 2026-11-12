"""Promote a gate-approved adapter, roll back the last promotion, or show what is deployed.

    scripts/promote.py show
    scripts/promote.py promote runs/gate-round-2-.../gate_decision.json
    scripts/promote.py rollback

Only a gate decision of "promote" with an adapter is accepted. The ledger is ``promotions/ledger.jsonl``;
after every change ``configs/systems/current.yaml`` is rewritten from it. That file is the ``--current`` of the next
round's gate (scripts/run_round.py uses it when it exists), so a new adapter has to beat the deployed one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.heal import promotion  # noqa: E402
from shipwright.observability import log  # noqa: E402

LEDGER_PATH = ROOT / "promotions" / "ledger.jsonl"
CURRENT_SYSTEM_PATH = ROOT / "configs" / "systems" / "current.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("show")
    subcommands.add_parser("rollback")
    promote_parser = subcommands.add_parser("promote")
    promote_parser.add_argument("gate_decision", type=Path, help="a gate_decision.json")
    return parser.parse_args(argv)


def write_current_system() -> dict:
    system = promotion.deployed_system(promotion.read_ledger(LEDGER_PATH))
    CURRENT_SYSTEM_PATH.write_text(
        "# Written by scripts/promote.py from promotions/ledger.jsonl: the deployed system. Do not edit.\n"
        + yaml.safe_dump(system, sort_keys=False)
    )
    return system


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if options.command == "promote":
        gate_decision = json.loads(options.gate_decision.read_text())
        promotion.promote(LEDGER_PATH, gate_decision, str(options.gate_decision))
    elif options.command == "rollback":
        promotion.rollback(LEDGER_PATH)
    if options.command == "show":
        deployed = promotion.deployed_system(promotion.read_ledger(LEDGER_PATH))
    else:
        deployed = write_current_system()
    print(f"Deployed: {deployed['name']}" + (f" ({deployed['adapter']['repo']}@{deployed['adapter']['revision']})"
                                             if deployed.get("adapter") else " (the base model)"))
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="promote", meta=True))
