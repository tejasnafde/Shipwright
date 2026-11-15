"""Which adapter is deployed: an append-only ledger of gate-approved promotions, with one-step rollback.

``promotions/ledger.jsonl`` (committed, small config like ``patches/ledger.jsonl``) holds one row per event:

    {"action": "promote", "round": 2, "candidate": "r2-weight", "adapter": {"repo": ..., "revision": ...},
     "decision_file": "...", "promoted_at": ...}
    {"action": "rollback", "promoted_at": ...}

The deployed adapter is the last promotion that was not rolled back. No adapter means the base model. The next
round's gate compares its candidate against the deployed system (``configs/systems/current.yaml``, written from the
ledger), so each round has to beat the last promoted one. Rows are never deleted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

BASE_MODEL = "CohereLabs/tiny-aya-global"
SERVING_SWITCHES = ("tool_guards", "state_reminders", "condensed_policy")


def read_ledger(ledger_path: Path) -> list[dict]:
    if not ledger_path.exists():
        return []
    return [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]


def deployed_promotions(ledger_rows: list[dict]) -> list[dict]:
    """The promotions still in force, oldest first (a rollback removes the newest one)."""
    in_force: list[dict] = []
    for row in ledger_rows:
        if row["action"] == "promote":
            in_force.append(row)
        elif row["action"] == "rollback" and in_force:
            in_force.pop()
    return in_force


def deployed_system(ledger_rows: list[dict]) -> dict:
    """The system config (``heal/systems.py`` shape) of what is deployed now."""
    in_force = deployed_promotions(ledger_rows)
    if not in_force:
        return {"name": "base", "model": BASE_MODEL, "route": "none"}
    latest = in_force[-1]
    system = {
        "name": f"promoted-r{latest['round']}",
        "model": BASE_MODEL,
        "adapter": latest["adapter"],
        "route": "weight_fix",
    }
    # A candidate is deployed with the serving switches it was gated with (the guards, the per-turn reminders).
    for switch in SERVING_SWITCHES:
        if latest.get(switch):
            system[switch] = True
    return system


def append_row(ledger_path: Path, row: dict) -> dict:
    row = {**row, "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as ledger_file:
        ledger_file.write(json.dumps(row) + "\n")
    return row


def promote(ledger_path: Path, gate_decision: dict, decision_file: str) -> dict:
    """Record a promotion. Refuses anything the gate did not approve, and candidates without an adapter."""
    if gate_decision.get("decision") != "promote":
        raise ValueError(f"the gate decision is {gate_decision.get('decision')!r}, not 'promote'")
    candidate = (gate_decision.get("systems") or {}).get("candidate") or {}
    if not candidate.get("adapter"):
        raise ValueError("the candidate has no adapter: only weight fixes are promoted here (tool patches have their own ledger)")
    return append_row(ledger_path, {
        "action": "promote",
        "round": gate_decision["round"],
        "candidate": candidate["name"],
        "adapter": candidate["adapter"],
        **{switch: bool(candidate.get(switch)) for switch in SERVING_SWITCHES},
        "decision_file": decision_file,
    })


def rollback(ledger_path: Path) -> dict:
    if not deployed_promotions(read_ledger(ledger_path)):
        raise ValueError("nothing to roll back: the base model is already what is deployed")
    return append_row(ledger_path, {"action": "rollback"})
