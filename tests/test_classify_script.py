"""scripts/classify.py: a copied example is a tool fault only when the student's tool list showed it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import classify  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "get_order_details", "description": "Get an order.",
    "parameters": {"type": "object", "required": ["order_id"], "properties": {
        "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."}}}}}]


def write_run(directory: Path, run_config: dict) -> None:
    directory.mkdir()
    (directory / "run_config.json").write_text(json.dumps(run_config))
    (directory / "prompts.json").write_text(json.dumps({"sha": {"system": "policy", "tools": TOOLS}}))


def test_version_1_runs_showed_every_example(tmp_path):
    write_run(tmp_path / "old", {})
    assert classify.visible_example_values(tmp_path / "old", {"#W0000000"}) == {"#W0000000"}


def test_version_2_runs_hid_real_looking_examples(tmp_path):
    write_run(tmp_path / "new", {"tool_list_version": 2})
    assert classify.visible_example_values(tmp_path / "new", {"#W0000000"}) == set()
