"""The general-replay examples: what they say, and that they never overlap the samples the gate scores."""

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_bfcl_replay as replay  # noqa: E402

TOOL = {"type": "function", "function": {"name": "light_travel_time", "description": "Time for light.",
        "parameters": {"type": "object", "required": ["distance"], "properties": {"distance": {"type": "number"}}}}}


def fake_sample(category, calls, sample_id="simple_100"):
    return argparse.Namespace(
        metadata={"bfcl_category": category, "bfcl_id": sample_id, "messages": None}, input="How long does light take?",
        tools=[TOOL], expected_tool_calls=calls)


def test_the_call_uses_the_first_accepted_value_and_leaves_out_optional_arguments():
    calls = [[{"name": "f", "arguments": {"a": [4, 5], "b": ["x"], "c": ["", "y"]}}]]
    assert replay.accepted_call(calls) == {"name": "f", "arguments": {"a": 4, "b": "x"}}
    assert replay.accepted_call([]) is None


def test_a_simple_sample_gets_a_json_call_and_an_irrelevant_one_gets_words():
    call_row = replay.replay_row(fake_sample("simple", [[{"name": "light_travel_time", "arguments": {"distance": [4]}}]]), 0)
    assert call_row["messages"][-1]["content"].startswith("```json") and '"distance": 4' in call_row["messages"][-1]["content"]
    assert [message["role"] for message in call_row["messages"]] == ["system", "user", "assistant"]
    words_row = replay.replay_row(fake_sample("irrelevance", [], "irrelevance_100"), 1)
    assert "```json" not in words_row["messages"][-1]["content"] and "light_travel_time" in words_row["messages"][-1]["content"]


def test_the_first_samples_of_each_category_are_skipped_and_the_count_is_per_category():
    samples = {"simple": [fake_sample("simple", [[{"name": "f", "arguments": {}}]], f"simple_{number}") for number in range(10)]}
    rows = replay.build_rows(samples, skip=4, per_category=3)
    assert [row["task_id"] for row in rows] == ["simple_4", "simple_5", "simple_6"]


def test_the_replay_set_never_overlaps_the_samples_the_gate_scores():
    """Against the real data: the gate takes the first limit_per_category of each category."""
    pytest.importorskip("auditkit")
    import yaml

    from shipwright.heal import bfcl_eval

    config = yaml.safe_load((ROOT / "configs" / "gate.yaml").read_text())["bfcl"]
    try:
        root = bfcl_eval.download(config["revision"], list(config["categories"]))
    except Exception as download_error:  # noqa: BLE001 - offline machine
        pytest.skip(f"BFCL data not available: {download_error}")
    import auditkit

    for category in config["categories"]:
        questions_path, answers_path = bfcl_eval.bfcl_files(root, category)
        arguments = (str(questions_path), str(answers_path) if answers_path else None)
        gate_ids = {sample.metadata["bfcl_id"] for sample in auditkit.load_bfcl(
            *arguments, revision=config["revision"], limit=config["limit_per_category"])}
        everything = auditkit.load_bfcl(*arguments, revision=config["revision"], limit=None)
        for sample in everything:
            sample.metadata["bfcl_category"] = category
        replay_ids = {row["task_id"] for row in replay.build_rows({category: everything}, config["limit_per_category"], 100)}
        assert len(gate_ids) == config["limit_per_category"] and replay_ids and not (gate_ids & replay_ids), category
