"""Correction rows: only the teacher's final reply is trained, never the student's wrong earlier turns."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_lora  # noqa: E402


def test_only_the_last_assistant_turn_of_a_correction_row_is_trained():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "wrong call 1"}, {"role": "user", "content": "result"},
                {"role": "assistant", "content": "wrong call 2"}, {"role": "user", "content": "result"},
                {"role": "assistant", "content": "the teacher's reply"}]
    all_assistant_trained = [True, True, False, True, False, True, False]  # True = masked
    masked = train_lora.mask_all_but_last_assistant(messages, all_assistant_trained)
    assert masked == [True, True, True, True, True, True, False]


def test_a_row_without_assistant_turns_is_left_alone():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    assert train_lora.mask_all_but_last_assistant(messages, [True, True]) == [True, True]


def agenttune_provenance_without_torch(monkeypatch):
    """agenttune/__init__ and agenttune/utils/__init__ import torch (not installed here or in CI): load only
    utils/provenance.py by file path, under package stubs that carry the installed version."""
    import importlib.util
    import types
    from importlib.metadata import version

    spec = importlib.util.find_spec("agenttune")
    package_directory = Path(spec.origin).parent
    agenttune_stub = types.ModuleType("agenttune")
    agenttune_stub.__path__ = [str(package_directory)]
    agenttune_stub.__version__ = version("agenttune")
    utils_stub = types.ModuleType("agenttune.utils")
    utils_stub.__path__ = [str(package_directory / "utils")]
    monkeypatch.setitem(sys.modules, "agenttune", agenttune_stub)
    monkeypatch.setitem(sys.modules, "agenttune.utils", utils_stub)
    provenance_spec = importlib.util.spec_from_file_location("agenttune.utils.provenance",
                                                             package_directory / "utils" / "provenance.py")
    provenance = importlib.util.module_from_spec(provenance_spec)
    monkeypatch.setitem(sys.modules, "agenttune.utils.provenance", provenance)
    provenance_spec.loader.exec_module(provenance)


def test_the_adapter_carries_agenttunes_lineage_record(tmp_path, monkeypatch):
    import json

    agenttune_provenance_without_torch(monkeypatch)

    options = train_lora.parse_args(["--dataset", "mitushigoyal5/shipwright-sft", "--dataset-revision", "round-16",
                                     "--revision", "round-16"])
    train_lora.write_lineage_record(tmp_path, options, 0.48)
    record = json.loads((tmp_path / "lexsi_provenance.json").read_text())
    assert record["schema"] == "lexsi.provenance/1" and record["library"] == "agenttune"
    assert record["base_model"] == options.student
    assert record["inputs"][0]["ref"] == "mitushigoyal5/shipwright-sft" and record["inputs"][0]["config"] == "round-16"
    assert record["params"]["revision"] == "round-16" and record["params"]["train_loss"] == 0.48


def test_a_lineage_failure_never_stops_the_push(tmp_path, monkeypatch, capsys):
    agenttune_provenance_without_torch(monkeypatch)
    options = train_lora.parse_args([])
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory")
    train_lora.write_lineage_record(blocked, options, None)
    assert "not written" in capsys.readouterr().out
