import json
from pathlib import Path

import pytest

from shipwright import otel

FIX = Path(__file__).parent / "fixtures"
NAMES = ["passed_run", "wrong_tool_delivered_order", "invented_user_id_and_loop", "premature_transfer"]
META = {"agent_model": "openai/CohereLabs/tiny-aya-global:cohere", "domain": "retail", "task_split": "train"}


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    traces = [load(n) for n in NAMES]
    path = otel.export_run(traces, tmp_path, META)
    return traces, [json.loads(line) for line in path.read_text().splitlines()]


def test_span_tree_and_attributes(exported):
    traces, spans = exported
    assert len({s["trace_id"] for s in spans}) == len(traces)  # one OTel trace per task run
    roots = [s for s in spans if s["parent_span_id"] is None]
    assert len(roots) == len(traces)
    root = next(r for r in roots if r["attributes"]["shipwright.task_id"] == str(traces[1]["task_id"]))
    assert root["name"] == "invoke_agent retail"
    assert root["attributes"]["shipwright.reward"] == traces[1]["reward"]
    assert root["attributes"]["shipwright.termination_reason"] == traces[1]["termination_reason"]

    by_id = {s["span_id"]: s for s in spans}
    chats = [s for s in spans if s["attributes"].get("gen_ai.operation.name") == "chat"]
    tools = [s for s in spans if s["attributes"].get("gen_ai.operation.name") == "execute_tool"]
    assert chats and tools
    for s in chats + tools:
        assert by_id[s["parent_span_id"]]["name"] == "agent_turn"
    n_turns = sum(len(t["agent_turns"]) for t in traces)
    assert len(chats) == n_turns
    assert all("shipwright.raw_output" in s["attributes"] for s in chats)
    errored = [s for s in tools if s["attributes"]["shipwright.tool_error"]]
    assert errored and all(s["status"] == "ERROR" for s in errored)
    assert all(s["attributes"]["shipwright.result_size"] >= 0 for s in tools)
    for s in spans:
        assert s["end_time_ns"] >= s["start_time_ns"]


def test_export_failure_only_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(otel, "emit_trace", lambda *a, **k: 1 / 0)
    assert otel.export_run([load("passed_run")], tmp_path, META) is None
    assert "warning" in capsys.readouterr().out


def test_otlp_only_with_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    p = otel.build_provider(tmp_path / "a.jsonl")
    assert len(p._active_span_processor._span_processors) == 1
    p.shutdown()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    p = otel.build_provider(tmp_path / "b.jsonl")
    assert len(p._active_span_processor._span_processors) == 2
    p.shutdown()  # no spans, so no network call
