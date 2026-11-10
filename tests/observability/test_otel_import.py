import json
from pathlib import Path

import pytest

from shipwright.heal import classify as c
from shipwright.observability import otel_export
from shipwright.observability.otel_import import read_spans, spans_to_traces

FIX = Path(__file__).parent.parent / "fixtures"
NAMES = ["passed_run", "wrong_tool_delivered_order", "invented_user_id_and_loop", "premature_transfer"]
EXAMPLES = {"#W0000000", "sara_doe_496", "1008292230", "6086499569", "gift_card_0000000"}
META = {"agent_model": "openai/CohereLabs/tiny-aya-global:cohere", "domain": "retail", "task_split": "train"}
TURN_KEYS = ["message_index", "turn_idx", "raw_output", "tool_call_parse", "prompt_tokens_est",
             "context_trimmed", "raw_tool_calls", "parsed_tool_calls", "errors",
             "text_tool_call", "finish_reason"]


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    traces = [load(n) for n in NAMES]
    path = otel_export.export_run(traces, tmp_path, META)
    return traces, read_spans(path)


def test_round_trip_keeps_turns_and_labels(exported):
    traces, spans = exported
    back = spans_to_traces(spans)
    assert len(back) == len(traces)
    for orig, conv in zip(traces, back):
        assert conv["task_id"] == str(orig["task_id"])
        assert conv["reward"] == orig["reward"] and conv["passed"] == orig["passed"]
        assert conv["termination_reason"] == orig["termination_reason"]
        assert len(conv["agent_turns"]) == len(orig["agent_turns"])
        for a, b in zip(orig["agent_turns"], conv["agent_turns"]):
            for k in TURN_KEYS:
                if k in a:  # the fixtures omit some keys
                    assert (a[k] or None) == (b.get(k) or None), k
            # tool result text is cut by hand in some fixtures; the messages keep it whole
            assert [(r["id"], r["name"], r["error"]) for r in a["tool_results"]] == \
                [(r["id"], r["name"], r["error"]) for r in b["tool_results"]]
        assert [(m["role"], m.get("content")) for m in orig["messages"]] == \
            [(m["role"], m.get("content")) for m in conv["messages"]]
        assert c.rule_labels(orig, EXAMPLES) == c.rule_labels(conv, EXAMPLES)


def test_generic_genai_spans_convert():
    """Spans from another tracer: only semconv keys, no shipwright attributes."""
    base = {"trace_id": "t1", "status": "UNSET"}
    spans = [
        {**base, "name": "invoke_agent", "span_id": "r", "parent_span_id": None, "start_time_ns": 1,
         "end_time_ns": 9, "attributes": {"gen_ai.operation.name": "invoke_agent"}},
        {**base, "name": "chat m", "span_id": "c1", "parent_span_id": "r", "start_time_ns": 2, "end_time_ns": 3,
         "attributes": {"gen_ai.operation.name": "chat", "gen_ai.completion": "looking it up"}},
        {**base, "name": "execute_tool get_order", "span_id": "t", "parent_span_id": "r", "start_time_ns": 4,
         "end_time_ns": 5, "status": "ERROR",
         "attributes": {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_order",
                        "gen_ai.tool.call.id": "x1", "gen_ai.tool.call.arguments": '{"order_id": "#W1"}',
                        "gen_ai.tool.call.result": "Error: not found"}},
        {**base, "name": "chat m", "span_id": "c2", "parent_span_id": "r", "start_time_ns": 6, "end_time_ns": 7,
         "attributes": {"gen_ai.operation.name": "chat", "output.value": "Sorry."}},
    ]
    [rec] = spans_to_traces(spans)
    assert [m["role"] for m in rec["messages"]] == ["assistant", "tool", "assistant"]
    first = rec["agent_turns"][0]
    assert first["parsed_tool_calls"][0]["name"] == "get_order"
    assert first["parsed_tool_calls"][0]["arguments"] == {"order_id": "#W1"}
    assert first["tool_results"][0]["error"] is True
    assert rec["agent_turns"][1]["raw_output"] == "Sorry."
