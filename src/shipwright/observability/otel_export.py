"""OpenTelemetry spans for a run, built from our trace records.

traces.jsonl stays the source of truth. After a run, ``export_run`` turns each trace
record into one OTel trace (GenAI semantic conventions where cheap):

  invoke_agent <domain>           one per task run: task id, reward, termination reason
    user_message                  one per user simulator message
    agent_turn                    one per agent turn
      chat <model>                the LLM call: raw output, parsed tool call, parse
                                  status, prompt token estimate, context trimming
      execute_tool <name>         one per tool call: arguments, result, result size, error

Span times come from the tau2 message timestamps, so the spans show real latency even
though they are built after the run. Building them after the run means no code runs
inside tau2 for tracing, and ``export_run`` catches every error: tracing never breaks a run.

Export: always a JSONL file (``otel_spans.jsonl`` in the run dir). Also OTLP/HTTP when
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (Phoenix, Langfuse, any OTel backend), see
docs/ops/observability.md. ``scripts/otel_to_traces.py`` converts spans back to trace records.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

log = logging.getLogger(__name__)

SPANS_FILE = "otel_spans.jsonl"
SERVICE_NAME = "shipwright-agent"
NS = 1_000_000_000

# Our own attributes. GenAI semconv keys are written inline (gen_ai.*), and a few
# OpenInference keys (openinference.span.kind, input.value, output.value) so Phoenix
# shows the span kinds and text.
P = "shipwright."


class JsonlFileSpanExporter(SpanExporter):
    """Writes each finished span as one JSON line."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Iterable[ReadableSpan]) -> SpanExportResult:
        lines = [json.dumps(span_to_dict(s), default=str) for s in spans]
        if lines:
            with self._lock, self._path.open("a") as f:
                f.write("\n".join(lines) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    parent = span.parent
    return {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "attributes": dict(span.attributes or {}),
        "status": span.status.status_code.name,
        "status_description": span.status.description,
        "resource": dict(span.resource.attributes) if span.resource else {},
    }


def build_provider(spans_path: Path, resource: dict[str, Any] | None = None) -> TracerProvider:
    """File exporter always; OTLP/HTTP too when OTEL_EXPORTER_OTLP_ENDPOINT is set."""
    attrs = {"service.name": SERVICE_NAME, **(resource or {})}
    provider = TracerProvider(resource=Resource.create(attrs))
    provider.add_span_processor(SimpleSpanProcessor(JsonlFileSpanExporter(spans_path)))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            # Endpoint and headers come from the standard OTEL_EXPORTER_OTLP_* env vars.
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except Exception as e:  # noqa: BLE001
            log.warning("OTLP export off: %s", e)
    return provider


def _ns(ts: Any) -> int | None:
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(str(ts)).timestamp() * NS)
    except (TypeError, ValueError):
        return None


def _set(span: trace_api.Span, attrs: dict[str, Any]) -> None:
    """Set attributes, skipping None and turning dicts and lists of dicts into JSON."""
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, dict) or (isinstance(value, list) and any(not isinstance(v, (str, int, float, bool)) for v in value)):
            value = json.dumps(value, default=str)
        span.set_attribute(key, value)


class _Clock:
    """Message timestamps in ns, with a fallback that keeps spans ordered if one is missing."""

    def __init__(self, messages: list[dict]):
        self.times = [_ns(m.get("timestamp")) for m in messages]
        self.last = next((t for t in self.times if t), None) or time.time_ns()

    def at(self, index: int) -> int:
        t = self.times[index] if 0 <= index < len(self.times) else None
        if t is None or t < self.last:
            t = self.last
        self.last = t
        return t


def emit_trace(tracer: trace_api.Tracer, rec: dict, meta: dict | None = None) -> None:
    """Emit the spans of one trace record (one task run) as one OTel trace."""
    meta = meta or {}
    messages = rec.get("messages") or []
    clock = _Clock(messages)
    turns = {t["message_index"]: t for t in rec.get("agent_turns") or []}
    tool_msgs = {m.get("id"): (i, m) for i, m in enumerate(messages) if m.get("role") == "tool"}
    model = meta.get("agent_model") or ""
    domain = meta.get("domain") or "agent"

    start = clock.at(0) if messages else time.time_ns()
    root = tracer.start_span(f"invoke_agent {domain}", start_time=start, context=trace_api.set_span_in_context(trace_api.INVALID_SPAN))
    root_ctx = trace_api.set_span_in_context(root)
    _set(root, {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": domain,
        "gen_ai.provider.name": meta.get("provider"),
        "gen_ai.request.model": model or None,
        "gen_ai.conversation.id": rec.get("simulation_id"),
        "openinference.span.kind": "AGENT",
        P + "simulation_id": rec.get("simulation_id"),
        P + "task_id": None if rec.get("task_id") is None else str(rec.get("task_id")),
        P + "trial": rec.get("trial"),
        P + "seed": rec.get("seed"),
        P + "reward": rec.get("reward"),
        P + "passed": rec.get("passed"),
        P + "termination_reason": rec.get("termination_reason"),
        P + "error": rec.get("error"),
        P + "error_type": rec.get("error_type"),
        P + "duration": rec.get("duration"),
        P + "agent_cost": rec.get("agent_cost"),
        P + "user_cost": rec.get("user_cost"),
        P + "reward_info": rec.get("reward_info") or None,
        P + "num_messages": rec.get("num_messages"),
        P + "user_model": meta.get("user_model"),
        P + "task_split": meta.get("task_split"),
        P + "active_tool_patches": meta.get("active_tool_patches") or None,
    })
    if rec.get("error"):
        root.set_status(Status(StatusCode.ERROR, str(rec["error"])[:500]))

    last_end = start
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "user":
            t = clock.at(i)
            span = tracer.start_span("user_message", context=root_ctx, start_time=t)
            _set(span, {
                "openinference.span.kind": "CHAIN",
                "output.value": msg.get("content"),
                P + "message_index": i,
                P + "role": "user",
                P + "content": msg.get("content"),
                P + "turn_idx": msg.get("turn_idx"),
            })
            span.end(end_time=t)
            last_end = max(last_end, t)
        elif role == "assistant":
            last_end = max(last_end, _emit_turn(tracer, root_ctx, clock, i, msg, turns.get(i), tool_msgs, model))
    root.end(end_time=max(last_end, start))


def _emit_turn(tracer, root_ctx, clock: _Clock, i: int, msg: dict, turn: dict | None,
               tool_msgs: dict, model: str) -> int:
    end = clock.at(i)
    gen = msg.get("generation_time_seconds")
    start = end - int(gen * NS) if isinstance(gen, (int, float)) and gen > 0 else end
    turn = turn or {}
    span = tracer.start_span("agent_turn", context=root_ctx, start_time=start)
    ctx = trace_api.set_span_in_context(span)
    errors = turn.get("errors") or []
    _set(span, {
        "openinference.span.kind": "CHAIN",
        P + "message_index": i,
        P + "turn_idx": msg.get("turn_idx"),
        P + "errors": errors or None,
    })

    usage = turn.get("usage") or msg.get("usage") or {}
    raw_output = turn.get("raw_output", msg.get("content"))
    parsed = turn.get("parsed_tool_calls") or []
    chat = tracer.start_span(f"chat {model}".strip(), context=ctx, start_time=start)
    output_messages = [{
        "role": "assistant",
        "parts": ([{"type": "text", "content": msg.get("content")}] if msg.get("content") else [])
        + [{"type": "tool_call", "id": tc.get("id"), "name": tc.get("name"), "arguments": tc.get("arguments")}
           for tc in (msg.get("tool_calls") or [])],
        "finish_reason": turn.get("finish_reason"),
    }]
    _set(chat, {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model or None,
        "gen_ai.response.finish_reasons": [turn["finish_reason"]] if turn.get("finish_reason") else None,
        "gen_ai.usage.input_tokens": usage.get("prompt_tokens"),
        "gen_ai.usage.output_tokens": usage.get("completion_tokens"),
        "gen_ai.usage.cache_read.input_tokens": usage.get("cached_prompt_tokens"),
        "gen_ai.output.messages": output_messages,
        "openinference.span.kind": "LLM",
        "llm.model_name": model or None,
        "output.value": raw_output,
        P + "message_index": i,
        P + "turn_idx": msg.get("turn_idx"),
        P + "content": msg.get("content"),
        P + "tool_calls": msg.get("tool_calls") or None,
        P + "raw_output": raw_output,
        P + "reasoning": turn.get("reasoning"),
        P + "tool_call_parse": turn.get("tool_call_parse"),
        P + "parse_failed": turn.get("tool_call_parse") == "parse_failed",
        P + "raw_tool_calls": turn.get("raw_tool_calls") or None,
        P + "parsed_tool_calls": parsed or None,
        P + "text_tool_call": turn.get("text_tool_call"),
        P + "prompt_tokens_est": turn.get("prompt_tokens_est"),
        P + "prompt_sha": turn.get("prompt_sha"),
        P + "context_trimmed": turn.get("context_trimmed"),
        P + "context_dropped_messages": turn.get("context_dropped_messages"),
        P + "context_truncated_chars": turn.get("context_truncated_chars"),
        P + "usage": usage or None,
    })
    if turn.get("tool_call_parse") == "parse_failed":
        chat.set_status(Status(StatusCode.ERROR, "tool_call_parse_failed"))
    chat.end(end_time=end)

    last = end
    for tc in parsed:
        idx, res = tool_msgs.get(tc.get("id"), (None, None))
        t_end = clock.at(idx) if idx is not None else end
        tool = tracer.start_span(f"execute_tool {tc.get('name')}", context=ctx, start_time=end)
        content = res.get("content") if res else None
        _set(tool, {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tc.get("name"),
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": tc.get("id"),
            "gen_ai.tool.call.arguments": tc.get("arguments") if tc.get("arguments") is not None else {},
            "gen_ai.tool.call.result": content,
            "openinference.span.kind": "TOOL",
            "tool.name": tc.get("name"),
            "input.value": json.dumps(tc.get("arguments"), default=str),
            "output.value": content,
            P + "message_index": idx,
            P + "turn_idx": res.get("turn_idx") if res else None,
            P + "requestor": res.get("requestor") if res else None,
            P + "result_size": len(str(content)) if content is not None else 0,
            P + "tool_error": bool(res.get("error")) if res else None,
            P + "executed": res is not None,
            P + "malformed_arguments": tc.get("malformed_arguments"),
            P + "unknown_tool": tc.get("unknown_tool"),
        })
        if res and res.get("error"):
            tool.set_status(Status(StatusCode.ERROR, str(content)[:500]))
            tool.set_attribute("error.type", "tool_error")
        tool.end(end_time=max(t_end, end))
        last = max(last, t_end)
    if errors:
        span.set_status(Status(StatusCode.ERROR, ", ".join(errors)[:500]))
    span.end(end_time=last)
    return last


def export_run(traces: list[dict], out_dir: Path, meta: dict | None = None) -> Path | None:
    """Write OTel spans for every trace record. Never raises: a failure only logs a warning."""
    path = Path(out_dir) / SPANS_FILE
    try:
        meta = meta or {}
        resource = {
            "deployment.environment": meta.get("task_split") or "",
            "shipwright.run_dir": Path(out_dir).name,
        }
        provider = build_provider(path, resource)
        tracer = provider.get_tracer("shipwright")
        try:
            for rec in traces:
                emit_trace(tracer, rec, meta)
        finally:
            provider.shutdown()
        return path
    except Exception as e:  # noqa: BLE001
        log.warning("OTel span export failed, the run is not affected: %s", e)
        print(f"warning: OTel span export failed ({e}); traces.jsonl is unaffected")
        return None
