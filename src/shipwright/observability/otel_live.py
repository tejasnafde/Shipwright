"""Live OpenTelemetry spans, created while a script runs (docs/ops/debugging.md).

Span tree of one script run (one OTel trace; ``job_entry.py`` passes its trace to the
script with the ``TRACEPARENT`` env var, so a Cloud Run job is one trace end to end):

  job <script>                      scripts/job_entry.py (Cloud Run only)
    <script>                        shipwright.observability.log.run_script: the run
      task <id>                     one tau2 task run (shipwright.observability.instrument)
        agent_turn                  one agent step
          chat <model>              one LLM call (agent, user, judge, teacher, classifier)
          execute_tool <name>       one tau2 tool call
      gcs_upload / hf_hub / ...     other external calls

Exporters: always ``live_spans.jsonl`` in every run dir the script attaches
(``shipwright.log.attach_run_dir``); spans that end before a dir is attached are kept in
memory and written then. On GCP (``shipwright.llm.vertex.on_gcp()``) also Cloud Trace, through the
v2 REST API with the metadata-server token (the runtime service account needs
``roles/cloudtrace.agent``). ``SHIPWRIGHT_CLOUD_TRACE=0`` turns Cloud Trace off, ``=1`` forces it on.

Every function here only warns on an error: tracing never breaks a run.
``otel_spans.jsonl`` (``shipwright.observability.otel_export``) is a different file: spans rebuilt from
traces.jsonl after the run, with the full message text.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from opentelemetry import context as otel_context
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

warn = logging.getLogger("shipwright.tracing")

LIVE_SPANS_FILE = "live_spans.jsonl"
SERVICE_NAME = "shipwright"
CLOUD_TRACE_URL = "https://cloudtrace.googleapis.com/v2/projects/{project}/traces:batchWrite"
# Cloud Trace v2 limits: 32 attributes per span, 128-byte keys, 256-byte string values.
MAX_ATTRS = 32
MAX_VALUE_BYTES = 256
MAX_NAME_BYTES = 128

_lock = threading.Lock()
_provider: TracerProvider | None = None
_tracer: trace_api.Tracer | None = None
_root_ctx: otel_context.Context | None = None
_file_exporter: "FanOutFileExporter | None" = None
_cloud_exporter: "CloudTraceExporter | None" = None
# (task_id, trial) -> span id of the task span, for the failure report.
TASK_SPANS: dict[tuple[str, int | None], str] = {}


def project() -> str:
    return os.environ.get("SHIPWRIGHT_GCP_PROJECT") or "lexsi-hackathon"


def _span_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    return {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "attributes": dict(span.attributes or {}),
        "status": span.status.status_code.name,
        "status_description": span.status.description,
    }


class FanOutFileExporter(SpanExporter):
    """Appends finished spans to ``live_spans.jsonl`` in every attached dir.

    Spans that end before any dir is attached are buffered (bounded) and written on attach.
    """

    def __init__(self, max_buffer: int = 5000):
        self._dirs: list[Path] = []
        self._buffer: list[str] = []
        self._max = max_buffer
        self._lock = threading.Lock()

    def add_dir(self, run_dir: Path) -> None:
        with self._lock:
            run_dir = Path(run_dir)
            if run_dir in self._dirs:
                return
            self._dirs.append(run_dir)
            if self._buffer:
                self._write(run_dir, self._buffer)

    @staticmethod
    def _write(run_dir: Path, lines: list[str]) -> None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / LIVE_SPANS_FILE).open("a") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:
            warn.warning("live span file write failed: %s", e)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        lines = [json.dumps(_span_dict(s), default=str) for s in spans]
        with self._lock:
            if len(self._buffer) < self._max:
                self._buffer.extend(lines[: self._max - len(self._buffer)])
            for d in self._dirs:
                self._write(d, lines)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _truncated(value: str, limit: int) -> dict:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return {"value": value, "truncatedByteCount": 0}
    cut = raw[:limit].decode("utf-8", errors="ignore")
    return {"value": cut, "truncatedByteCount": len(raw) - len(cut.encode("utf-8"))}


def _rfc3339(ns: int | None) -> str:
    ns = int(ns or 0)
    base = datetime.fromtimestamp(ns // 1_000_000_000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{ns % 1_000_000_000:09d}Z"


def to_cloud_trace(span: ReadableSpan, project_id: str) -> dict:
    """One span in the Cloud Trace v2 ``batchWrite`` format."""
    ctx = span.get_span_context()
    trace_id, span_id = format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    attrs: dict[str, dict] = {}
    items = list((span.attributes or {}).items())
    for key, value in items[:MAX_ATTRS]:
        key = str(key)[:MAX_NAME_BYTES]
        if isinstance(value, bool):
            attrs[key] = {"boolValue": value}
        elif isinstance(value, int):
            attrs[key] = {"intValue": str(value)}
        else:
            text = value if isinstance(value, str) else json.dumps(list(value) if isinstance(value, tuple) else value,
                                                                     default=str)
            attrs[key] = {"stringValue": _truncated(text, MAX_VALUE_BYTES)}
    out = {
        "name": f"projects/{project_id}/traces/{trace_id}/spans/{span_id}",
        "spanId": span_id,
        "displayName": _truncated(span.name, MAX_NAME_BYTES),
        "startTime": _rfc3339(span.start_time),
        "endTime": _rfc3339(span.end_time or span.start_time),
        "attributes": {"attributeMap": attrs, "droppedAttributesCount": max(0, len(items) - MAX_ATTRS)},
    }
    if span.parent is not None:
        out["parentSpanId"] = format(span.parent.span_id, "016x")
    if span.status.status_code == StatusCode.ERROR:
        out["status"] = {"code": 2, "message": (span.status.description or "error")[:500]}
    return out


class CloudTraceExporter(SpanExporter):
    """Cloud Trace v2 REST exporter. Auth: a bearer token from ``token()`` (metadata server)."""

    def __init__(self, project_id: str, token=None, post=None):
        self.project_id = project_id
        self._token = token
        self._post = post
        self.failures = 0
        self.exported = 0
        self._warned = False

    def _default_post(self, url: str, body: dict, headers: dict) -> int:
        import httpx

        r = httpx.post(url, json=body, headers=headers, timeout=20)
        if r.status_code >= 300:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.status_code

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            if self._token is None:
                from shipwright.llm import vertex

                provider = vertex.TokenProvider(fetch=vertex.metadata_token)
                self._token = provider.get
            body = {"spans": [to_cloud_trace(s, self.project_id) for s in spans]}
            headers = {"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"}
            (self._post or self._default_post)(CLOUD_TRACE_URL.format(project=self.project_id), body, headers)
            self.exported += len(spans)
            return SpanExportResult.SUCCESS
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            if not self._warned:  # once: a missing grant would otherwise warn on every batch
                self._warned = True
                warn.warning("Cloud Trace export failed (the file export still works): %s: %s",
                             type(e).__name__, str(e)[:300])
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


def cloud_trace_enabled() -> bool:
    flag = os.environ.get("SHIPWRIGHT_CLOUD_TRACE", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    try:
        from shipwright.llm import vertex

        return vertex.on_gcp() and bool(os.environ.get("CLOUD_RUN_JOB") or os.environ.get("K_SERVICE"))
    except Exception:  # noqa: BLE001
        return False


def init(resource: dict[str, Any] | None = None) -> None:
    """Start the tracer once per process. Safe to call again (no-op)."""
    global _provider, _tracer, _file_exporter, _cloud_exporter, _root_ctx
    with _lock:
        if _provider is not None:
            return
        try:
            attrs = {"service.name": SERVICE_NAME, **{k: v for k, v in (resource or {}).items() if v is not None}}
            provider = TracerProvider(resource=Resource.create(attrs))
            _file_exporter = FanOutFileExporter()
            provider.add_span_processor(SimpleSpanProcessor(_file_exporter))
            if cloud_trace_enabled():
                _cloud_exporter = CloudTraceExporter(project())
                provider.add_span_processor(BatchSpanProcessor(_cloud_exporter, schedule_delay_millis=2000))
            _provider = provider
            _tracer = provider.get_tracer("shipwright")
            parent = os.environ.get("TRACEPARENT")
            if parent:
                _root_ctx = TraceContextTextMapPropagator().extract({"traceparent": parent})
        except Exception as e:  # noqa: BLE001
            warn.warning("tracing off: %s", e)
            _provider = None
            _tracer = None


def attach_dir(run_dir: Path) -> None:
    if _file_exporter is not None:
        _file_exporter.add_dir(Path(run_dir))


def set_root(span: trace_api.Span | None) -> None:
    """Spans with no current parent (e.g. in tau2 worker threads) become children of ``span``."""
    global _root_ctx
    if span is not None:
        _root_ctx = trace_api.set_span_in_context(span)


def root_context() -> otel_context.Context | None:
    return _root_ctx


def traceparent() -> str | None:
    """W3C traceparent of the current span (or the root), for a child process."""
    carrier: dict[str, str] = {}
    try:
        ctx = None if trace_api.get_current_span().get_span_context().is_valid else _root_ctx
        TraceContextTextMapPropagator().inject(carrier, context=ctx)
    except Exception:  # noqa: BLE001
        return None
    return carrier.get("traceparent")


def current_ids() -> tuple[str | None, str | None]:
    """(trace id, span id) of the current span, as hex; (None, None) when there is none."""
    try:
        ctx = trace_api.get_current_span().get_span_context()
        if not ctx.is_valid and _root_ctx is not None:
            ctx = trace_api.get_current_span(_root_ctx).get_span_context()
        if not ctx.is_valid:
            return None, None
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    except Exception:  # noqa: BLE001
        return None, None


def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            out[k] = v
        else:
            out[k] = json.dumps(v, default=str)
    return out


@contextlib.contextmanager
def span(name: str, attrs: dict[str, Any] | None = None, *, root: bool = False) -> Iterator[trace_api.Span]:
    """A span that is current inside the block. Exceptions mark it ERROR and propagate.

    ``root=True``: parent is the TRACEPARENT context (or none), and it becomes the fallback
    parent for spans started in other threads. With tracing off this yields a no-op span.
    """
    if _tracer is None:
        yield trace_api.INVALID_SPAN
        return
    try:
        parent = None
        if root or not trace_api.get_current_span().get_span_context().is_valid:
            parent = _root_ctx
        sp = _tracer.start_span(name, context=parent, attributes=_clean(attrs or {}))
    except Exception as e:  # noqa: BLE001
        warn.warning("span %s not started: %s", name, e)
        yield trace_api.INVALID_SPAN
        return
    if root:
        set_root(sp)
    token = otel_context.attach(trace_api.set_span_in_context(sp))
    try:
        yield sp
    except BaseException as e:
        try:
            sp.record_exception(e)
            sp.set_status(Status(StatusCode.ERROR, f"{type(e).__name__}: {e}"[:500]))
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        try:
            otel_context.detach(token)
        except Exception:  # noqa: BLE001
            pass
        try:
            sp.end()
        except Exception:  # noqa: BLE001
            pass


def set_attrs(sp: trace_api.Span, attrs: dict[str, Any]) -> None:
    try:
        sp.set_attributes(_clean(attrs))
    except Exception:  # noqa: BLE001
        pass


def mark_error(sp: trace_api.Span, message: str) -> None:
    try:
        sp.set_status(Status(StatusCode.ERROR, message[:500]))
    except Exception:  # noqa: BLE001
        pass


def shutdown() -> dict[str, int] | None:
    """Flush and stop. Returns Cloud Trace counts (exported, failed batches) when it was on."""
    global _provider, _tracer
    with _lock:
        provider, _provider, _tracer = _provider, None, None
    if provider is None:
        return None
    try:
        provider.force_flush(10_000)
        provider.shutdown()
    except Exception as e:  # noqa: BLE001
        warn.warning("tracing shutdown: %s", e)
    if _cloud_exporter is None:
        return None
    return {"exported": _cloud_exporter.exported, "failed_batches": _cloud_exporter.failures}


def reset_for_tests() -> None:
    global _provider, _tracer, _file_exporter, _cloud_exporter, _root_ctx
    _provider = _tracer = _file_exporter = _cloud_exporter = _root_ctx = None
    TASK_SPANS.clear()
