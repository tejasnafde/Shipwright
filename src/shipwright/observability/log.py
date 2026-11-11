"""One logging setup for every Shipwright script (docs/ops/debugging.md).

Use it like this in a script:

    from shipwright.observability import log
    logger = log.get_logger(__name__)
    ...
    logger.info("tool patches active", patches=active)       # extra fields as keywords
    ...
    if __name__ == "__main__":
        sys.exit(log.run_script(main, step="run"))

``run_script`` sets up logging and tracing, opens the run span, catches a crash, and at the
end writes ``errors.json``, ``repro.json`` and the run status (summary.json) into every run dir
the script attached with ``log.attach_run_dir(path)``. A script that attaches none gets
``runs/<script>-<stamp>/``.

Output:
- stdout: one JSON object per line (Cloud Run sends it to Cloud Logging, which reads severity,
  message, time, ``logging.googleapis.com/trace`` and ``logging.googleapis.com/spanId``, so a log
  line links to its Cloud Trace span). Human-readable when stdout is a TTY.
  ``SHIPWRIGHT_LOG_FORMAT=json|human`` overrides.
- ``<run_dir>/log.jsonl``: the same JSON lines (lines from before the attach are kept too).

Fields: severity, message, time, logger, run_id, task_id, trial, turn, step, component, model,
git_sha, image_tag, owner, trace_id, plus the call's own fields, and ``exc_info`` (full stack),
``error_class`` and ``error`` for an exception. Context (task_id, step, ...) goes through
contextvars (``log.bind``). Secrets (``hf_...``, ``ya29.``, ``Bearer ...``, ``sk-...``) are
redacted from every field.

``run_id``: ``$SHIPWRIGHT_RUN_ID`` if set (job_entry sets it to the Cloud Run execution name, so
the job and its script share it), else ``local-<stamp>-<hex>``; it is put back into the env so
child processes share it.

``--fail-at <step|agent_call|tool_call>`` (hidden, any script run through ``run_script``) raises a
``ForcedFailure`` at that point, to test the failure path.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from shipwright.observability import otel_live, report, repro

STEPS = ("run", "detect", "classify", "route", "tool_fix", "teacher", "sft", "train", "gate")
FAIL_POINTS = ("agent_call", "tool_call", "search_task", "summary") + STEPS
ROOT = Path(__file__).resolve().parents[3]
SEVERITY = {"DEBUG": "DEBUG", "INFO": "INFO", "WARNING": "WARNING", "ERROR": "ERROR", "CRITICAL": "CRITICAL"}
NOISY = ("httpx", "httpcore", "LiteLLM", "litellm", "urllib3", "opentelemetry", "huggingface_hub", "filelock")

_base: dict[str, Any] = {}
_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("shipwright_log_ctx", default=None)
_lock = threading.Lock()
_state: dict[str, Any] = {"setup": False, "fail_at": None, "run_dirs": [], "errors": [], "repro": {},
                          "buffer": [], "handlers": []}
MAX_ERRORS = 500
MAX_BUFFER = 5000


class ForcedFailure(RuntimeError):
    """Raised by ``fail_point`` when ``--fail-at`` names this point (debug only)."""


# ---------------------------------------------------------------- redaction

_SECRETS = [
    (re.compile(r"hf_[A-Za-z0-9]{8,}"), "hf_[REDACTED]"),
    (re.compile(r"ya29\.[A-Za-z0-9_\-.]+"), "ya29.[REDACTED]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-.=+/]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "sk-[REDACTED]"),
]


def redact(text: str) -> str:
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return text


def redact_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj


# ---------------------------------------------------------------- context

def fields() -> dict[str, Any]:
    return {**_base, **(_ctx.get() or {})}


def set_base(**kw: Any) -> None:
    _base.update({k: v for k, v in kw.items() if v is not None})


@contextlib.contextmanager
def bind(**kw: Any) -> Iterator[None]:
    """Add fields to every log line in this block (and in this thread's calls)."""
    new = {**(_ctx.get() or {}), **{k: (str(v) if k == "task_id" else v) for k, v in kw.items() if v is not None}}
    token = _ctx.set(new)
    try:
        yield
    finally:
        _ctx.reset(token)


def current(key: str) -> Any:
    return fields().get(key)


# ---------------------------------------------------------------- records

def record_dict(record: logging.LogRecord) -> dict[str, Any]:
    d: dict[str, Any] = {
        "severity": SEVERITY.get(record.levelname, record.levelname),
        "message": record.getMessage(),
        "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
        "logger": record.name,
    }
    d.update(getattr(record, "sw_ctx", None) or fields())
    d.update(getattr(record, "fields", None) or {})
    trace_id, span_id = getattr(record, "sw_ids", None) or otel_live.current_ids()
    if trace_id:
        d["trace_id"] = trace_id
        d["span_id"] = span_id
        d["logging.googleapis.com/trace"] = f"projects/{otel_live.project()}/traces/{trace_id}"
        d["logging.googleapis.com/spanId"] = span_id
        d["logging.googleapis.com/trace_sampled"] = True
    if record.exc_info and record.exc_info[0] is not None:
        exc = record.exc_info[1]
        d["error_class"] = type(exc).__name__
        d["error"] = str(exc)[:2000]
        d["exc_info"] = "".join(traceback.format_exception(*record.exc_info))
    if record.stack_info:
        d["call_stack"] = "\n".join(record.stack_info.splitlines()[-30:])
    return redact_obj(d)


class _ContextFilter(logging.Filter):
    """Capture context and span ids in the emitting thread (handlers may format later)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "sw_ctx"):
            record.sw_ctx = fields()
            record.sw_ids = otel_live.current_ids()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(record_dict(record), default=str, ensure_ascii=False)


HUMAN_SKIP = {"severity", "message", "time", "logger", "run_id", "git_sha", "image_tag", "owner", "component",
              "step", "task_id", "trial", "turn", "trace_id", "span_id", "exc_info", "call_stack",
              "logging.googleapis.com/trace", "logging.googleapis.com/spanId", "logging.googleapis.com/trace_sampled"}


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = record_dict(record)
        where = [d.get("step") or ""]
        for key in ("task_id", "trial", "turn"):
            if d.get(key) is not None:
                where.append(f"{key.replace('_id', '')}={d[key]}")
        extra = " ".join(f"{k}={v}" for k, v in d.items() if k not in HUMAN_SKIP and v is not None)
        line = f"{d['time'][11:23]} {d['severity']:7} [{' '.join(w for w in where if w)}] {d['message']}"
        if extra:
            line += f"  {extra}"
        if d.get("exc_info"):
            line += "\n" + d["exc_info"].rstrip()
        return line


class _Collector(logging.Handler):
    """Keeps every ERROR record (bounded) for errors.json, and buffers JSON lines until a run dir exists."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= logging.ERROR and len(_state["errors"]) < MAX_ERRORS:
                d = record_dict(record)
                d["stack"] = d.pop("exc_info", None)
                _state["errors"].append(d)
            if not _state["run_dirs"] and len(_state["buffer"]) < MAX_BUFFER:
                _state["buffer"].append(JsonFormatter().format(record))
        except Exception:  # noqa: BLE001
            pass


class Logger(logging.LoggerAdapter):
    """``logger.info("msg", key=value, ...)``: keywords become JSON fields."""

    RESERVED = ("exc_info", "stack_info", "stacklevel", "extra")

    def process(self, msg: Any, kwargs: dict) -> tuple[Any, dict]:
        extra_fields = {k: kwargs.pop(k) for k in list(kwargs) if k not in self.RESERVED}
        kwargs["extra"] = {**(kwargs.get("extra") or {}), "fields": extra_fields}
        return msg, kwargs


def get_logger(name: str = "shipwright") -> Logger:
    return Logger(logging.getLogger(name), {})


# ---------------------------------------------------------------- setup

def _new_run_id() -> str:
    return f"local-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _human() -> bool:
    fmt = os.environ.get("SHIPWRIGHT_LOG_FORMAT", "").strip().lower()
    if fmt in ("json", "human"):
        return fmt == "human"
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def setup(*, step: str | None = None, component: str | None = None, level: int = logging.INFO,
          tracing_on: bool = True) -> str:
    """Configure the root logger once per process. Returns the run id."""
    with _lock:
        run_id = os.environ.get("SHIPWRIGHT_RUN_ID") or _base.get("run_id") or _new_run_id()
        os.environ["SHIPWRIGHT_RUN_ID"] = run_id
        git = repro.git_info()
        image = os.environ.get("SHIPWRIGHT_IMAGE") or None
        set_base(run_id=run_id, step=step, component=component, owner=os.environ.get("OWNER") or None,
                 git_sha=(git.get("sha") or "")[:12] or None, image_tag=image.rsplit(":", 1)[-1] if image else None)
        if _state["setup"]:
            return run_id
        root = logging.getLogger()
        root.setLevel(level)
        for h in list(root.handlers):  # one setup: drop basicConfig console handlers (keep pytest's)
            if type(h) is logging.StreamHandler:
                root.removeHandler(h)
        out = logging.StreamHandler(sys.stdout)
        out.setFormatter(HumanFormatter() if _human() else JsonFormatter())
        collector = _Collector(level=logging.DEBUG)
        for h in (out, collector):
            h.addFilter(_ContextFilter())
            root.addHandler(h)
        for name in NOISY:
            logging.getLogger(name).setLevel(logging.WARNING)
        _state["handlers"] = [out, collector]
        _state["setup"] = True
    if tracing_on:
        otel_live.init({"shipwright.run_id": run_id, "shipwright.owner": _base.get("owner"),
                      "shipwright.git_sha": _base.get("git_sha"), "service.version": _base.get("git_sha")})
    return run_id


def attach_run_dir(run_dir: Path | str) -> Path:
    """Write log.jsonl (and live spans) into ``run_dir``; it gets errors.json and repro.json at the end."""
    run_dir = Path(run_dir).resolve()
    if not _state["setup"]:  # a library call or a test, not a script run: nothing to write
        return run_dir
    with _lock:
        if run_dir in _state["run_dirs"]:
            return run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "log.jsonl"
        if _state["buffer"]:
            with path.open("a") as f:
                f.write("\n".join(_state["buffer"]) + "\n")
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        handler.addFilter(_ContextFilter())
        logging.getLogger().addHandler(handler)
        _state["handlers"].append(handler)
        _state["run_dirs"].append(run_dir)
    otel_live.attach_dir(run_dir)
    get_logger("shipwright.log").info("run dir attached", run_dir=str(run_dir))
    return run_dir


def run_dirs() -> list[Path]:
    return list(_state["run_dirs"])


def note_repro(**kw: Any) -> None:
    """Add facts to repro.json (models, seeds, task ids, tool patch ids, ...)."""
    _state["repro"].update({k: v for k, v in kw.items() if v is not None})


def expected_exit(code: int, reason: str) -> None:
    """Declare a non-zero exit code that is an outcome, not an error (e.g. the gate rejects: 1)."""
    _state["expected_exit"] = (code, reason)


def collected_errors() -> list[dict]:
    return list(_state["errors"])


# ---------------------------------------------------------------- forced failures

def pop_fail_at(argv: list[str]) -> str | None:
    """Remove ``--fail-at X`` (or ``--fail-at=X``) from ``argv`` in place; return X."""
    for i, a in enumerate(argv):
        if a == "--fail-at" and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if a.startswith("--fail-at="):
            del argv[i]
            return a.split("=", 1)[1]
    return None


def set_fail_at(point: str | None) -> None:
    if point is not None and point not in FAIL_POINTS:
        raise SystemExit(f"--fail-at must be one of {', '.join(FAIL_POINTS)}")
    _state["fail_at"] = point


def fail_point(point: str) -> None:
    if _state["fail_at"] == point:
        raise ForcedFailure(f"forced failure at {point} (--fail-at {point})")


# ---------------------------------------------------------------- run_script

def finalize(*, exit_code: int | None, crashed: bool, default_dir: Path | None = None) -> dict[Path, str]:
    """Write errors.json, repro.json and the status into every attached run dir. Never raises.

    Returns this step's status per run dir (ok, partial, failed)."""
    logger = get_logger("shipwright.log")
    dirs = run_dirs()
    if not dirs and default_dir is not None:
        try:
            attach_run_dir(default_dir)
            dirs = run_dirs()
        except OSError as e:
            logger.warning("no run dir for the report", error=str(e))
    run_id, step = _base.get("run_id"), _base.get("step")
    trace_id = otel_live.current_ids()[0]
    statuses: dict[Path, str] = {}
    expected = _state.get("expected_exit")
    status_code = 0 if expected and exit_code == expected[0] and not crashed else exit_code
    for d in dirs:
        try:
            body = report.build(d, run_id=run_id, step=step, exit_code=status_code, crashed=crashed,
                                errors=collected_errors(), trace_id=trace_id,
                                task_spans=dict(otel_live.TASK_SPANS), project=otel_live.project())
            statuses[d] = body["status"]
            report.write(d, body)
            path = d / "repro.json"
            rec = repro.collect(run_id=run_id or "", script=_state.get("script") or repro.script_name(),
                                argv=_state.get("argv") or sys.argv[1:], owner=_base.get("owner"),
                                extra={**_state["repro"], "trace_id": trace_id, "step": step})
            if path.exists():
                old = json.loads(path.read_text())
                if old.get("run_id") != run_id:  # a later script on the same dir (classify, ...)
                    later = [x for x in old.get("later_steps") or [] if x.get("run_id") != run_id] + [rec]
                    rec = {**old, "later_steps": later}
            path.write_text(json.dumps(rec, indent=2, default=str))
            logger.info("run report written", run_dir=str(d), status=body["status"], counts=body["counts"],
                        links=body["links"])
        except Exception:  # noqa: BLE001
            logger.warning("run report failed", run_dir=str(d), exc_info=True)
    return statuses


def run_script(main: Callable[[], int | None], *, step: str, component: str | None = None) -> int:
    """Run a script's ``main()`` with logging, tracing and the failure report. Returns the exit code."""
    argv = sys.argv[1:]
    original = list(argv)
    fail_at = pop_fail_at(argv)
    sys.argv[1:] = argv
    component = component or Path(sys.argv[0]).stem
    run_id = setup(step=step, component=component)
    _state["script"] = Path(sys.argv[0]).name
    _state["argv"] = original
    set_fail_at(fail_at)
    logger = get_logger("shipwright.log")
    started = time.monotonic()
    code: int | None = 1
    crashed = False
    with otel_live.span(component, {"shipwright.run_id": run_id, "shipwright.step": step,
                                  "shipwright.argv": " ".join(original)[:1000]}, root=True) as sp:
        tp = otel_live.traceparent()
        if tp:
            os.environ["TRACEPARENT"] = tp  # child processes join this trace
        logger.info("script start", argv=original, fail_at=fail_at)
        try:
            fail_point(step)
            code = main()
            code = 0 if code is None else int(code)
        except SystemExit as e:
            if isinstance(e.code, int) or e.code is None:
                code = e.code or 0
            else:
                logger.error(f"script exited: {e.code}")
                code = 1
        except KeyboardInterrupt:
            logger.error("interrupted")
            code, crashed = 130, True
        except BaseException:  # noqa: BLE001
            logger.exception("script crashed")
            code, crashed = 1, True
        if code:
            otel_live.mark_error(sp, f"exit {code}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        statuses = finalize(exit_code=code, crashed=crashed,
                            default_dir=ROOT / "runs" / f"{component}-{stamp}")
        if code == 0 and "failed" in statuses.values():
            # e.g. every task run hit an infra error: make the job show as failed too
            logger.error("run status is failed: exit 1", statuses={str(k): v for k, v in statuses.items()})
            code = 1
            otel_live.mark_error(sp, "status failed")
        logger.info("script end", exit_code=code, crashed=crashed, seconds=round(time.monotonic() - started, 1),
                    status=report.worst(*statuses.values()), run_dirs=[str(d) for d in statuses])
    counts = otel_live.shutdown()
    if counts is not None:
        logger.info("cloud trace export", **counts)
    for h in _state["handlers"]:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass
    return code


def reset_for_tests() -> None:
    root = logging.getLogger()
    for h in _state["handlers"]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass
    _state.update({"setup": False, "fail_at": None, "run_dirs": [], "errors": [], "repro": {}, "buffer": [],
                   "handlers": []})
    _base.clear()
    for key in ("script", "argv", "expected_exit"):
        _state.pop(key, None)
    otel_live.reset_for_tests()
