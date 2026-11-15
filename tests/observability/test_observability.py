"""Debuggability (docs/ops/debugging.md): log shape, trace fields, redaction, the failure report,
why.py, rerun.py, the Cloud Trace exporter. Offline: no network, GCS is a fake runner."""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from shipwright.observability import log, otel_live, report

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


why = load_script("why")
rerun = load_script("rerun")


def json_lines(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


@pytest.fixture
def json_logs(monkeypatch):
    monkeypatch.setenv("SHIPWRIGHT_LOG_FORMAT", "json")
    monkeypatch.setenv("SHIPWRIGHT_RUN_ID", "test-run-1")
    monkeypatch.setenv("SHIPWRIGHT_CLOUD_TRACE", "0")
    monkeypatch.setenv("OWNER", "tejas")
    monkeypatch.delenv("TRACEPARENT", raising=False)


# ---------------------------------------------------------------- log lines

def test_log_json_shape_and_trace_fields(json_logs, capsys, tmp_path):
    run_id = log.setup(step="run", component="test")
    assert run_id == "test-run-1"
    log.attach_run_dir(tmp_path)
    logger = log.get_logger("t")
    with otel_live.span("task 3") as sp, log.bind(task_id=3, trial=0, turn=2, model="m"):
        logger.info("hello", latency_ms=12.5, tokens_in=10)
        ctx = sp.get_span_context()
    lines = [d for d in json_lines(capsys.readouterr().out) if d["message"] == "hello"]
    assert len(lines) == 1
    d = lines[0]
    for key in ("severity", "message", "time", "run_id", "task_id", "trial", "turn", "step", "component",
                "model", "owner", "git_sha"):
        assert key in d, key
    assert d["severity"] == "INFO" and d["run_id"] == "test-run-1" and d["task_id"] == "3"
    assert d["step"] == "run" and d["owner"] == "tejas" and d["latency_ms"] == 12.5
    trace_id, span_id = format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    assert d["logging.googleapis.com/trace"] == f"projects/lexsi-hackathon/traces/{trace_id}"
    assert d["logging.googleapis.com/spanId"] == span_id
    # the same line in log.jsonl
    file_lines = [x for x in json_lines((tmp_path / "log.jsonl").read_text()) if x["message"] == "hello"]
    assert file_lines and file_lines[0]["trace_id"] == trace_id


def test_log_exception_has_stack_and_class(json_logs, capsys):
    log.setup(step="classify", component="test")
    logger = log.get_logger("t")
    try:
        raise ValueError("bad thing")
    except ValueError:
        logger.exception("it broke")
    d = next(x for x in json_lines(capsys.readouterr().out) if x["message"] == "it broke")
    assert d["severity"] == "ERROR" and d["error_class"] == "ValueError" and d["error"] == "bad thing"
    assert "Traceback" in d["exc_info"] and "raise ValueError" in d["exc_info"]
    assert log.collected_errors()[0]["error_class"] == "ValueError"


def test_context_does_not_leak_out_of_bind(json_logs):
    with log.bind(task_id="7"):
        assert log.current("task_id") == "7"
    assert log.current("task_id") is None


def test_redaction(json_logs, capsys, tmp_path):
    secret_hf = "hf_" + "AbCdEfGhIjKlMnOpQrSt12"
    secret_g = "ya29." + "a0AfH6SMBx-very_long.token"
    assert log.redact(f"token {secret_hf} x") == "token hf_[REDACTED] x"
    assert secret_g not in log.redact(f"t={secret_g}")
    assert log.redact("Authorization: Bearer abc.def-123") == "Authorization: Bearer [REDACTED]"
    assert log.redact("key sk-" + "x" * 30) == "key sk-[REDACTED]"
    assert log.redact("no secret here, hf_ alone") == "no secret here, hf_ alone"
    log.setup(step="run", component="test")
    log.attach_run_dir(tmp_path)
    logger = log.get_logger("t")
    logger.error(f"call failed with {secret_hf}", headers={"Authorization": f"Bearer {secret_g}"})
    try:
        raise RuntimeError(f"401 for {secret_hf}")
    except RuntimeError:
        logger.exception("crash")
    out = capsys.readouterr().out + (tmp_path / "log.jsonl").read_text()
    assert secret_hf not in out and secret_g not in out
    assert "hf_[REDACTED]" in out and "Bearer [REDACTED]" in out
    assert secret_hf not in json.dumps(log.collected_errors())


def test_human_format_when_asked(monkeypatch, capsys):
    monkeypatch.setenv("SHIPWRIGHT_LOG_FORMAT", "human")
    monkeypatch.setenv("SHIPWRIGHT_CLOUD_TRACE", "0")
    log.setup(step="run", component="test")
    with log.bind(task_id="4", turn=1):
        log.get_logger("t").warning("slow call", latency_ms=900)
    out = capsys.readouterr().out
    assert "WARNING" in out and "[run task=4 turn=1] slow call" in out and "latency_ms=900" in out
    assert not out.lstrip().startswith("{")


def test_pop_fail_at():
    argv = ["--dry-run", "--fail-at", "agent_call", "--num-tasks", "2"]
    assert log.pop_fail_at(argv) == "agent_call"
    assert argv == ["--dry-run", "--num-tasks", "2"]
    argv = ["--fail-at=summary"]
    assert log.pop_fail_at(argv) == "summary" and argv == []
    assert log.pop_fail_at(["--x"]) is None
    log.set_fail_at("tool_call")
    log.fail_point("agent_call")  # not this one
    with pytest.raises(log.ForcedFailure):
        log.fail_point("tool_call")
    with pytest.raises(SystemExit):
        log.set_fail_at("nowhere")


# ---------------------------------------------------------------- failure report

def crafted_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    failed = json.loads((FIXTURES / "premature_transfer.json").read_text())
    passed = json.loads((FIXTURES / "passed_run.json").read_text())
    infra = {"task_id": "9", "trial": 0, "reward": None, "passed": False,
             "termination_reason": "infrastructure_error", "error": "RateLimitError: 429", "error_type": "RateLimitError",
             "messages": [], "agent_turns": []}
    (run / "traces.jsonl").write_text("\n".join(json.dumps(t) for t in (failed, passed, infra)) + "\n")
    (run / "summary.json").write_text(json.dumps({"pass_rate": 1 / 3}))
    return run


def test_error_report_from_crafted_failing_run(tmp_path):
    run = crafted_run(tmp_path)
    err = {"severity": "ERROR", "message": "llm call failed", "task_id": "9", "turn": 3, "step": "run",
           "component": "llm.agent", "error_class": "RateLimitError", "error": "429", "stack": "Traceback ...",
           "last_input": "[user] where is my order", "last_output": "let me check", "run_id": "r1"}
    body = report.build(run, run_id="r1", step="run", exit_code=0, crashed=False, errors=[err],
                        trace_id="ab" * 16, task_spans={("9", 0): "cd" * 8})
    assert body["status"] == "partial"
    assert body["counts"] == {"tasks": 3, "passed": 1, "failed": 2, "infra_errors": 1, "errors_logged": 1,
                              "exit_code": 0}
    assert body["first_error"]["error_class"] == "RateLimitError"
    assert "tid=" + "ab" * 16 in body["links"]["trace"]
    assert "jsonPayload.run_id%3D%22r1%22" in body["links"]["logs"] and "project=lexsi-hackathon" in body["links"]["logs"]
    tasks = {t["task_id"]: t for t in body["failed_tasks"]}
    assert set(tasks) == {"2", "9"}
    t2 = tasks["2"]
    assert t2["turn"] is not None and t2["first_error"] and t2["last_raw_output"]
    assert t2["last_model_input"].startswith("[")
    assert "task_id%3D%222%22" in t2["links"]["logs"]
    t9 = tasks["9"]
    assert t9["infra_error"] and t9["span_id"] == "cd" * 8 and t9["turn"] == 3
    assert t9["logged_error"]["error_class"] == "RateLimitError"
    assert t9["last_model_input"] == "[user] where is my order" and t9["last_raw_output"] == "let me check"

    report.write(run, body)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "partial" and summary["pass_rate"] == 1 / 3
    # a second step on the same dir keeps the first step's errors and the worst status
    later = report.build(run, run_id="r2", step="classify", exit_code=0, crashed=False, errors=[],
                         trace_id=None)
    report.write(run, later)
    merged = json.loads((run / "errors.json").read_text())
    assert merged["status_by_step"] == {"run": "partial", "classify": "partial"}
    assert merged["errors"][0]["run_id"] == "r1"


def test_status_rules():
    ok = [{"passed": True}]
    assert report.status_of(0, ok, [], False)[0] == "ok"
    assert report.status_of(0, [{"passed": False, "termination_reason": "user_stop"}], [], False)[0] == "ok"
    assert report.status_of(0, [{"termination_reason": "infrastructure_error"}], [], False)[0] == "failed"
    assert report.status_of(1, ok, [], False)[0] == "failed"
    assert report.status_of(0, ok, [{"x": 1}], False)[0] == "partial"
    assert report.status_of(0, [], [], True)[0] == "failed"


def run_fake_script(monkeypatch, tmp_path, main, argv=("--flag", "x")) -> int:
    monkeypatch.setattr(sys, "argv", ["scripts/fake.py", *argv])
    monkeypatch.setattr(log, "ROOT", tmp_path)
    return log.run_script(main, step="run")


def test_run_script_crash_writes_report(json_logs, monkeypatch, tmp_path, capsys):
    out = tmp_path / "runs" / "r"

    def main():
        log.attach_run_dir(out)
        log.note_repro(models={"agent_model": "a"}, seed=3)
        with log.bind(task_id="5", turn=2):
            raise KeyError("missing order")

    code = run_fake_script(monkeypatch, tmp_path, main, ("--fail-at", "agent_call", "--n", "1"))
    assert code == 1
    errors = json.loads((out / "errors.json").read_text())
    assert errors["status"] == "failed" and errors["run_id"] == "test-run-1"
    first = errors["first_error"]
    assert first["error_class"] == "KeyError" and "Traceback" in first["stack"]
    assert errors["trace_id"] and errors["links"]["trace"].endswith(errors["trace_id"])
    rep = json.loads((out / "repro.json").read_text())
    assert rep["script"] == "fake.py" and rep["argv"] == ["--fail-at", "agent_call", "--n", "1"]
    assert rep["seed"] == 3 and rep["models"] == {"agent_model": "a"} and rep["tau2_commit"]
    assert json.loads((out / "summary.json").read_text())["status"] == "failed"
    msgs = [d["message"] for d in json_lines((out / "log.jsonl").read_text())]
    assert "script start" in msgs and "script crashed" in msgs and "script end" in msgs
    spans = [json.loads(x) for x in (out / otel_live.LIVE_SPANS_FILE).read_text().splitlines()]
    assert any(s["name"] == "fake" and s["status"] == "ERROR" for s in spans)


def test_run_script_without_run_dir_gets_default(json_logs, monkeypatch, tmp_path):
    assert run_fake_script(monkeypatch, tmp_path, lambda: 0, ()) == 0
    dirs = list((tmp_path / "runs").iterdir())
    assert len(dirs) == 1 and dirs[0].name.startswith("fake-")
    assert json.loads((dirs[0] / "errors.json").read_text())["status"] == "ok"


def test_expected_exit_is_not_an_error(json_logs, monkeypatch, tmp_path):
    def main():
        log.attach_run_dir(tmp_path / "runs" / "g")
        log.expected_exit(1, "gate rejected")
        return 1

    assert run_fake_script(monkeypatch, tmp_path, main, ()) == 1
    assert json.loads((tmp_path / "runs" / "g" / "errors.json").read_text())["status"] == "ok"


# ---------------------------------------------------------------- why.py

def finished_failed_run(json_logs, monkeypatch, tmp_path) -> Path:
    run = crafted_run(tmp_path)

    def main():
        log.attach_run_dir(run)
        with log.bind(task_id="9", trial=0, turn=3, model="tiny"):
            try:
                raise TimeoutError("router timed out")
            except TimeoutError:
                log.get_logger("x").error("llm call failed: agent tiny", last_input="[user] hi",
                                          last_output="```json {}```", exc_info=True)
        return 0

    run_fake_script(monkeypatch, tmp_path, main, ("--dry-run", "--num-tasks", "3", "--output-dir", "runs/run"))
    return run


def test_why_on_a_local_run(json_logs, monkeypatch, tmp_path, capsys):
    run = finished_failed_run(json_logs, monkeypatch, tmp_path)
    capsys.readouterr()
    assert why.main([str(run)]) == 0
    out = capsys.readouterr().out
    assert "status PARTIAL" in out
    assert "FIRST ERROR" in out and "TimeoutError: router timed out" in out
    assert "task 9 trial 0 > turn 3" in out and "input:  [user] hi" in out
    assert "FAILED TASKS (2)" in out and "task 2 trial 0" in out
    assert "TIMELINE" in out and "script start" in out
    assert "logs:" in out and "trace:" in out and "console.cloud.google.com" in out
    assert f"scripts/rerun.py {run} --task 9 --where local-dry-run" in out

    assert why.main([str(run), "--task", "2"]) == 0
    out = capsys.readouterr().out
    assert "TASK 2" in out and "agent turn" in out and "LINKS" in out and "task_id%3D%222%22" in out


def test_why_by_run_id_on_gcs(tmp_path, capsys, monkeypatch):
    """A run id resolves through the job index; files come from `gcloud storage cat` (fake)."""
    base = "gs://lexsi-hackathon-runs/tejas/baseline-x"
    errors = {"run_id": "shipwright-tejas-ab12", "status": "failed",
              "counts": {"tasks": 1, "passed": 0, "failed": 1, "infra_errors": 1, "errors_logged": 1, "exit_code": 1},
              "first_error": {"message": "llm call failed", "error_class": "ForcedFailure", "error": "forced",
                              "task_id": "0", "turn": 2, "step": "run", "component": "llm.agent"},
              "errors": [], "failed_tasks": [], "links": {"logs": "L", "trace": "T"}}
    files = {f"{base}/errors.json": json.dumps(errors), f"{base}/repro.json": json.dumps({"script": "baseline.py"})}
    index = {"run_id": "shipwright-tejas-ab12", "run_dirs": [base + "/"], "log": "gs://x/jobs/a.log"}
    calls = []

    def runner(cmd):
        calls.append(cmd)
        assert "--configuration=personal" in cmd[1] or any(c.startswith("--configuration=") for c in cmd)
        if cmd[-2:] == ["cat", "gs://lexsi-hackathon-runs/*/jobs/shipwright-tejas-ab12.json"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(index), "")
        if "ls" in cmd:
            listing = "\n".join(f"  {len(v)}  2026-11-09T00:00:00Z  {k}" for k, v in files.items()) + "\nTOTAL: 2 objects"
            return subprocess.CompletedProcess(cmd, 0, listing, "")
        if "cat" in cmd and cmd[-1] in files:
            return subprocess.CompletedProcess(cmd, 0, files[cmd[-1]], "")
        return subprocess.CompletedProcess(cmd, 1, "", "not found")

    monkeypatch.setattr(why, "resolve", lambda ref: __import__("shipwright.observability.runinfo", fromlist=["x"]).resolve(
        ref, runner, runs=tmp_path))
    assert why.main(["shipwright-tejas-ab12"]) == 0
    out = capsys.readouterr().out
    assert "RUN shipwright-tejas-ab12  status FAILED" in out and "ForcedFailure: forced" in out
    assert "joblog: gs://x/jobs/a.log" in out
    assert not any("traces.jsonl" in " ".join(c) for c in calls)  # small files only


# ---------------------------------------------------------------- rerun.py

REPRO = {"run_id": "shipwright-tejas-ab12", "script": "baseline.py", "owner": "tejas",
         "argv": ["--num-tasks", "20", "--user-model", "vertex/openai/gpt-oss-120b-maas",
                  "--output-dir", "/app/runs/b1", "--task-ids", "1", "2"],
         "image": "asia-south1-docker.pkg.dev/lexsi-hackathon/shipwright/harness:abc", "git_sha": "abc"}


def test_rerun_local_dry_run_command():
    cmd, notes = rerun.build(REPRO, where="local-dry-run", task="7", stamp="S")
    assert cmd[:2] == [".venv/bin/python", "scripts/baseline.py"]
    args = cmd[2:]
    assert "--dry-run" in args and "--num-tasks" not in args
    assert args[args.index("--task-ids") + 1:args.index("--task-ids") + 2] == ["7"] and args.count("--task-ids") == 1
    assert "2" not in args and args[args.index("--output-dir") + 1] == "runs/rerun-shipwright-tejas-ab12-S"
    assert rerun.is_offline("baseline.py", args)


def test_rerun_cloud_command_and_offline_variants():
    cmd, notes = rerun.build(REPRO, where="cloud", stamp="S")
    assert "run" in cmd and "execute" in cmd and "shipwright-tejas" in cmd and "--region=asia-south1" in cmd
    assert any(c.startswith("--configuration=") for c in cmd) and "--project=lexsi-hackathon" in cmd
    args = next(c for c in cmd if c.startswith("--args="))
    assert args.startswith("--args=^|^baseline.py|--num-tasks|20|") and "--dry-run" not in args
    assert any("harness:abc" in n for n in notes)
    cmd, _ = rerun.build(REPRO, where="cloud", offline_cloud=True, owner="ishank")
    assert "shipwright-ishank" in cmd and "|--dry-run" in next(c for c in cmd if c.startswith("--args="))
    classify = {"script": "classify.py", "argv": ["/app/runs/b1", "--judge"], "owner": "tejas"}
    cmd, notes = rerun.build(classify, where="local-dry-run")
    assert cmd[2:] == ["runs/b1", "--no-llm"] and notes
    with pytest.raises(ValueError):
        rerun.build(classify, where="local-dry-run", task="1")
    with pytest.raises(ValueError):
        rerun.build({**REPRO, "owner": None}, where="cloud")
    assert rerun.is_offline("train_job.py", ["--timeout", "2h"]) and not rerun.is_offline("train_job.py", ["--launch"])


def test_rerun_never_spends_by_accident(tmp_path, monkeypatch, capsys):
    run = tmp_path / "r"
    run.mkdir()
    (run / "repro.json").write_text(json.dumps(REPRO))
    called = []
    monkeypatch.setattr(rerun.subprocess, "call", lambda *a, **k: called.append(a) or 0)
    assert rerun.main([str(run), "--where", "cloud"]) == 0  # print only
    assert "WARNING" in capsys.readouterr().out and not called
    assert rerun.main([str(run), "--where", "cloud", "--execute"]) == 2  # spends: needs --allow-spend
    assert not called
    assert rerun.main([str(run), "--execute"]) == 0  # local dry run: offline, runs
    assert len(called) == 1 and "--dry-run" in called[0][0]


# ---------------------------------------------------------------- Cloud Trace exporter

def test_cloud_trace_export_format_and_failure_only_warns(caplog):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    sent = []
    exporter = otel_live.CloudTraceExporter("lexsi-hackathon", token=lambda: "tok",
                                          post=lambda url, body, headers: sent.append((url, body, headers)))
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("task 1", attributes={"a": "x" * 400, "n": 3, "ok": True}):
        with tracer.start_as_current_span("chat m"):
            pass
    assert len(sent) == 2
    url, body, headers = sent[1]
    assert url == "https://cloudtrace.googleapis.com/v2/projects/lexsi-hackathon/traces:batchWrite"
    assert headers["Authorization"] == "Bearer tok"
    root = body["spans"][0]
    assert root["name"].startswith("projects/lexsi-hackathon/traces/") and "/spans/" in root["name"]
    assert root["displayName"]["value"] == "task 1" and root["startTime"].endswith("Z")
    attrs = root["attributes"]["attributeMap"]
    assert attrs["n"] == {"intValue": "3"} and attrs["ok"] == {"boolValue": True}
    assert len(attrs["a"]["stringValue"]["value"]) == 256 and attrs["a"]["stringValue"]["truncatedByteCount"] == 144
    assert sent[0][1]["spans"][0]["parentSpanId"] == root["spanId"]

    def boom(url, body, headers):
        raise RuntimeError("HTTP 403: permission denied")

    bad = otel_live.CloudTraceExporter("lexsi-hackathon", token=lambda: "tok", post=boom)
    provider2 = TracerProvider()
    provider2.add_span_processor(SimpleSpanProcessor(bad))
    with caplog.at_level(logging.WARNING, logger="shipwright.tracing"):
        with provider2.get_tracer("t").start_as_current_span("x"):
            pass
        with provider2.get_tracer("t").start_as_current_span("y"):
            pass
    assert bad.failures == 2
    assert sum("Cloud Trace export failed" in r.message for r in caplog.records) == 1  # warns once


def test_traceparent_joins_the_parent_trace(json_logs, monkeypatch):
    parent = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
    monkeypatch.setenv("TRACEPARENT", parent)
    log.setup(step="run", component="t")
    with otel_live.span("child", root=True):
        trace_id, _ = otel_live.current_ids()
    assert trace_id == "1" * 32
