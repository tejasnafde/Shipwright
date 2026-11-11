"""Entry point of the harness image (Cloud Run Jobs shipwright-<owner>, docs/setup/cloud-runs.md).

  job_entry.py <script.py> [args...]

Runs ``python scripts/<script.py> args...`` in /app, then uploads every run dir it wrote under
``runs/`` to ``gs://$SHIPWRIGHT_RUNS_BUCKET/$OWNER/<run-dir-name>/`` and the full console log to
``gs://$SHIPWRIGHT_RUNS_BUCKET/$OWNER/jobs/<execution>.log``. It uploads after a failure too,
then exits with the script's exit code.

Debugging (docs/ops/debugging.md): the run id is the Cloud Run execution name; the script gets it as
``SHIPWRIGHT_RUN_ID`` and joins this process's trace through ``TRACEPARENT``. This entry logs job
start and end, the exit code and every upload. If the script dies without a report (killed, out of
memory), it writes ``errors.json`` with the tail of the console log into the run dirs (or into
``runs/job-<run id>/``). ``gs://$SHIPWRIGHT_RUNS_BUCKET/$OWNER/jobs/<run id>.json`` lists the run
dirs of the execution, so ``scripts/why.py <run id>`` finds them.

Auth: the service account token from the metadata server (``shipwright.llm.vertex.metadata_token``).
Nothing here works off GCP, and nothing here needs gcloud.

Env vars:
  OWNER                  required (tejas, ishank, mitushi, main)
  SHIPWRIGHT_RUNS_BUCKET default lexsi-hackathon-runs
  CLOUD_RUN_EXECUTION    set by Cloud Run; names the log file
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import json

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.observability import instrument, log, otel_live, report, repro  # noqa: E402
from shipwright.llm import vertex  # noqa: E402

logger = log.get_logger("job_entry")

DEFAULT_BUCKET = "lexsi-hackathon-runs"
UPLOAD_URL = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={name}"


def run_dirs(runs_directory: Path) -> dict[str, float]:
    """Run dirs under ``runs_directory`` and the newest mtime of any file in each."""
    newest_mtime_by_name = {}
    if not runs_directory.is_dir():
        return newest_mtime_by_name
    for run_directory in runs_directory.iterdir():
        if not run_directory.is_dir():
            continue
        file_mtimes = [path.stat().st_mtime for path in run_directory.rglob("*") if path.is_file()]
        newest_mtime_by_name[run_directory.name] = max(file_mtimes, default=0.0)
    return newest_mtime_by_name


def changed_dirs(mtimes_before: dict[str, float], mtimes_after: dict[str, float]) -> list[str]:
    """Names of run dirs that are new, or that have a file newer than before."""
    changed_names = []
    for name, newest_mtime in mtimes_after.items():
        if name not in mtimes_before or newest_mtime > mtimes_before[name]:
            changed_names.append(name)
    return sorted(changed_names)


TOKENS = vertex.TokenProvider(fetch=vertex.metadata_token)


def upload(http_client: httpx.Client, bucket: str, object_name: str, data: bytes) -> None:
    token = TOKENS.get()
    url = UPLOAD_URL.format(bucket=bucket, name=quote(object_name, safe=""))
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"}
    with instrument.external("gcs_upload", f"gs://{bucket}/{object_name}", bytes=len(data)) as call_record:
        response = http_client.post(url, content=data, headers=headers)
        call_record["status"] = response.status_code
        response.raise_for_status()


def ensure_report(
    run_directory: Path,
    run_id: str,
    script_name: str,
    script_arguments: list[str],
    exit_code: int,
    output_tail: str,
) -> None:
    """errors.json for a dir the script left without one (killed, out of memory, crash before setup)."""
    errors_path = run_directory / "errors.json"
    if errors_path.exists():
        try:
            if json.loads(errors_path.read_text()).get("run_id") == run_id:
                return
        except ValueError:
            pass

    trace_id = otel_live.current_ids()[0]
    error_record = {
        "severity": "ERROR",
        "run_id": run_id,
        "step": None,
        "component": "job_entry",
        "message": f"{script_name} exited {exit_code} without a failure report",
        "error_class": "ExitCode",
        "error": f"exit {exit_code}",
        "stack": output_tail,
        "trace_id": trace_id,
    }
    report_body = report.build(
        run_directory,
        run_id=run_id,
        step="job",
        exit_code=exit_code,
        crashed=True,
        errors=[error_record],
        trace_id=trace_id,
        project=otel_live.project(),
    )
    report.write(run_directory, report_body)

    if not (run_directory / "repro.json").exists():
        repro_record = repro.collect(
            run_id=run_id, script=script_name, argv=script_arguments, owner=os.environ.get("OWNER")
        )
        (run_directory / "repro.json").write_text(json.dumps(repro_record, indent=2))


def find_script(argv: list[str]) -> Path | None:
    """scripts/<name>.py for the first argument (".py" is optional). None if there is no such script."""
    script_name = argv[0]
    if not script_name.endswith(".py"):
        script_name += ".py"
    script_path = ROOT / "scripts" / Path(script_name).name
    if script_path.is_file():
        return script_path
    print(f"no such script: scripts/{Path(script_name).name}", file=sys.stderr)
    return None


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        if argv:
            return 0
        return 2

    owner = os.environ.get("OWNER", "").strip()
    if not owner:
        print("OWNER is not set (tejas, ishank, mitushi or main).", file=sys.stderr)
        return 2
    bucket = os.environ.get("SHIPWRIGHT_RUNS_BUCKET") or DEFAULT_BUCKET
    script_path = find_script(argv)
    if script_path is None:
        return 2
    script_arguments = argv[1:]

    cloud_run_execution = os.environ.get("CLOUD_RUN_EXECUTION") or ""
    requested_run_id = os.environ.get("SHIPWRIGHT_RUN_ID") or cloud_run_execution or None
    if requested_run_id:
        os.environ["SHIPWRIGHT_RUN_ID"] = requested_run_id
    run_id = log.setup(component="job_entry")
    execution = cloud_run_execution or run_id

    started_at = time.time()
    span_attributes = {
        "shipwright.run_id": run_id,
        "shipwright.script": script_path.name,
        "shipwright.argv": " ".join(script_arguments)[:1000],
    }
    with otel_live.span(f"job {script_path.name}", span_attributes, root=True) as job_span:
        exit_code, uploaded_run_dirs, log_path = _run_and_upload(
            owner, bucket, script_path, script_arguments, run_id, execution
        )
        if exit_code:
            otel_live.mark_error(job_span, f"exit {exit_code}")
        trace_id = otel_live.current_ids()[0]
        job_index = {
            "run_id": run_id,
            "owner": owner,
            "execution": execution,
            "script": script_path.name,
            "args": script_arguments,
            "exit_code": exit_code,
            "started": started_at,
            "ended": time.time(),
            "run_dirs": uploaded_run_dirs,
            "log": log_path,
            "trace_id": trace_id,
            "links": report.links(run_id, trace_id, otel_live.project()),
            **repro.image_info(),
            "git_sha": repro.git_info().get("sha"),
        }
        try:
            with httpx.Client(timeout=60) as http_client:
                upload_bytes = json.dumps(job_index, indent=2).encode()
                upload(http_client, bucket, f"{owner}/jobs/{run_id}.json", upload_bytes)
        except Exception:  # noqa: BLE001
            logger.warning("job index upload failed", exc_info=True)
        logger.info(
            "job end",
            exit_code=exit_code,
            run_dirs=uploaded_run_dirs,
            log=log_path,
            links=job_index["links"],
            seconds=round(time.time() - started_at, 1),
        )
    otel_live.shutdown()
    return exit_code


def run_script_process(
    script_path: Path,
    script_arguments: list[str],
    owner: str,
    execution: str,
) -> tuple[int, bytearray]:
    """Run the script, echoing its output. Returns (exit code, everything it printed)."""
    command = [sys.executable, str(script_path), *script_arguments]
    environment = dict(os.environ)
    traceparent = otel_live.traceparent()
    if traceparent:
        environment["TRACEPARENT"] = traceparent

    logger.info("job start", script=script_path.name, args=script_arguments, owner=owner, execution=execution)
    print(f"[job_entry] {' '.join(command[1:])}", flush=True)

    output_buffer = bytearray()
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment
        )
        assert process.stdout is not None
        for output_line in process.stdout:
            sys.stdout.buffer.write(output_line)
            sys.stdout.flush()
            output_buffer += output_line
        exit_code = process.wait()
    except Exception:  # noqa: BLE001
        logger.exception("could not run the script")
        exit_code = 1

    print(f"[job_entry] exit {exit_code}", flush=True)
    log_level = 40 if exit_code else 20
    logger.log(log_level, f"script exit {exit_code}", exit_code=exit_code, script=script_path.name)
    return exit_code, output_buffer


def _run_and_upload(
    owner: str,
    bucket: str,
    script_path: Path,
    script_arguments: list[str],
    run_id: str,
    execution: str,
) -> tuple[int, list[str], str | None]:
    """Run the script, write failure reports, upload changed run dirs and the log.

    Returns (exit code, uploaded run dir URLs, log URL). A clean script whose upload failed
    still exits 1.
    """
    runs_directory = ROOT / "runs"
    mtimes_before = run_dirs(runs_directory)
    exit_code, output_buffer = run_script_process(script_path, script_arguments, owner, execution)

    changed_names = changed_dirs(mtimes_before, run_dirs(runs_directory))
    if exit_code:
        output_tail = bytes(output_buffer[-6000:]).decode("utf-8", errors="replace")
        if changed_names:
            report_directories = [runs_directory / name for name in changed_names]
        else:
            report_directories = [runs_directory / f"job-{run_id}"]
        for report_directory in report_directories:
            try:
                ensure_report(report_directory, run_id, script_path.name, script_arguments, exit_code, output_tail)
            except Exception:  # noqa: BLE001
                logger.warning("failure report not written", run_dir=str(report_directory), exc_info=True)
        changed_names = changed_dirs(mtimes_before, run_dirs(runs_directory))

    upload_failed = False
    uploaded_urls: list[str] = []
    log_url = None
    with httpx.Client(timeout=300) as http_client:
        for name in changed_names:
            run_directory = runs_directory / name
            files = [path for path in sorted(run_directory.rglob("*")) if path.is_file()]
            try:
                for file_path in files:
                    object_name = f"{owner}/{name}/{file_path.relative_to(run_directory).as_posix()}"
                    upload(http_client, bucket, object_name, file_path.read_bytes())
                uploaded_urls.append(f"gs://{bucket}/{owner}/{name}/")
                print(f"[job_entry] uploaded {len(files)} files to gs://{bucket}/{owner}/{name}/", flush=True)
            except Exception:  # noqa: BLE001
                logger.error(f"upload of {name} failed", exc_info=True)
                upload_failed = True
        try:
            upload(http_client, bucket, f"{owner}/jobs/{execution}.log", bytes(output_buffer))
            log_url = f"gs://{bucket}/{owner}/jobs/{execution}.log"
            print(f"[job_entry] log at {log_url}", flush=True)
        except Exception as log_upload_error:  # noqa: BLE001
            print(f"[job_entry] log upload failed: {log_upload_error}", file=sys.stderr)
            upload_failed = True

    if exit_code:
        return exit_code, uploaded_urls, log_url
    if upload_failed:
        return 1, uploaded_urls, log_url
    return 0, uploaded_urls, log_url


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
