"""Entry point of the harness image (Cloud Run Jobs shipwright-<owner>, docs/setup/cloud-runs.md).

  job_entry.py [--fetch <owner>/<run-dir>]... <script.py> [args...]

Runs ``python scripts/<script.py> args...`` in /app, then uploads every run dir it wrote under
``runs/`` to ``gs://$SHIPWRIGHT_RUNS_BUCKET/$OWNER/<run-dir-name>/`` and the full console log to
``gs://$SHIPWRIGHT_RUNS_BUCKET/$OWNER/jobs/<execution>.log``. It uploads after a failure too,
then exits with the script's exit code.

``--fetch tejas/baseline-x`` first downloads ``gs://$SHIPWRIGHT_RUNS_BUCKET/tejas/baseline-x/`` to
``runs/baseline-x/``, so a step can read the run dir of an earlier execution (classify a baseline,
teacher runs from its failures). A fetched dir is uploaded again only if the script changes it.

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

from shipwright.observability import instrument, log, otel_live, report, repro, retry  # noqa: E402
from shipwright.llm import vertex  # noqa: E402

logger = log.get_logger("job_entry")

DEFAULT_BUCKET = "lexsi-hackathon-runs"
UPLOAD_URL = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={name}"
LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
OBJECT_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o/{name}?alt=media"


def run_dirs(runs: Path) -> dict[str, float]:
    """Run dirs under ``runs`` and the newest mtime of any file in each."""
    out = {}
    if runs.is_dir():
        for d in runs.iterdir():
            if d.is_dir():
                out[d.name] = max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=0.0)
    return out


def changed_dirs(before: dict[str, float], after: dict[str, float]) -> list[str]:
    return sorted(n for n, t in after.items() if n not in before or t > before[n])


TOKENS = vertex.TokenProvider(fetch=vertex.metadata_token)


def upload(client: httpx.Client, bucket: str, name: str, data: bytes) -> None:
    token = TOKENS.get()
    with instrument.external("gcs_upload", f"gs://{bucket}/{name}", bytes=len(data)) as call:
        def once() -> None:  # a GCS object upload is idempotent: same name, same bytes
            r = client.post(UPLOAD_URL.format(bucket=bucket, name=quote(name, safe="")), content=data,
                            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"})
            call["status"] = r.status_code
            r.raise_for_status()

        retry.call_with_retry(once, what="gcs_upload")


def split_fetch(argv: list[str]) -> tuple[list[str], list[str]]:
    """(['owner/run-dir', ...], the rest of argv) from leading ``--fetch X`` or ``--fetch=X``.

    ``--fetch=X`` exists because gcloud ``--args`` rejects the same value twice (two ``--fetch``).
    """
    fetch = []
    while argv:
        if len(argv) >= 2 and argv[0] == "--fetch":
            value, argv = argv[1], argv[2:]
        elif argv[0].startswith("--fetch="):
            value, argv = argv[0].removeprefix("--fetch="), argv[1:]
        else:
            break
        fetch.append(value.strip("/").removeprefix("gs://"))
    return fetch, argv


def fetch_dir(client: httpx.Client, bucket: str, prefix: str, runs: Path) -> int:
    """Download ``gs://bucket/<prefix>/`` (``owner/run-dir``) into ``runs/<run-dir>/``. Returns the file count."""
    if prefix.startswith(bucket + "/"):
        prefix = prefix[len(bucket) + 1:]
    name = prefix.rsplit("/", 1)[-1]
    if not name or name in (".", ".."):
        raise ValueError(f"bad --fetch value {prefix!r}")
    headers = {"Authorization": f"Bearer {TOKENS.get()}"}

    def get(url: str, params: dict | None = None) -> httpx.Response:
        r = client.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r

    def get_json(url: str, params: dict) -> dict:
        return get(url, params).json()

    count, page = 0, None
    with instrument.external("gcs_download", f"gs://{bucket}/{prefix}/"):
        while True:
            params = {"prefix": prefix + "/", "fields": "items(name),nextPageToken"}
            if page:
                params["pageToken"] = page
            body = retry.call_with_retry(get_json, LIST_URL.format(bucket=bucket), params, what="gcs_list")
            for item in body.get("items", []):
                rel = item["name"][len(prefix) + 1:]
                if not rel or rel.endswith("/") or ".." in Path(rel).parts:
                    continue
                obj = retry.call_with_retry(get, OBJECT_URL.format(bucket=bucket, name=quote(item["name"], safe="")),
                                            what="gcs_download")
                dest = runs / name / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(obj.content)
                count += 1
            page = body.get("nextPageToken")
            if not page:
                break
    if not count:
        raise FileNotFoundError(f"nothing under gs://{bucket}/{prefix}/")
    return count


def ensure_report(run_dir: Path, run_id: str, script: str, args: list[str], code: int, tail: str) -> None:
    """errors.json for a dir the script left without one (killed, out of memory, crash before setup)."""
    path = run_dir / "errors.json"
    if path.exists():
        try:
            if json.loads(path.read_text()).get("run_id") == run_id:
                return
        except ValueError:
            pass
    trace_id = otel_live.current_ids()[0]
    err = {"severity": "ERROR", "run_id": run_id, "step": None, "component": "job_entry",
           "message": f"{script} exited {code} without a failure report",
           "error_class": "ExitCode", "error": f"exit {code}", "stack": tail, "trace_id": trace_id}
    body = report.build(run_dir, run_id=run_id, step="job", exit_code=code, crashed=True, errors=[err],
                        trace_id=trace_id, project=otel_live.project())
    report.write(run_dir, body)
    if not (run_dir / "repro.json").exists():
        rec = repro.collect(run_id=run_id, script=script, argv=args, owner=os.environ.get("OWNER"))
        (run_dir / "repro.json").write_text(json.dumps(rec, indent=2))


def main(argv: list[str]) -> int:
    fetch, argv = split_fetch(list(argv))
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    owner = os.environ.get("OWNER", "").strip()
    if not owner:
        print("OWNER is not set (tejas, ishank, mitushi or main).", file=sys.stderr)
        return 2
    bucket = os.environ.get("SHIPWRIGHT_RUNS_BUCKET") or DEFAULT_BUCKET
    script = argv[0] if argv[0].endswith(".py") else argv[0] + ".py"
    path = ROOT / "scripts" / Path(script).name
    if not path.is_file():
        print(f"no such script: scripts/{Path(script).name}", file=sys.stderr)
        return 2

    execution = os.environ.get("CLOUD_RUN_EXECUTION") or ""
    run_id = os.environ.get("SHIPWRIGHT_RUN_ID") or execution or None
    if run_id:
        os.environ["SHIPWRIGHT_RUN_ID"] = run_id
    run_id = log.setup(component="job_entry")
    execution = execution or run_id
    started = time.time()
    with otel_live.span(f"job {path.name}", {"shipwright.run_id": run_id, "shipwright.script": path.name,
                                            "shipwright.argv": " ".join(argv[1:])[:1000]}, root=True) as sp:
        code, dirs, log_path = _run_and_upload(owner, bucket, path, argv[1:], run_id, execution, fetch)
        if code:
            otel_live.mark_error(sp, f"exit {code}")
        trace_id = otel_live.current_ids()[0]
        index = {"run_id": run_id, "owner": owner, "execution": execution, "script": path.name,
                 "args": argv[1:], "exit_code": code, "started": started, "ended": time.time(),
                 "run_dirs": dirs, "log": log_path, "trace_id": trace_id,
                 "links": report.links(run_id, trace_id, otel_live.project()), **repro.image_info(),
                 "git_sha": repro.git_info().get("sha")}
        try:
            with httpx.Client(timeout=60) as client:
                upload(client, bucket, f"{owner}/jobs/{run_id}.json", json.dumps(index, indent=2).encode())
        except Exception:  # noqa: BLE001
            logger.warning("job index upload failed", exc_info=True)
        logger.info("job end", exit_code=code, run_dirs=dirs, log=log_path, links=index["links"],
                    seconds=round(time.time() - started, 1))
    otel_live.shutdown()
    return code


def _run_and_upload(owner: str, bucket: str, path: Path, args: list[str], run_id: str, execution: str,
                    fetch: list[str] | None = None) -> tuple[int, list[str], str | None]:
    runs = ROOT / "runs"
    fetch_error = None
    for prefix in fetch or []:
        try:
            with httpx.Client(timeout=300) as client:
                n = fetch_dir(client, bucket, prefix, runs)
            print(f"[job_entry] fetched {n} files from gs://{bucket}/{prefix}/", flush=True)
        except Exception as e:  # noqa: BLE001
            logger.error(f"fetch of {prefix} failed", exc_info=True)
            fetch_error = f"fetch of {prefix} failed: {e}"
            break
    before = run_dirs(runs)
    cmd = [sys.executable, str(path), *args]
    env = dict(os.environ)
    tp = otel_live.traceparent()
    if tp:
        env["TRACEPARENT"] = tp
    logger.info("job start", script=path.name, args=args, owner=owner, execution=execution)
    print(f"[job_entry] {' '.join(cmd[1:])}", flush=True)
    log_buf = bytearray()
    if fetch_error:
        print(f"[job_entry] {fetch_error}", flush=True)
        log_buf += (fetch_error + "\n").encode()
        code = 2
    else:
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                log_buf += line
            code = proc.wait()
        except Exception:  # noqa: BLE001
            logger.exception("could not run the script")
            code = 1
    print(f"[job_entry] exit {code}", flush=True)
    logger.log(40 if code else 20, f"script exit {code}", exit_code=code, script=path.name)

    changed = changed_dirs(before, run_dirs(runs))
    if code:
        tail = bytes(log_buf[-6000:]).decode("utf-8", errors="replace")
        targets = [runs / n for n in changed] or [runs / f"job-{run_id}"]
        for d in targets:
            try:
                ensure_report(d, run_id, path.name, args, code, tail)
            except Exception:  # noqa: BLE001
                logger.warning("failure report not written", run_dir=str(d), exc_info=True)
        changed = changed_dirs(before, run_dirs(runs))

    upload_failed = False
    uploaded: list[str] = []
    log_path = None
    with httpx.Client(timeout=300) as client:
        for name in changed:
            files = [f for f in sorted((runs / name).rglob("*")) if f.is_file()]
            try:
                for f in files:
                    upload(client, bucket, f"{owner}/{name}/{f.relative_to(runs / name).as_posix()}", f.read_bytes())
                uploaded.append(f"gs://{bucket}/{owner}/{name}/")
                print(f"[job_entry] uploaded {len(files)} files to gs://{bucket}/{owner}/{name}/", flush=True)
            except Exception:  # noqa: BLE001
                logger.error(f"upload of {name} failed", exc_info=True)
                upload_failed = True
        try:
            upload(client, bucket, f"{owner}/jobs/{execution}.log", bytes(log_buf))
            log_path = f"gs://{bucket}/{owner}/jobs/{execution}.log"
            print(f"[job_entry] log at {log_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[job_entry] log upload failed: {e}", file=sys.stderr)
            upload_failed = True
    return code or (1 if upload_failed else 0), uploaded, log_path

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
