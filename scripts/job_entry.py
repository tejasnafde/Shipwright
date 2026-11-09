"""Entry point of the harness image (Cloud Run Jobs shipwright-<owner>, docs/cloud-runs.md).

  job_entry.py <script.py> [args...]

Runs ``python scripts/<script.py> args...`` in /app, then uploads every run dir it wrote under
``runs/`` to ``gs://$SHIPWRIGHT_RUNS_BUCKET/$OWNER/<run-dir-name>/`` and the full console log to
``gs://$SHIPWRIGHT_RUNS_BUCKET/$OWNER/jobs/<execution>.log``. It uploads after a failure too,
then exits with the script's exit code.

Auth: the service account token from the metadata server (``shipwright.vertex.metadata_token``).
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

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright import vertex  # noqa: E402

DEFAULT_BUCKET = "lexsi-hackathon-runs"
UPLOAD_URL = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={name}"


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
    r = client.post(UPLOAD_URL.format(bucket=bucket, name=quote(name, safe="")), content=data,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"})
    r.raise_for_status()


def main(argv: list[str]) -> int:
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

    runs = ROOT / "runs"
    before = run_dirs(runs)
    cmd = [sys.executable, str(path), *argv[1:]]
    print(f"[job_entry] {' '.join(cmd[1:])}", flush=True)
    log = bytearray()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.buffer.write(line)
        sys.stdout.flush()
        log += line
    code = proc.wait()
    print(f"[job_entry] exit {code}", flush=True)

    execution = os.environ.get("CLOUD_RUN_EXECUTION") or time.strftime("local-%Y%m%d-%H%M%S")
    upload_failed = False
    with httpx.Client(timeout=300) as client:
        for name in changed_dirs(before, run_dirs(runs)):
            files = [f for f in sorted((runs / name).rglob("*")) if f.is_file()]
            for f in files:
                upload(client, bucket, f"{owner}/{name}/{f.relative_to(runs / name).as_posix()}", f.read_bytes())
            print(f"[job_entry] uploaded {len(files)} files to gs://{bucket}/{owner}/{name}/", flush=True)
        try:
            upload(client, bucket, f"{owner}/jobs/{execution}.log", bytes(log))
            print(f"[job_entry] log at gs://{bucket}/{owner}/jobs/{execution}.log", flush=True)
        except httpx.HTTPError as e:
            print(f"[job_entry] log upload failed: {e}", file=sys.stderr)
            upload_failed = True
    return code or (1 if upload_failed else 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
