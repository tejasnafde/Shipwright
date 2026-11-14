"""Where everything outside this machine lives, and how to reach it: one place for every name, price and token rule.

Before 2026-11-14 these were copied into many files (the GPU prices three times, the runs repo four times, the gcloud
config rule three times, two default user models that disagreed). A value changes here or nowhere.

    GCP        project, region, the harness image, the runs bucket (Cloud Run Jobs upload run dirs there)
    Hugging Face   the router, the student model, the runs / SFT / adapter repos, the HF Jobs GPU prices
    tokens     the HF token (``hf_token``), a GCS / Vertex bearer token (metadata server in the cloud, gcloud on a laptop)
    GCS        upload a file or a dir, fetch a dir (HTTP JSON API, no gcloud needed inside the image)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- GCP

PROJECT = "lexsi-hackathon"
REGION = "asia-south1"
IMAGE = "asia-south1-docker.pkg.dev/lexsi-hackathon/shipwright/harness"
DEFAULT_BUCKET = "lexsi-hackathon-runs"
OWNERS = ("tejas", "ishank", "mitushi", "main")
HOMEBREW_GCLOUD = "/opt/homebrew/share/google-cloud-sdk/bin/gcloud"
TEAM_GCLOUD_CONFIG = "lexsi"        # docs/setup/gcp-setup.md, step 2 (Ishank, Mitushi)
FALLBACK_GCLOUD_CONFIG = "personal"  # Tejas's config name

# ---------------------------------------------------------------- Hugging Face

HF_ROUTER = "https://router.huggingface.co/v1"
STUDENT_MODEL = "CohereLabs/tiny-aya-global"
RUNS_REPO = "mitushigoyal5/shipwright-runs"    # gate results, candidate-on-train runs, code bundles, the promotion ledger
SFT_REPO = "mitushigoyal5/shipwright-sft"      # one branch per round: round-N
LORA_REPO = "mitushigoyal5/shipwright-lora"    # one branch per round: round-N

# HF Jobs, USD per hour (AGENTS.md, 2026-11-08). A job is billed for its running seconds, not its time in the queue.
GPU_PRICES = {"t4-small": 0.40, "l4x1": 0.80, "a10g-small": 1.00, "a10g-large": 1.50, "l40sx1": 1.80,
              "a100-large": 2.50, "rtx-pro-6000": 2.75}


def timeout_hours(timeout_text: str) -> float:
    """"2h", "90m", "3600s" or "3600" (seconds) -> hours."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", str(timeout_text).strip())
    if not match:
        raise ValueError(f"timeout {timeout_text!r}: write it as 45m or 1h")
    amount = float(match.group(1))
    hours_per_unit = {"s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}
    return amount * hours_per_unit[match.group(2) or "s"]


def gpu_price(flavor: str, price_per_hour: float | None = None) -> float:
    if price_per_hour is not None:
        return price_per_hour
    if flavor not in GPU_PRICES:
        raise SystemExit(f"No known price for flavor {flavor!r}: add it to cloud.GPU_PRICES or pass --price-per-hour.")
    return GPU_PRICES[flavor]


def worst_case_gpu_usd(flavor: str, timeout: str, price_per_hour: float | None = None) -> float:
    """What one job can cost at most: its price for its whole time limit."""
    return round(gpu_price(flavor, price_per_hour) * timeout_hours(timeout), 2)


# ---------------------------------------------------------------- tokens

def hf_token() -> str | None:
    """$SWITCHBOARD_SECRETS_DIR/HF_TOKEN, else $HF_TOKEN. Never printed or logged."""
    secrets_directory = os.environ.get("SWITCHBOARD_SECRETS_DIR")
    if secrets_directory:
        try:
            token = (Path(secrets_directory) / "HF_TOKEN").read_text().strip()
            if token:
                return token
        except OSError:
            pass
    return os.environ.get("HF_TOKEN") or None


def in_the_image() -> bool:
    """True inside the harness image (Cloud Run): it has no .git, only the built tree."""
    return not (ROOT / ".git").exists()


def bucket() -> str:
    return os.environ.get("SHIPWRIGHT_RUNS_BUCKET") or DEFAULT_BUCKET


def gcloud_binary() -> str | None:
    """$SHIPWRIGHT_GCLOUD, else gcloud on the PATH, else Homebrew's. None when there is none."""
    explicit = os.environ.get("SHIPWRIGHT_GCLOUD")
    if explicit:
        return explicit
    found = shutil.which("gcloud")
    if found:
        return found
    if Path(HOMEBREW_GCLOUD).exists():
        return HOMEBREW_GCLOUD
    return None


def gcloud_config(binary: str | None = None) -> str:
    """$SHIPWRIGHT_GCLOUD_CONFIG, else `lexsi` if this machine has it, else `personal`.

    AGENTS.md: every gcloud command names the account config; the active one is never trusted."""
    from_environment = os.environ.get("SHIPWRIGHT_GCLOUD_CONFIG")
    if from_environment:
        return from_environment
    binary = binary or gcloud_binary()
    if binary is None:
        return FALLBACK_GCLOUD_CONFIG
    try:
        listed = subprocess.run([binary, "config", "configurations", "list", "--format=value(name)"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return FALLBACK_GCLOUD_CONFIG
    if TEAM_GCLOUD_CONFIG in listed.stdout.split():
        return TEAM_GCLOUD_CONFIG
    return FALLBACK_GCLOUD_CONFIG


def gcloud_command(*arguments: str) -> list[str]:
    """gcloud with the project and the account config named, as AGENTS.md requires."""
    binary = gcloud_binary()
    if binary is None:
        raise SystemExit("gcloud not found: install it or set SHIPWRIGHT_GCLOUD")
    return [binary, f"--configuration={gcloud_config(binary)}", f"--project={PROJECT}", *arguments]


_bearer_tokens = None


def bearer_token() -> str:
    """A Google access token for GCS: the metadata server inside GCP, else gcloud. Cached, refreshed before expiry."""
    global _bearer_tokens
    if _bearer_tokens is None:
        from shipwright.llm import vertex

        if vertex.on_gcp():
            _bearer_tokens = vertex.TokenProvider(fetch=vertex.metadata_token)
        else:
            _bearer_tokens = vertex.provider()
    return _bearer_tokens.get()


# ---------------------------------------------------------------- GCS (HTTP JSON API)

UPLOAD_URL = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={name}"
LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
OBJECT_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o/{name}?alt=media"


def upload_object(http_client, bucket_name: str, object_name: str, data: bytes, token: str | None = None) -> None:
    from shipwright.observability import instrument

    url = UPLOAD_URL.format(bucket=bucket_name, name=quote(object_name, safe=""))
    headers = {"Authorization": f"Bearer {token or bearer_token()}", "Content-Type": "application/octet-stream"}
    with instrument.external("gcs_upload", f"gs://{bucket_name}/{object_name}", bytes=len(data)) as call_record:
        response = http_client.post(url, content=data, headers=headers)
        call_record["status"] = response.status_code
        response.raise_for_status()


def upload_dir(directory: Path, owner: str) -> int:
    """Copy a local run dir to gs://<bucket>/<owner>/<dir name>/. Returns the file count."""
    import httpx

    files = [path for path in sorted(directory.rglob("*")) if path.is_file()]
    with httpx.Client(timeout=300) as http_client:
        for path in files:
            object_name = f"{owner}/{directory.name}/{path.relative_to(directory).as_posix()}"
            upload_object(http_client, bucket(), object_name, path.read_bytes())
    return len(files)


def fetch_dir(http_client, bucket_name: str, prefix: str, runs_directory: Path, token: str | None = None) -> int:
    """Download gs://<bucket>/<owner>/<dir>/ into runs/<dir>/. Returns the file count; FileNotFoundError when empty."""
    from shipwright.observability import instrument

    if prefix.startswith(bucket_name + "/"):
        prefix = prefix[len(bucket_name) + 1:]
    run_directory_name = prefix.rsplit("/", 1)[-1]
    if not run_directory_name or run_directory_name in (".", ".."):
        raise ValueError(f"bad run dir prefix {prefix!r}")

    headers = {"Authorization": f"Bearer {token or bearer_token()}"}
    downloaded_count = 0
    page_token = None
    with instrument.external("gcs_download", f"gs://{bucket_name}/{prefix}/"):
        while True:
            list_parameters = {"prefix": prefix + "/", "fields": "items(name),nextPageToken"}
            if page_token:
                list_parameters["pageToken"] = page_token
            list_response = http_client.get(LIST_URL.format(bucket=bucket_name), params=list_parameters, headers=headers)
            list_response.raise_for_status()
            list_body = list_response.json()
            for item in list_body.get("items", []):
                relative_path = item["name"][len(prefix) + 1:]
                if not relative_path or relative_path.endswith("/") or ".." in Path(relative_path).parts:
                    continue
                object_url = OBJECT_URL.format(bucket=bucket_name, name=quote(item["name"], safe=""))
                object_response = http_client.get(object_url, headers=headers)
                object_response.raise_for_status()
                destination = runs_directory / run_directory_name / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(object_response.content)
                downloaded_count += 1
            page_token = list_body.get("nextPageToken")
            if not page_token:
                break
    if not downloaded_count:
        raise FileNotFoundError(f"nothing under gs://{bucket_name}/{prefix}/")
    return downloaded_count
