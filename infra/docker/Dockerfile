# Harness image for the Cloud Run Jobs shipwright-<owner> (docs/cloud-runs.md).
# Adapted from Ishank's ig/main infra/docker/Dockerfile.
#
# CPU only: the agent, user simulator, teacher and judge are remote APIs (HF router, Vertex AI).
# No model weights in the image. Build it with Cloud Build only (infra/cloudbuild.yaml), never on a laptop.
FROM python:3.12-slim-bookworm

# git: scripts/setup_tau2.sh fetches tau2-bench at a pinned commit.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app

# Install step first, so a code-only change reuses the cached dependency layer.
COPY pyproject.toml README.md ./
COPY scripts/setup_tau2.sh scripts/
COPY src/ src/
# The same install as a laptop and CI: tau2 (pinned, editable), shipwright[dev], agenttune --no-deps.
# Then drop what a run never reads: tau2's git history, its sample results and voice data (about 600 MB).
RUN ./scripts/setup_tau2.sh \
    && rm -rf third_party/tau2-bench/.git third_party/tau2-bench/data/tau2/results \
              third_party/tau2-bench/data/voice /root/.cache

COPY scripts/ scripts/
COPY configs/ configs/
COPY patches/ patches/

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    SHIPWRIGHT_VERTEX_AUTH=metadata

# Args are a script under scripts/ and its flags, e.g. `baseline.py --dry-run --num-tasks 2`.
ENTRYPOINT ["python", "/app/scripts/job_entry.py"]
