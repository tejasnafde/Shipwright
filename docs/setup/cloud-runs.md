# Cloud Run Jobs

Every live run executes in a Cloud Run Job, never on a laptop: the models are remote (HF router, Vertex AI), so a job
needs no GPU. The loop launches its own execution (`shipwright run --launch`); this page is for running one stage by
hand and for the deployment itself.

## Deployment

| Branch | Cloud Build trigger | Image tag | Job | Output |
|---|---|---|---|---|
| `main` | `build-main` | `harness:main` | `shipwright-main` | `gs://lexsi-hackathon-runs/main/` |
| a developer branch | `build-<owner>` | `harness:<owner>` | `shipwright-<owner>` | `gs://lexsi-hackathon-runs/<owner>/` |

A push to a deploy branch builds `infra/docker/Dockerfile` with `infra/cloudbuild.yaml`, pushes `harness:<owner>` and
`harness:<short sha>`, and points the job at the pushed digest (`SHIPWRIGHT_IMAGE_DIGEST`), so a job always runs the
latest build of its branch and `repro.json` records the exact image. Images are only built by Cloud Build.

Job settings: region `asia-south1`, 1 CPU, 2 GiB, one task, no automatic retries (the loop retries a failed stage
itself), task timeout set per run from the round config. Environment: `OWNER=<owner>` and `HF_TOKEN` from Secret
Manager.

## Run one stage

The container runs `scripts/job_entry.py <script.py> [args...]`, which resolves the script under `scripts/`, runs it,
and uploads every run directory it wrote plus the console log, also on failure.

```sh
GC="gcloud --configuration=lexsi --project=lexsi-hackathon"

# Free: the offline dry run
$GC run jobs execute shipwright-main --region=asia-south1 --wait --args=baseline.py,--dry-run,--num-tasks,2

# Reuse an earlier run dir: --fetch=<owner>/<dir> downloads it to runs/<dir>/ first
$GC run jobs execute shipwright-main --region=asia-south1 --wait \
  --args=--fetch=main/r16-cand-train,classify.py,runs/r16-cand-train
```

- `--args` is comma-separated and rejects a repeated value: write flags as `--flag=value`. A value with a comma needs
  another delimiter: `--args='^|^baseline.py|--task-ids|1,2'`.
- `--output-dir` must be under `runs/`; only `runs/<dir>/` is uploaded.
- Each execution starts empty; a fetched directory is uploaded again only if the script changed it.

## Read the results

```sh
$GC storage ls gs://lexsi-hackathon-runs/<owner>/
$GC storage cat gs://lexsi-hackathon-runs/<owner>/jobs/<execution>.json    # run dirs, exit code, links
.venv/bin/python scripts/pull_runs.py --owner <owner> [--run NAME]         # copy run dirs into runs/
.venv/bin/python scripts/why.py <execution>                                # where and why it failed
```

The control room reads `runs/`: `streamlit run src/shipwright/control_room/app.py`.

## Build an image by hand

```sh
$GC builds submit --region=asia-south1 --config=infra/cloudbuild.yaml \
  --gcs-source-staging-dir=gs://lexsi-hackathon_cloudbuild/source \
  --substitutions=_OWNER=<owner>,_SHA=$(git rev-parse --short HEAD) .
```

`.gcloudignore` keeps `.venv`, `runs/` and `third_party/` out of the upload. A build takes 5 to 10 minutes.
