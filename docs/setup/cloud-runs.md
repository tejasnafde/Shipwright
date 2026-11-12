# Live runs on GCP: Cloud Run Jobs (one per person)

Live harness runs (baseline, teacher runs, classify with a judge, gate evals that call APIs) run on
GCP CPU, in a Cloud Run Job. They never run on a laptop. The models are remote (HF router, Vertex
AI), so a job needs no GPU.

**The shared-credit rule still applies.** A job that calls the HF router (Tiny Aya) spends the
team's HF credits. Start one only when a human on the team asked for that run, and state the
expected cost first. `--help` and `--dry-run` cost nothing.

## What exists

| Branch | Cloud Build trigger | Image tag | Cloud Run Job | Output |
|---|---|---|---|---|
| `tejas` | `build-tejas` | `harness:tejas` | `shipwright-tejas` | `gs://lexsi-hackathon-runs/tejas/` |
| `ig/main` | `build-ishank` | `harness:ishank` | `shipwright-ishank` | `gs://lexsi-hackathon-runs/ishank/` |
| `mitushi` | `build-mitushi` | `harness:mitushi` | `shipwright-mitushi` | `gs://lexsi-hackathon-runs/mitushi/` |
| `main` | `build-main` | `harness:main` | `shipwright-main` | `gs://lexsi-hackathon-runs/main/` |

- Image: `asia-south1-docker.pkg.dev/lexsi-hackathon/shipwright/harness`, built from
  `infra/docker/Dockerfile` by `infra/cloudbuild.yaml`. Each build pushes `harness:<owner>` and
  `harness:<short sha>`, then points `shipwright-<owner>` at the pushed digest (and sets
  `SHIPWRIGHT_IMAGE_DIGEST`). So a job always runs the latest build of its branch, and
  `repro.json` records the exact image.
- Artifact Registry cleanup: keeps the 4 branch tags, the 40 newest images, and deletes the rest
  after 1 day.
- Jobs: region `asia-south1`, 1 CPU, 2 GiB, 1 task, no retries, task timeout 4 h (Cloud Run
  allows up to 168 h; raise it with `gcloud run jobs update --task-timeout`).
  Env: `OWNER=<owner>`, `HF_TOKEN` from Secret Manager `SHIPWRIGHT_HF_TOKEN:latest` (a secret
  ref, never a plain env value).
- Runtime service account `shipwright-runner@`: Vertex AI user, object admin on the runs bucket
  only, accessor on `SHIPWRIGHT_HF_TOKEN` only, log writer, Cloud Trace agent (2026-11-09). Vertex calls use its token from the
  metadata server (`src/shipwright/llm/vertex.py`), not gcloud.
- Build service account `shipwright-builder@`: pushes to the `shipwright` repo, `run.developer`
  on the 4 jobs only, acts as `shipwright-runner@` only.

## Run your job

The job's container runs `scripts/job_entry.py <script.py> [args...]`. Pass the script and its
flags with `--args` (comma separated). Each person uses their own job.

```sh
GC="gcloud --configuration=lexsi --project=lexsi-hackathon"   # Tejas: --configuration=personal

# Free: the offline dry run (scripted fake LLM)
$GC run jobs execute shipwright-ishank --region=asia-south1 --wait \
  --args=baseline.py,--dry-run,--num-tasks,2

# Spends HF credits (Tiny Aya): only when the team asked for this run
$GC run jobs execute shipwright-ishank --region=asia-south1 \
  --args=baseline.py,--num-tasks,20,--user-model,vertex/openai/gpt-oss-120b-maas
```

A value that holds a comma needs another delimiter: `--args='^|^baseline.py|--task-ids|1,2'`.
`--args` rejects a list with the same value twice (`"2" cannot be specified multiple times`, for
example `--trials 2` with task id 2): write flags as `--trials=2` (2026-11-12).
Without `--wait`, the command returns at once; follow the run in the console
(Cloud Run, Jobs, `shipwright-<owner>`, Executions) or with
`$GC run jobs executions list --job=shipwright-ishank --region=asia-south1`.

`--output-dir` must be under `runs/` (the default is fine): only `runs/<dir>/` is uploaded.

Each execution starts empty. To read the run dir of an earlier execution, put `--fetch <owner>/<dir>`
(repeatable) before the script: it downloads `gs://lexsi-hackathon-runs/<owner>/<dir>/` to `runs/<dir>/`
first (2026-11-11). A fetched dir is uploaded again only if the script changes it.

```sh
$GC run jobs execute shipwright-tejas --region=asia-south1 --wait \
  --args=--fetch,tejas/r1-baseline-train,classify.py,runs/r1-baseline-train
```

## Read the results

Something failed? `scripts/why.py <execution name>` (docs/ops/debugging.md) prints where and why,
with the Logs Explorer and Cloud Trace links. The execution name is the run id of every log line.

The entrypoint uploads every run dir the script wrote, and the full console log, even when the
script fails:

```sh
$GC storage ls gs://lexsi-hackathon-runs/ishank/
$GC storage cp -r gs://lexsi-hackathon-runs/ishank/<run-dir> runs/
$GC storage cat gs://lexsi-hackathon-runs/ishank/jobs/<execution-name>.log
$GC storage cat gs://lexsi-hackathon-runs/ishank/jobs/<execution-name>.json   # run dirs, exit code, links
```

Project Editors (Ishank, Mitushi) can read and write the bucket and execute the jobs; no extra grant
is needed.

## Build a new image

- Automatic: a push to your branch runs your trigger (once the GitHub connection exists, below).
- Manual (Cloud Build, never local Docker):

```sh
$GC builds submit --region=asia-south1 --config=infra/cloudbuild.yaml \
  --gcs-source-staging-dir=gs://lexsi-hackathon_cloudbuild/source \
  --substitutions=_OWNER=ishank,_SHA=$(git rev-parse --short HEAD) .
```

`.gcloudignore` keeps `.venv`, `runs/` and `third_party/` out of the upload. A build takes about
5 to 10 minutes.

## Triggers (one-time setup, Tejas)

The triggers need the GitHub repo connected to Cloud Build. Connect it once in the console:
<https://console.cloud.google.com/cloud-build/repositories/2nd-gen?project=lexsi-hackathon>
(region `asia-south1`, connection name `shipwright-github`, then "Link repository"
`tejasnafde/Shipwright` with the name `shipwright`). Then create the triggers:

```sh
GC="gcloud --configuration=personal --project=lexsi-hackathon"
REPO=projects/lexsi-hackathon/locations/asia-south1/connections/shipwright-github/repositories/shipwright
SA=projects/lexsi-hackathon/serviceAccounts/shipwright-builder@lexsi-hackathon.iam.gserviceaccount.com
for pair in tejas:tejas ishank:ig/main mitushi:mitushi main:main; do
  owner=${pair%%:*}; branch=${pair#*:}
  $GC builds triggers create github --region=asia-south1 --name=build-$owner \
    --repository=$REPO --branch-pattern="^${branch}\$" \
    --build-config=infra/cloudbuild.yaml --service-account=$SA \
    --include-logs-with-status
done
```

## Cost

- Cloud Run Jobs bill only while a task runs: about $0.09 per hour at 1 CPU and 2 GiB in
  asia-south1. A 20-task baseline takes well under an hour of job time.
- The API calls cost more than the CPU: HF router calls use the shared HF credits, Vertex calls
  bill to this project (prices in `src/shipwright/llm/costs.py`).
- Cloud Build: the first 2500 build-minutes per month on the default machine are free.
- Artifact Registry: 0.5 GB free, then about $0.10 per GB-month; the cleanup policy keeps it small.

## Pull runs to your laptop

`scripts/pull_runs.py --owner tejas|ishank|mitushi|main|all [--run NAME]` copies the run dirs a cloud job
uploaded from `gs://lexsi-hackathon-runs/<owner>/` into `runs/` (the control room reads them from there).
It uses the `lexsi` gcloud config, or `SHIPWRIGHT_GCLOUD_CONFIG`.
