# Google Cloud

Shipwright runs its CPU work (every stage that calls model APIs, and the loop's controller) as Cloud Run Jobs in one GCP
project, and calls the teacher, classifier and judge models on Vertex AI. GPUs are on HF Jobs (`docs/setup/hf-setup.md`).

## Resources

| Resource | Name | Purpose |
|---|---|---|
| Project | `lexsi-hackathon`, region `asia-south1` | everything below; every resource labelled `app=lexsi-hackathon` |
| Artifact Registry | `asia-south1-docker.pkg.dev/lexsi-hackathon/shipwright/harness` | the harness image (`infra/docker/Dockerfile`) |
| Cloud Build | `infra/cloudbuild.yaml`, one trigger per deploy branch | builds the image and points the job at its digest |
| Cloud Run Jobs | `shipwright-<owner>` (`main` and one per developer) | one execution runs one stage, or a whole loop run |
| Cloud Storage | `gs://lexsi-hackathon-runs/<owner>/` | every run directory and execution log |
| Secret Manager | `SHIPWRIGHT_HF_TOKEN` | the HF token, mounted into the job as a secret reference |
| Vertex AI | the `global` OpenAI-compatible endpoint | DeepSeek V4.1 Flash (teacher), gpt-oss-120b (classifier, judge, proposer) |
| Cloud Function | `infra/billing-kill/` | unlinks billing when the project budget is reached |

Every name lives in one place, `src/shipwright/cloud.py`; nothing else hard-codes a project, bucket, image or repo.

## Service accounts

- `shipwright-runner@` (the job's identity): Vertex AI user, object admin on the runs bucket only, accessor on
  `SHIPWRIGHT_HF_TOKEN` only, log writer, Cloud Trace agent. Inside Cloud Run, Vertex tokens come from the metadata
  server and are refreshed before they expire.
- `shipwright-builder@` (Cloud Build): pushes to the `shipwright` repository, `run.developer` on the jobs only, acts
  as `shipwright-runner@` only.

## A developer machine

The laptop only checks, plans, launches and reads; a live run refuses to start there.

```sh
brew install --cask google-cloud-sdk
gcloud config configurations create lexsi
gcloud auth login
gcloud config set project lexsi-hackathon
gcloud config set run/region asia-south1
```

Every gcloud call the code makes names `--configuration` and `--project` explicitly (`cloud.gcloud_command`), so another
active account on the same machine is never used by accident. The code uses the `lexsi` config when it exists, else
`personal`, or `SHIPWRIGHT_GCLOUD_CONFIG` when set. Vertex tokens on a laptop come from
`gcloud auth print-access-token` with that config, never from Application Default Credentials. Settings are in
`.env.example`.

APIs the project needs (once): `run`, `cloudbuild`, `artifactregistry`, `secretmanager`, `storage`, `aiplatform`,
`logging`, `cloudtrace`.

## Spend protection

- Budget alerts every Rs 500. At Rs 3,000 a month `infra/billing-kill` unlinks billing and every paid resource stops.
  Billing data lags hours, so the loop does not rely on it: each stage's worst case is checked against its cap
  before it starts (`docs/pipeline/loop.md`).
- Cloud Run bills only while an execution runs (about $0.09 an hour at 1 CPU and 2 GiB); the model API calls cost
  more than the CPU.
- Artifact Registry keeps the branch tags and the 40 newest images; older ones are deleted after a day.
