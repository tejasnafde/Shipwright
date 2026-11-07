# GCP setup for the Lexsi hackathon (for you and your coding agent)

Give this whole file to your agent. It has the facts and the commands it needs.

## Facts

- Project ID: `lexsi-hackathon`. Default region: `asia-south1` (Mumbai).
- Your role: **Editor**. You can create and delete almost anything.
- You can NOT change IAM (grant roles to people or service accounts). Ask Tejas for that.
- Billing is Tejas's card. Currency is INR.
- **Budget: alerts at Rs 500, 1000, 1500, 2000, 2500, 3000 per month.**
- **Hard cap at Rs 3000: billing is switched off automatically for the whole project.**
  Every VM, bucket and service in it stops. Only Tejas can switch it back on.
  Billing data lags a few hours, so stop expensive things early.

## Rules for agents

1. Always pass `--project=lexsi-hackathon` (or use the `lexsi` config below).
2. Label every resource: `--labels=app=lexsi-hackathon`. Costs are split by this label.
3. Use **Spot** VMs for GPU work (about 60 to 90% cheaper). Save checkpoints often, because Spot VMs can stop at any time.
4. **Stop or delete a VM the moment a job ends.** An idle GPU VM costs money every hour.
5. Never put a secret (HF token, API key) in code, in a VM metadata field, or in git. Use Secret Manager.
6. Before you end a session, run the "what is running" check at the bottom of this file.

## 1. Install the gcloud CLI

```sh
# macOS
brew install --cask google-cloud-sdk
# Linux / WSL
curl -sSL https://sdk.cloud.google.com | bash && exec -l $SHELL
gcloud --version
```

## 2. Log in and make a named config

A named config keeps this project apart from any other Google account you use.

```sh
gcloud config configurations create lexsi
gcloud auth login                      # sign in with the Gmail that Tejas added
gcloud config set project lexsi-hackathon
gcloud config set compute/region asia-south1
gcloud config set compute/zone asia-south1-a

# Credentials for Python SDKs (google-cloud-storage, vertexai, ...)
gcloud auth application-default login
gcloud auth application-default set-quota-project lexsi-hackathon
```

Later, switch to it with `gcloud config configurations activate lexsi`,
or pass `--configuration=lexsi` on one command.

## 3. Turn on the APIs you need (once per project)

```sh
gcloud services enable compute.googleapis.com storage.googleapis.com \
  secretmanager.googleapis.com artifactregistry.googleapis.com \
  run.googleapis.com aiplatform.googleapis.com
```

## 4. GPU quota: new projects start at 0 GPUs

A new project cannot start any GPU until the quota is raised. Check it:

```sh
gcloud compute regions describe asia-south1 --format=json \
  | grep -B1 -A2 -E 'GPU|NVIDIA'
gcloud compute project-info describe --format=json | grep -A2 GPUS_ALL_REGIONS
```

If `GPUS_ALL_REGIONS` or the GPU limit for the region is 0, ask for an increase in the console:
IAM & Admin > Quotas > filter "GPUs (all regions)" and the GPU type (for example
"Preemptible NVIDIA L4 GPUs") > Edit quota. Ask for 1. Approval can take minutes to a day,
so do this first.

Which GPUs a zone has:

```sh
gcloud compute accelerator-types list --filter="zone~asia-south1" --format="table(name,zone)"
```

If a GPU is not in Mumbai, use `us-central1` or `europe-west4` and pass `--zone`.

## 5. Examples

### A bucket for traces, datasets and checkpoints

```sh
gcloud storage buckets create gs://lexsi-hackathon-data \
  --location=asia-south1 --uniform-bucket-level-access
gcloud storage buckets update gs://lexsi-hackathon-data --update-labels=app=lexsi-hackathon

gcloud storage cp runs/round1.jsonl gs://lexsi-hackathon-data/traces/
gcloud storage ls gs://lexsi-hackathon-data/traces/
```

### A Spot GPU VM for training (Deep Learning VM image, CUDA ready)

Find a current PyTorch image family first, because the names change:

```sh
gcloud compute images list --project=deeplearning-platform-release \
  --filter="family~pytorch AND family~cu" --format="value(family)" | sort -u | tail -5
```

Then create the VM (change the family and the GPU type to what the commands above show):

```sh
gcloud compute instances create train-1 \
  --zone=asia-south1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-project=deeplearning-platform-release --image-family=<FAMILY> \
  --boot-disk-size=150GB \
  --metadata=install-nvidia-driver=True \
  --labels=app=lexsi-hackathon

gcloud compute ssh train-1 --zone=asia-south1-a
nvidia-smi                                # on the VM: check the GPU
```

Stop it (disk kept, small cost) or delete it (no cost):

```sh
gcloud compute instances stop train-1 --zone=asia-south1-a
gcloud compute instances delete train-1 --zone=asia-south1-a
```

### Secrets (HF token, teacher model API key)

```sh
printf '%s' "$HF_TOKEN" | gcloud secrets create HF_TOKEN --data-file=- \
  --labels=app=lexsi-hackathon
gcloud secrets versions access latest --secret=HF_TOKEN
```

On a VM, read it at run time: `export HF_TOKEN=$(gcloud secrets versions access latest --secret=HF_TOKEN)`.
The VM's default service account may need access to the secret. That is an IAM change, so ask Tejas.

### A small web service (dashboard or demo) on Cloud Run

```sh
gcloud run deploy lexsi-demo --source=. --region=asia-south1 \
  --allow-unauthenticated --labels=app=lexsi-hackathon --max-instances=2
```

## 6. What is running right now (run this before you stop work)

```sh
gcloud compute instances list --format="table(name,zone,status,machineType.basename())"
gcloud compute disks list --format="table(name,zone,sizeGb,users.basename())"
gcloud run services list
gcloud storage du -s -r gs://lexsi-hackathon-data 2>/dev/null
```

A `RUNNING` VM with a GPU is the expensive item. Unattached disks also cost a little.

## If a command fails

- `PERMISSION_DENIED` on IAM, roles or service accounts: Editor cannot do that. Ask Tejas.
- `Quota 'GPUS_ALL_REGIONS' exceeded`: see section 4.
- `ZONE_RESOURCE_POOL_EXHAUSTED`: no Spot GPU free in that zone now. Try another zone.
- `Billing account for project is not found` or `billing is disabled`: the Rs 3000 cap fired. Tell Tejas.
