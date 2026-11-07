"""Hard spend cap for the lexsi-hackathon GCP project.

GCP budgets only alert. This function listens on the shared budget topic
(teejayproject/gcp-cost-alerts) and unlinks billing from lexsi-hackathon when
its budget's actual spend reaches the budget amount. That stops every paid
resource in the project. Relink with:
  gcloud --configuration=personal billing projects link lexsi-hackathon \
    --billing-account=014156-885762-D612A7

ponytail: billing data lags a few hours, so real spend can pass the cap
before this fires. Lower the budget if that margin matters.
"""
import base64
import json

BUDGET_NAME = "lexsi-hackathon"
PROJECT = "lexsi-hackathon"


def should_kill(payload):
    if payload.get("budgetDisplayName") != BUDGET_NAME:
        return False  # the topic also carries other budgets
    return float(payload.get("costAmount", 0)) >= float(payload.get("budgetAmount", 1e18))


def billing_kill(cloud_event):
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    payload = json.loads(base64.b64decode(cloud_event.data["message"]["data"]).decode())
    if not should_kill(payload):
        return
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-billing"])
    url = f"https://cloudbilling.googleapis.com/v1/projects/{PROJECT}/billingInfo"
    r = AuthorizedSession(creds).put(url, json={"billingAccountName": ""})
    print(f"billing unlink for {PROJECT}: {r.status_code} {r.text[:300]}")
    r.raise_for_status()


if __name__ == "__main__":
    p = {"budgetDisplayName": BUDGET_NAME, "budgetAmount": 3000}
    assert not should_kill({**p, "costAmount": 2999.9})
    assert should_kill({**p, "costAmount": 3000})
    assert not should_kill({**p, "budgetDisplayName": "GCP monthly cap (Discord)", "costAmount": 9999})
    assert not should_kill({"budgetDisplayName": BUDGET_NAME})
    print("ok")
