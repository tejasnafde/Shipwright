"""pull_runs: orchestrator logs go to runs/_meta/, real runs to runs/."""

from __future__ import annotations

import subprocess

from shipwright.control_room import gcs_runs


def test_orchestrator_logs_are_pulled_into_meta(tmp_path, monkeypatch):
    copied_to = {}

    def fake_copy(command, **keyword_arguments):
        copied_to[command[-2].rsplit("/", 1)[-1]] = command[-1]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(gcs_runs, "gcloud_binary", lambda: "gcloud")
    monkeypatch.setattr(gcs_runs, "_gcloud_command", lambda binary, *arguments: [binary, *arguments])
    monkeypatch.setattr(gcs_runs, "list_remote_runs", lambda owner: ["r2-baseline", "train_job-20261111-210754"])
    monkeypatch.setattr(gcs_runs.subprocess, "run", fake_copy)

    gcs_runs.pull_runs(["ishank"], tmp_path)

    assert copied_to["r2-baseline"] == str(tmp_path) + "/"
    assert copied_to["train_job-20261111-210754"] == str(tmp_path / "_meta") + "/"
