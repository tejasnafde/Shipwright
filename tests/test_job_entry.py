"""scripts/job_entry.py: which run dirs it uploads and where. Offline: GCS is mocked."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("job_entry", ROOT / "scripts" / "job_entry.py")
job_entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job_entry)


def test_changed_dirs_new_and_modified(tmp_path):
    runs = tmp_path / "runs"
    (runs / "old").mkdir(parents=True)
    (runs / "old" / "a.json").write_text("{}")
    (runs / "kept").mkdir()
    (runs / "kept" / "b.json").write_text("{}")
    before = job_entry.run_dirs(runs)
    (runs / "new").mkdir()
    (runs / "new" / "c.json").write_text("{}")
    f = runs / "kept" / "b.json"
    os.utime(f, (f.stat().st_mtime + 10, f.stat().st_mtime + 10))
    assert job_entry.changed_dirs(before, job_entry.run_dirs(runs)) == ["kept", "new"]
    assert job_entry.run_dirs(tmp_path / "missing") == {}


def test_main_runs_script_and_uploads_under_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(job_entry, "ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fake.py").write_text(
        "import pathlib, sys\n"
        "d = pathlib.Path('runs/r1/sub'); d.mkdir(parents=True)\n"
        "(d / 'x.txt').write_text('hi')\n"
        "print('ran', sys.argv[1:])\n"
        "sys.exit(3)\n")
    uploaded = {}
    monkeypatch.setattr(job_entry, "upload", lambda client, bucket, name, data: uploaded.__setitem__(
        f"{bucket}/{name}", data))
    monkeypatch.setenv("OWNER", "tejas")
    monkeypatch.setenv("CLOUD_RUN_EXECUTION", "shipwright-tejas-abc")
    monkeypatch.delenv("SHIPWRIGHT_RUNS_BUCKET", raising=False)
    assert job_entry.main(["fake.py", "--flag"]) == 3  # the script's exit code
    assert uploaded["lexsi-hackathon-runs/tejas/r1/sub/x.txt"] == b"hi"
    log = uploaded["lexsi-hackathon-runs/tejas/jobs/shipwright-tejas-abc.log"]
    assert b"ran ['--flag']" in log


def test_main_needs_owner_and_a_real_script(monkeypatch):
    monkeypatch.delenv("OWNER", raising=False)
    assert job_entry.main(["baseline.py"]) == 2
    monkeypatch.setenv("OWNER", "tejas")
    assert job_entry.main(["no_such_script.py"]) == 2
    assert job_entry.main([]) == 2


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help(flag, capsys):
    assert job_entry.main([flag]) == 0
    assert "OWNER" in capsys.readouterr().out
