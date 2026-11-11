"""configs/splits.yaml and the hidden-set guard (AGENTS.md, Evaluation rules)."""

from __future__ import annotations

import copy

import pytest

from shipwright.heal import splits


def test_repo_splits_are_disjoint_20_20_and_cover_the_tau2_test_split():
    cfg = splits.load()
    held, hidden = splits.heldout_ids(cfg), splits.hidden_ids(cfg)
    assert len(held) == len(hidden) == 20 and not set(held) & set(hidden)
    assert cfg["status"] == "proposed" and cfg["train"]["task_split"] == "train"
    tau2 = pytest.importorskip("tau2.registry")
    load = tau2.registry.get_tasks_loader("retail")
    test_ids = {str(t.id) for t in load(task_split_name="test")}
    train_ids = {str(t.id) for t in load(task_split_name="train")}
    assert set(held) | set(hidden) == test_ids
    assert not (set(held) | set(hidden)) & train_ids


def test_check_run_guards_the_hidden_set():
    cfg = splits.load()
    held, hidden = splits.heldout_ids(cfg), splits.hidden_ids(cfg)
    splits.check_run("train", None, cfg=cfg)  # the train split never touches test tasks
    splits.check_run("test", held, cfg=cfg)
    with pytest.raises(ValueError, match="must pass --task-ids"):
        splits.check_run("test", None, cfg=cfg)  # tau2 could pick hidden tasks itself
    with pytest.raises(ValueError, match="hidden task ids"):
        splits.check_run("test", [held[0], hidden[0]], cfg=cfg)
    splits.check_run("test", hidden, open_hidden=True, cfg=cfg)  # the one final-report run


def test_validate_rejects_overlap_and_shared_tau2_split():
    cfg = splits.load()
    bad = copy.deepcopy(cfg)
    bad["hidden"]["task_ids"][0] = bad["heldout"]["task_ids"][0]
    with pytest.raises(ValueError, match="overlap"):
        splits.validate(bad)
    bad = copy.deepcopy(cfg)
    bad["train"]["task_split"] = "test"
    with pytest.raises(ValueError, match="different tau2 split"):
        splits.validate(bad)


def test_baseline_refuses_hidden_tasks(tmp_path):
    import subprocess
    import sys

    from shipwright.heal.splits import ROOT

    hidden = splits.hidden_ids()[0]
    r = subprocess.run([sys.executable, "scripts/baseline.py", "--dry-run", "--task-split", "test",
                        "--task-ids", hidden, "--output-dir", str(tmp_path / "x")],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "hidden task ids" in (r.stdout + r.stderr)
