"""The frozen task splits (configs/splits.yaml): train pool, held-out (gate), hidden (final report).

``check_run`` is the guard every tau2 run goes through (scripts/baseline.py): a run may touch
hidden tasks only with ``open_hidden``, and a run on the test split must name its task ids, so
tau2 never picks hidden tasks by itself.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SPLITS_PATH = ROOT / "configs" / "splits.yaml"


def load(path: Path | None = None) -> dict:
    splits_config = yaml.safe_load(Path(path or SPLITS_PATH).read_text())
    validate(splits_config)
    return splits_config


def validate(cfg: dict) -> None:
    heldout_ids_list = cfg["heldout"]["task_ids"]
    hidden_ids_list = cfg["hidden"]["task_ids"]
    heldout_set = set(heldout_ids_list)
    hidden_set = set(hidden_ids_list)
    if len(heldout_set) != len(heldout_ids_list) or len(hidden_set) != len(hidden_ids_list):
        raise ValueError("duplicate task ids in a split")
    if heldout_set & hidden_set:
        raise ValueError(f"held-out and hidden overlap: {sorted(heldout_set & hidden_set)}")
    if cfg["train"]["task_split"] in (cfg["heldout"]["task_split"], cfg["hidden"]["task_split"]):
        raise ValueError("train must be a different tau2 split from held-out and hidden")


def heldout_ids(cfg: dict | None = None) -> list[str]:
    splits_config = cfg or load()
    return [str(task_id) for task_id in splits_config["heldout"]["task_ids"]]


def hidden_ids(cfg: dict | None = None) -> list[str]:
    splits_config = cfg or load()
    return [str(task_id) for task_id in splits_config["hidden"]["task_ids"]]


def check_run(
    task_split: str,
    task_ids: list[str] | None,
    open_hidden: bool = False,
    cfg: dict | None = None,
) -> None:
    """Raise ValueError if a run would touch hidden tasks without ``open_hidden``."""
    splits_config = cfg or load()
    if open_hidden or task_split != splits_config["hidden"]["task_split"]:
        return
    if not task_ids:
        raise ValueError(
            f"a run on the {task_split!r} split must pass --task-ids (the held-out ids), "
            "or tau2 may pick hidden tasks; --open-hidden opens the hidden set"
        )
    requested_ids = set(map(str, task_ids))
    leaked_ids = sorted(requested_ids & set(hidden_ids(splits_config)), key=lambda task_id: (len(task_id), task_id))
    if leaked_ids:
        raise ValueError(
            f"hidden task ids {leaked_ids}: the hidden set is opened once, for the final report "
            "(--open-hidden)"
        )
