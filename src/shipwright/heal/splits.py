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
    cfg = yaml.safe_load(Path(path or SPLITS_PATH).read_text())
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    held, hidden = set(cfg["heldout"]["task_ids"]), set(cfg["hidden"]["task_ids"])
    if len(held) != len(cfg["heldout"]["task_ids"]) or len(hidden) != len(cfg["hidden"]["task_ids"]):
        raise ValueError("duplicate task ids in a split")
    if held & hidden:
        raise ValueError(f"held-out and hidden overlap: {sorted(held & hidden)}")
    if cfg["train"]["task_split"] in (cfg["heldout"]["task_split"], cfg["hidden"]["task_split"]):
        raise ValueError("train must be a different tau2 split from held-out and hidden")


def heldout_ids(cfg: dict | None = None) -> list[str]:
    return [str(i) for i in (cfg or load())["heldout"]["task_ids"]]


def hidden_ids(cfg: dict | None = None) -> list[str]:
    return [str(i) for i in (cfg or load())["hidden"]["task_ids"]]


def check_run(task_split: str, task_ids: list[str] | None, open_hidden: bool = False,
              cfg: dict | None = None) -> None:
    """Raise ValueError if a run would touch hidden tasks without ``open_hidden``."""
    cfg = cfg or load()
    if open_hidden or task_split != cfg["hidden"]["task_split"]:
        return
    if not task_ids:
        raise ValueError(f"a run on the {task_split!r} split must pass --task-ids (the held-out ids), "
                         "or tau2 may pick hidden tasks; --open-hidden opens the hidden set")
    leaked = sorted(set(map(str, task_ids)) & set(hidden_ids(cfg)), key=lambda x: (len(x), x))
    if leaked:
        raise ValueError(f"hidden task ids {leaked}: the hidden set is opened once, for the final report "
                         "(--open-hidden)")
