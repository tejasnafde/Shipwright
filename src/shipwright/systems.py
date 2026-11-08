"""A "system under test" for the gate: base model, optional LoRA adapter, optional tool patches.

One YAML file per system (see ``configs/systems/``):

    name: base-r1-tools            # unique, used in paths and the vLLM served name
    model: CohereLabs/tiny-aya-global
    adapter:                       # optional (weight fix)
      repo: mitushigoyal5/shipwright-lora
      revision: round-1
    tool_patches:                  # optional (tool fix)
      ledger: patches/ledger.jsonl
      ids: applied                 # or an explicit list of patch ids
    route: tool_fix                # none | tool_fix | weight_fix (for the record)
    targets: [tool_example_copied] # failure labels this candidate should lower

The gate compares ``current`` against ``candidate``. A tool-fix candidate is the current
system plus patches; a weight-fix candidate is an adapter plus the current patches.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ALLOWED_MODELS = {
    # AGENTS.md hard rule: only these base models.
    "CohereLabs/tiny-aya-base", "CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-fire",
    "CohereLabs/tiny-aya-water", "CohereLabs/tiny-aya-earth",
    "CohereLabs/aya-expanse-8b", "CohereLabs/aya-expanse-32b",
}
ROUTES = ("none", "tool_fix", "weight_fix")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def validate_system(cfg: dict, root: Path | None = None) -> dict:
    """Check one system config and fill defaults. Raises ValueError on a bad config."""
    if not isinstance(cfg, dict):
        raise ValueError("system config must be a mapping")
    name = cfg.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"system name must match {_NAME.pattern}: {name!r}")
    model = cfg.get("model")
    if model not in ALLOWED_MODELS:
        raise ValueError(f"{name}: model {model!r} is not an allowed hackathon model")
    out: dict[str, Any] = {"name": name, "model": model, "adapter": None, "tool_patches": None,
                           "route": cfg.get("route", "none"), "targets": list(cfg.get("targets") or [])}
    if out["route"] not in ROUTES:
        raise ValueError(f"{name}: route must be one of {ROUTES}")
    adapter = cfg.get("adapter")
    if adapter:
        if not adapter.get("repo") or not adapter.get("revision"):
            raise ValueError(f"{name}: adapter needs repo and revision")
        out["adapter"] = {"repo": adapter["repo"], "revision": str(adapter["revision"])}
    patches = cfg.get("tool_patches")
    if patches:
        ids = patches.get("ids", "applied")
        if not (ids == "applied" or (isinstance(ids, list) and all(isinstance(i, str) for i in ids))):
            raise ValueError(f"{name}: tool_patches.ids must be 'applied' or a list of ids")
        ledger = patches.get("ledger") or "patches/ledger.jsonl"
        out["tool_patches"] = {"ledger": str(ledger), "ids": ids}
        if root is not None and not (root / ledger).exists():
            raise ValueError(f"{name}: ledger {ledger} not found under {root}")
    return out


def load_system(path: Path, root: Path | None = None) -> dict:
    import yaml

    return validate_system(yaml.safe_load(Path(path).read_text()), root)


def served_name(system: dict) -> str:
    """The model name vLLM serves this system under.

    The base model keeps its repo id. An adapter gets ``tiny-aya-<system name>``: the
    name must contain ``tiny-aya`` so ``prompted_tools`` gives it the prompted-JSON view.
    """
    if not system.get("adapter"):
        return system["model"]
    return f"tiny-aya-{system['name']}"


def agent_model_id(system: dict) -> str:
    """litellm id for the agent (OpenAI-compatible endpoint)."""
    return "openai/" + served_name(system)


def resolve_patch_ids(system: dict, ledger_rows: list[dict]) -> list[str]:
    """The exact patch ids a system runs with (``applied`` is resolved now, for the record)."""
    from shipwright.tool_fix import select_patches

    tp = system.get("tool_patches")
    if not tp:
        return []
    return [r["id"] for r in select_patches(ledger_rows, tp["ids"])]


def baseline_args(system: dict, patch_ids: list[str], root: Path) -> list[str]:
    """Extra ``scripts/baseline.py`` arguments for this system's tool patches."""
    tp = system.get("tool_patches")
    if not tp:
        return []
    return ["--tool-patches", str(root / tp["ledger"]), "--tool-patch-ids", *patch_ids]


def serve_plan(systems: list[dict], mode: str = "lora") -> list[dict]:
    """vLLM servers to start, in order, and the systems each one serves.

    ``lora``: one server with the base model and every adapter as a LoRA module.
    ``merged``: one server per system; an adapter system serves its merged weights
    (fallback if vLLM cannot load the LoRA for this architecture).
    All systems must share one base model.
    """
    models = {s["model"] for s in systems}
    if len(models) != 1:
        raise ValueError(f"systems use different base models: {sorted(models)}")
    base = models.pop()
    if mode == "lora":
        loras: dict[str, dict] = {}
        for s in systems:
            if s.get("adapter"):
                loras[served_name(s)] = s["adapter"]
        return [{"base": base, "served_base": base, "loras": loras, "merged": None,
                 "systems": [s["name"] for s in systems]}]
    if mode == "merged":
        plan = []
        for s in systems:
            plan.append({"base": base, "served_base": served_name(s), "loras": {},
                         "merged": s.get("adapter"), "systems": [s["name"]]})
        return plan
    raise ValueError(f"serve mode must be lora or merged, got {mode!r}")
