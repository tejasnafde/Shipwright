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
    tool_guards: true              # optional (tool fix): policy checks before a tool call runs
    state_reminders: true          # optional (tool fix): a status note before each agent turn (reminders.py)
    condensed_policy: true         # optional: the retail policy in fewer words (prompted_tools.CONDENSED_POLICY);
                                   # an adapter trained on rows rendered with it (build_sft --condensed-policy)
    identity_note: true            # optional: a trimmed request keeps the user id and orders (context.still_true_note)
    agent_voice_check: true        # optional: re-ask a simulated customer that talks as the agent (set on both systems)
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


def validate_system(system_config: dict, root: Path | None = None) -> dict:
    """Check one system config and fill defaults. Raises ValueError on a bad config."""
    if not isinstance(system_config, dict):
        raise ValueError("system config must be a mapping")

    name = system_config.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"system name must match {_NAME.pattern}: {name!r}")
    model = system_config.get("model")
    if model not in ALLOWED_MODELS:
        raise ValueError(f"{name}: model {model!r} is not an allowed hackathon model")

    system: dict[str, Any] = {
        "name": name,
        "model": model,
        "adapter": None,
        "tool_patches": None,
        "tool_guards": bool(system_config.get("tool_guards", False)),
        "state_reminders": bool(system_config.get("state_reminders", False)),
        "condensed_policy": bool(system_config.get("condensed_policy", False)),
        "identity_note": bool(system_config.get("identity_note", False)),
        "agent_voice_check": bool(system_config.get("agent_voice_check", False)),
        "route": system_config.get("route", "none"),
        "targets": list(system_config.get("targets") or []),
    }
    if system["route"] not in ROUTES:
        raise ValueError(f"{name}: route must be one of {ROUTES}")

    adapter = system_config.get("adapter")
    if adapter:
        if not adapter.get("repo") or not adapter.get("revision"):
            raise ValueError(f"{name}: adapter needs repo and revision")
        system["adapter"] = {"repo": adapter["repo"], "revision": str(adapter["revision"])}

    tool_patches = system_config.get("tool_patches")
    if tool_patches:
        patch_ids = tool_patches.get("ids", "applied")
        ids_are_a_list_of_strings = isinstance(patch_ids, list) and all(
            isinstance(patch_id, str) for patch_id in patch_ids
        )
        if not (patch_ids == "applied" or ids_are_a_list_of_strings):
            raise ValueError(f"{name}: tool_patches.ids must be 'applied' or a list of ids")
        ledger_path = tool_patches.get("ledger") or "patches/ledger.jsonl"
        system["tool_patches"] = {"ledger": str(ledger_path), "ids": patch_ids}
        if root is not None and not (root / ledger_path).exists():
            raise ValueError(f"{name}: ledger {ledger_path} not found under {root}")
    return system


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
    from shipwright.heal.tool_fix import select_patches

    tool_patches = system.get("tool_patches")
    if not tool_patches:
        return []
    return [row["id"] for row in select_patches(ledger_rows, tool_patches["ids"])]


def baseline_args(system: dict, patch_ids: list[str], root: Path) -> list[str]:
    """Extra ``scripts/baseline.py`` arguments for this system's tool patches and guards."""
    extra_arguments = []
    tool_patches = system.get("tool_patches")
    if tool_patches:
        extra_arguments += ["--tool-patches", str(root / tool_patches["ledger"]), "--tool-patch-ids", *patch_ids]
    if system.get("tool_guards"):
        extra_arguments.append("--tool-guards")
    if system.get("state_reminders"):
        extra_arguments.append("--state-reminders")
    if system.get("condensed_policy"):
        extra_arguments.append("--condensed-policy")
    if system.get("identity_note"):
        extra_arguments.append("--identity-note")
    if system.get("agent_voice_check"):
        extra_arguments.append("--agent-voice-check")
    return extra_arguments


def serve_plan(systems: list[dict], mode: str = "lora") -> list[dict]:
    """vLLM servers to start, in order, and the systems each one serves.

    ``lora``: one server with the base model and every adapter as a LoRA module.
    ``merged``: one server per system; an adapter system serves its merged weights
    (fallback if vLLM cannot load the LoRA for this architecture).
    All systems must share one base model.
    """
    base_models = {system["model"] for system in systems}
    if len(base_models) != 1:
        raise ValueError(f"systems use different base models: {sorted(base_models)}")
    base_model = base_models.pop()

    if mode == "lora":
        adapter_by_served_name: dict[str, dict] = {}
        for system in systems:
            if system.get("adapter"):
                adapter_by_served_name[served_name(system)] = system["adapter"]
        return [{
            "base": base_model,
            "served_base": base_model,
            "loras": adapter_by_served_name,
            "merged": None,
            "systems": [system["name"] for system in systems],
        }]
    if mode == "merged":
        return [
            {
                "base": base_model,
                "served_base": served_name(system),
                "loras": {},
                "merged": system.get("adapter"),
                "systems": [system["name"]],
            }
            for system in systems
        ]
    raise ValueError(f"serve mode must be lora or merged, got {mode!r}")
