"""The round config (``configs/rounds/<name>.yaml``): everything a run of the loop needs, checked before any spend.

One file pins the round number, the owner, the seed, every model, trials, concurrency, the data pool, training, the
gate, the stop rules and a spend cap per stage and in total. ``validate`` lists every problem that would make the loop
spend money on a run that cannot work; ``load`` refuses a config with any.

Optional blocks:

    recipe:   what the first round starts with (shipwright.loop.recipe): tool_guards, general_replay, boost
    start:    continue from a gate that already refused a candidate, instead of a fresh round:
                from_round: 8                                # its gate_decision.json is read from the runs dataset
                candidate: configs/systems/r8-weight.yaml    # the refused system (its adapter is reused)
                regate: true                                 # optional: gate it again first, decide the remedy from that
    data.corrections_pool:  earlier corrections runs ("owner/dir") that join every round's corrections
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from shipwright import cloud

# Every stage the loop knows, in the order a round runs them. Which of them a round runs: controller.planned_stages.
STAGES = ["baseline", "candidate_train", "classify", "teacher", "corrections", "data", "train", "bfcl_screen", "gate"]
# Run by hand, one at a time (``shipwright stage ceiling``): never part of a round.
SIDE_STAGES = ["ceiling", "smoke"]
GPU_STAGES = {"candidate_train": "candidate_train", "train": "train", "bfcl_screen": "gate", "gate": "gate"}

REQUIRED_SECTIONS = {
    "round": int, "owner": str, "seed": int, "models": dict, "baseline": dict, "teacher": dict, "data": dict,
    "train": dict, "gate": dict, "stop": dict, "spend": dict, "frozen": dict,
}
REQUIRED_MODELS = ("agent", "user", "teacher", "classifier")
CANDIDATE_EXTRA_KEYS = {"state_reminders"}  # serving-only switches a round config may set on its candidate


def gpu_settings(config: dict, stage: str) -> dict:
    """flavors, timeout, api_usd of a GPU stage (the BFCL screen uses the gate's flavors with its own timeout)."""
    section = config[GPU_STAGES[stage]]
    if stage == "bfcl_screen":
        return {"flavors": section["flavors"], "timeout": section.get("screen_timeout", "30m"), "api_usd": 0.0}
    return {"flavors": section["flavors"], "timeout": section["timeout"], "api_usd": section.get("api_usd", 0.0)}


def worst_case_gpu_usd(settings: dict) -> float:
    """The most one launch can cost: the dearest flavor it may fall back to, for its whole timeout, plus its API calls."""
    dearest = max(cloud.GPU_PRICES[flavor] for flavor in settings["flavors"])
    return dearest * cloud.timeout_hours(settings["timeout"]) + settings.get("api_usd", 0.0)


def validate(config: dict) -> list[str]:
    """Problems that would make the loop spend money on a run that cannot work. An empty list means it is fine."""
    problems = []
    for key, expected_type in REQUIRED_SECTIONS.items():
        if not isinstance(config.get(key), expected_type):
            problems.append(f"{key}: missing or not a {expected_type.__name__}")
    if problems:
        return problems
    for model_role in REQUIRED_MODELS:
        if not config["models"].get(model_role):
            problems.append(f"models.{model_role}: missing")
    caps = config["spend"]
    for stage in [*STAGES, "total"]:
        if not isinstance(caps.get(stage), (int, float)):
            problems.append(f"spend.{stage}: missing")
    if problems:
        return problems
    for stage in GPU_STAGES:
        if GPU_STAGES[stage] not in config:
            problems.append(f"{GPU_STAGES[stage]}: missing (the {stage} stage needs flavors and a timeout)")
            continue
        settings = gpu_settings(config, stage)
        unknown = [flavor for flavor in settings["flavors"] if flavor not in cloud.GPU_PRICES]
        if unknown:
            problems.append(f"{GPU_STAGES[stage]}.flavors: no price for {unknown}")
            continue
        if worst_case_gpu_usd(settings) > caps[stage]:
            problems.append(f"{stage}: worst case ${worst_case_gpu_usd(settings):.2f} (price x timeout) passes its "
                            f"cap ${caps[stage]:.2f}")
    if config["gate"].get("timeout") and cloud.timeout_hours(config["gate"]["timeout"]) < 1:
        problems.append("gate.timeout: under 60m; a gate needs 45 to 50 minutes of wall time (AGENTS.md, gate 8)")
    for key in config["gate"].get("candidate_extra") or {}:
        if key not in CANDIDATE_EXTRA_KEYS:
            problems.append(f"gate.candidate_extra: {key} is not one of {sorted(CANDIDATE_EXTRA_KEYS)}")
    start = config.get("start")
    if start is not None:
        if not isinstance(start.get("from_round"), int) or not start.get("candidate"):
            problems.append("start: needs from_round (a number) and candidate (a system config path)")
        elif not (cloud.ROOT / start["candidate"]).exists():
            problems.append(f"start.candidate: {start['candidate']} does not exist")
    return problems


def load(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    problems = validate(config)
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    return config


def dry_run(config: dict) -> dict:
    """The same loop, offline: fake models (the scripts' own --dry-run), fake HF jobs, run dirs named dry-*, and stop rules
    the fake models can meet (they never pass a task, so there is no SFT data)."""
    dry = json.loads(json.dumps(config))
    dry["name_prefix"] = "dry-"
    dry["baseline"]["num_tasks"] = 2
    dry["teacher"]["trials"] = 1
    dry["data"]["pool"] = []
    dry["data"]["corrections_pool"] = []
    dry["stop"]["min_sft_examples"] = 0
    dry["train"]["require_trigger"] = False
    return dry


def run_name(config: dict, round_number: int, kind: str) -> str:
    """runs/ dir of one stage's output: r10-baseline, r10-cand-train, r10-teacher, r10-corrections, r10-sft, r10-round
    (the manifest) and gate-r10. A dry run's ``name_prefix`` keeps it apart from real rounds."""
    prefix = config.get("name_prefix", "")
    if kind == "gate":
        return f"{prefix}gate-r{round_number}"
    return f"{prefix}r{round_number}-{kind}"


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
