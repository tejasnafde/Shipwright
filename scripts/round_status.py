"""Should this round train? AgentTune's RetrainingTrigger decides (step 3 of the loop).

Usage:
  .venv/bin/python scripts/round_status.py runs/baseline-<timestamp> [--sft runs/sft-<ts>/sft.jsonl]
      [--reward-baseline runs/<earlier run of the same tasks>]

Reads failures.jsonl (from scripts/classify.py). The weight_fix failures go through AgentTune's
SelfHealLoop into a RetrainingTrigger buffer (agenttune_bridge.retrain_status). A failure has a
correction when --sft has a teacher example for the same task. Config: configs/heal.yaml.
Writes round_status.json into the run dir: should_train, AgentTune's reason, buffer health.
Offline: no model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.heal.agenttune_bridge import retrain_status  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def corrections(sft_path: Path | None) -> dict[str, list[dict]]:
    """Task id -> the messages of its first teacher SFT example."""
    messages_by_task_id: dict[str, list[dict]] = {}
    if sft_path is None:
        return messages_by_task_id
    for example in read_jsonl(sft_path):
        task_id = str(example.get("task_id"))
        if task_id not in messages_by_task_id:
            messages_by_task_id[task_id] = example["messages"]
    return messages_by_task_id


def rewards(run_directory: Path) -> list[float]:
    traces = read_jsonl(run_directory / "traces.jsonl")
    return [float(trace["reward"]) for trace in traces if trace.get("reward") is not None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--sft",
        type=Path,
        default=None,
        help="sft.jsonl from scripts/build_sft.py (teacher corrections)",
    )
    parser.add_argument(
        "--reward-baseline",
        type=Path,
        default=None,
        help="earlier run dir: its rewards lock AgentTune's reward-drift baseline",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "heal.yaml")
    options = parser.parse_args(argv)

    import yaml

    heal_config = yaml.safe_load(options.config.read_text())
    failure_rows = read_jsonl(options.run_dir / "failures.jsonl")

    if options.reward_baseline:
        baseline_rewards = rewards(options.reward_baseline)
        current_rewards = rewards(options.run_dir)
    else:
        baseline_rewards = None
        current_rewards = None

    status = retrain_status(
        failure_rows,
        corrections(options.sft),
        heal_config.get("retrain_trigger"),
        baseline_rewards=baseline_rewards,
        rewards=current_rewards,
    )
    status["run_dir"] = str(options.run_dir)
    status["sft"] = str(options.sft) if options.sft else None
    status["config_status"] = heal_config.get("status")
    (options.run_dir / "round_status.json").write_text(json.dumps(status, indent=2, default=str))

    printed_fields = ("should_train", "reason", "num_weight_fix_failures", "num_with_correction")
    print(json.dumps({field: status[field] for field in printed_fields}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
