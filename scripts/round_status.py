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


def corrections(sft: Path | None) -> dict[str, list[dict]]:
    """Task id -> the messages of its first teacher SFT example."""
    out: dict[str, list[dict]] = {}
    for r in read_jsonl(sft) if sft else []:
        out.setdefault(str(r.get("task_id")), r["messages"])
    return out


def rewards(run_dir: Path) -> list[float]:
    return [float(t["reward"]) for t in read_jsonl(run_dir / "traces.jsonl") if t.get("reward") is not None]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--sft", type=Path, default=None, help="sft.jsonl from scripts/build_sft.py (teacher corrections)")
    p.add_argument("--reward-baseline", type=Path, default=None,
                   help="earlier run dir: its rewards lock AgentTune's reward-drift baseline")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "heal.yaml")
    args = p.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    rows = read_jsonl(args.run_dir / "failures.jsonl")
    status = retrain_status(
        rows, corrections(args.sft), cfg.get("retrain_trigger"),
        baseline_rewards=rewards(args.reward_baseline) if args.reward_baseline else None,
        rewards=rewards(args.run_dir) if args.reward_baseline else None)
    status.update({"run_dir": str(args.run_dir), "sft": str(args.sft) if args.sft else None,
                   "config_status": cfg.get("status")})
    (args.run_dir / "round_status.json").write_text(json.dumps(status, indent=2, default=str))
    print(json.dumps({k: status[k] for k in ("should_train", "reason", "num_weight_fix_failures",
                                             "num_with_correction")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
