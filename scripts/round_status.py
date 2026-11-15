"""Should this round train? AgentTune's RetrainingTrigger decides, by hand (the loop's train stage asks the same).

    .venv/bin/python scripts/round_status.py runs/<classified run> [--sft runs/rN-sft/sft.jsonl]
        [--reward-baseline runs/<earlier run of the same tasks>]

Reads failures.jsonl (scripts/classify.py). The weight_fix failures go through AgentTune's SelfHealLoop into a
RetrainingTrigger buffer; a failure has a correction when --sft holds an example of its task. Config: configs/heal.yaml.
Writes round_status.json into the run dir. Offline: no model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.heal.agenttune_bridge import retrain_status_for_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sft", type=Path, default=None, help="sft.jsonl (scripts/build_sft.py)")
    parser.add_argument("--reward-baseline", type=Path, default=None,
                        help="an earlier run dir of the same tasks: its rewards lock AgentTune's reward-drift baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "heal.yaml")
    options = parser.parse_args(argv)
    status = retrain_status_for_run(options.run_dir, options.sft, options.config, options.reward_baseline)
    printed_fields = ("should_train", "reason", "num_weight_fix_failures", "num_with_correction")
    print(json.dumps({field: status[field] for field in printed_fields}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
