"""Copy run dirs from the team's GCS bucket into runs/, so control-room (and why.py) can read them.

  .venv/bin/python scripts/pull_runs.py --owner tejas                      # every run dir not here yet
  .venv/bin/python scripts/pull_runs.py --owner tejas --run r1-teacher-t1  # just that one
  .venv/bin/python scripts/pull_runs.py --owner all                        # tejas, ishank, mitushi, main

Only reads from the bucket (no cost). A run dir that already exists in runs/ is skipped, never
overwritten: delete it first to fetch it again. gcloud config: $SHIPWRIGHT_GCLOUD_CONFIG, else
`lexsi` if you have it, else `personal`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.control_room import gcs_runs  # noqa: E402
from shipwright.observability import log  # noqa: E402

logger = log.get_logger("pull_runs")


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="tejas, ishank, mitushi, main, or all")
    parser.add_argument("--run", action="append", default=[], help="a run dir name to pull (repeatable); default: all")
    return parser.parse_args()


def main() -> int:
    options = parse_options()
    owners = list(gcs_runs.OWNERS) if options.owner == "all" else [options.owner]
    results = gcs_runs.pull_runs(owners, ROOT / "runs", options.run or None)

    for result in results:
        logger.info("run dir", owner=result.owner, run=result.run_name, status=result.status, detail=result.detail)
    pulled = [result for result in results if result.status == "pulled"]
    failed = [result for result in results if result.status == "failed"]
    print(f"pulled {len(pulled)}, already local {len(results) - len(pulled) - len(failed)}, failed {len(failed)}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="pull_runs", meta=True))
