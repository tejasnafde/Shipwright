"""Generate verified synthetic tau2 retail training tasks (src/shipwright/heal/synthetic_tasks.py).

    python scripts/make_synthetic_tasks.py --count 600         # data/synthetic/retail_tasks.json + _report.json

Free and offline: no model is called. Every task is replayed on the retail database before it is kept, and nothing from
the held-out or hidden tasks may appear in it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shipwright.heal import synthetic_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--output", type=Path, default=synthetic_tasks.SYNTHETIC_PATH)
    options = parser.parse_args()
    tasks, report = synthetic_tasks.generate(options.count, options.seed)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(tasks, indent=1))
    options.output.with_name(options.output.stem + "_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
