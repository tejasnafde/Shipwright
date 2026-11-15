"""The self-healing loop's one command (``shipwright`` once installed): see src/shipwright/loop/cli.py.

The Cloud Run image runs this file through scripts/job_entry.py (``job_entry.py loop.py run --config ... --here``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shipwright.loop.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
