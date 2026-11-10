"""Convert OpenTelemetry GenAI spans (JSONL) to trace records (the traces.jsonl shape).

Usage:
  .venv/bin/python scripts/otel_to_traces.py runs/<dir>/otel_spans.jsonl -o runs/<dir>/traces_from_otel.jsonl

The output feeds scripts/classify.py like a traces.jsonl from scripts/baseline.py.
Mapping: src/shipwright/observability/otel_import.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.observability.otel_import import read_spans, spans_to_traces  # noqa: E402
from shipwright.heal.traces import write_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spans", type=Path, help="span file, one JSON span per line")
    p.add_argument("-o", "--output", type=Path, required=True, help="traces.jsonl to write")
    args = p.parse_args(argv)
    traces = spans_to_traces(read_spans(args.spans))
    write_jsonl(args.output, traces)
    print(f"{len(traces)} trace records -> {args.output}")
    return 0 if traces else 1


if __name__ == "__main__":
    sys.exit(main())
