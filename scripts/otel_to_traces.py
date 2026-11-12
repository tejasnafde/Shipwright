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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spans", type=Path, help="span file, one JSON span per line")
    parser.add_argument("-o", "--output", type=Path, required=True, help="traces.jsonl to write")
    options = parser.parse_args(argv)

    spans = read_spans(options.spans)
    traces = spans_to_traces(spans)
    write_jsonl(options.output, traces)
    print(f"{len(traces)} trace records -> {options.output}")
    if traces:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
