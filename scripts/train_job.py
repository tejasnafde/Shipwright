"""Launch scripts/train_lora.py as an HF Job (GPU, billed to the HF credits of the namespace).

Without --launch this only prints the plan and the cost estimate; nothing runs and
nothing is billed. With --launch it submits the job with an explicit timeout (HF Jobs
stop after 30 minutes by default, which kills training).

  .venv/bin/python scripts/train_job.py --dataset-revision round-1 --revision round-1          # plan only
  .venv/bin/python scripts/train_job.py --dataset-revision round-1 --revision round-1 --launch # submit

Arguments after "--" go to train_lora.py, e.g. "-- --epochs 2 --lr 1e-4".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

TRAIN_SCRIPT = ROOT / "scripts" / "train_lora.py"
# USD per hour (AGENTS.md, 2026-11-08). Other flavors need --price-per-hour.
PRICES = {"t4-small": 0.40, "l4x1": 0.80, "a10g-small": 1.00, "a10g-large": 1.50, "l40sx1": 1.80,
          "a100-large": 2.50, "rtx-pro-6000": 2.75}


def parse_timeout_hours(value: str) -> float:
    """'2h', '90m', '3600s' or '3600' (seconds) -> hours."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", value.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad timeout {value!r}")
    n, unit = float(m.group(1)), m.group(2) or "s"
    return n * {"s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}[unit]


def estimate_cost(flavor: str, timeout: str, price_per_hour: float | None = None) -> tuple[float, float]:
    """(USD per hour, worst-case USD for the full timeout)."""
    price = price_per_hour if price_per_hour is not None else PRICES.get(flavor)
    if price is None:
        raise SystemExit(f"No known price for flavor {flavor!r}: pass --price-per-hour.")
    return price, round(price * parse_timeout_hours(timeout), 2)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flavor", default="l4x1", help="l4x1 (24 GB); fallback a10g-small (24 GB)")
    p.add_argument("--timeout", default="2h", help="job time limit, e.g. 2h or 90m (never omit it)")
    p.add_argument("--price-per-hour", type=float, default=None, help="USD/h for a flavor not in PRICES")
    p.add_argument("--dataset", default="mitushigoyal5/shipwright-sft")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--output-repo", default="mitushigoyal5/shipwright-lora")
    p.add_argument("--revision", default="round-1", help="branch of the adapter repo")
    p.add_argument("--namespace", default=None, help="HF namespace that runs (and pays for) the job")
    p.add_argument("--launch", action="store_true", help="really submit the job (spends credits)")
    return p.parse_args(argv), extra


def main(argv: list[str] | None = None) -> int:
    args, extra = parse_args(argv)
    price, worst = estimate_cost(args.flavor, args.timeout, args.price_per_hour)
    script_args = [
        "--dataset", args.dataset, "--dataset-revision", args.dataset_revision,
        "--output-repo", args.output_repo, "--revision", args.revision, *extra,
    ]
    print(f"script:    {TRAIN_SCRIPT.relative_to(ROOT)} {' '.join(script_args)}")
    print(f"flavor:    {args.flavor} at ${price:.2f}/h")
    print(f"timeout:   {args.timeout}")
    print(f"max cost:  ${worst:.2f} (price x timeout; billed only for the time the job runs)")
    print(f"namespace: {args.namespace or 'token owner'}")
    if not args.launch:
        print("\nPlan only. Nothing was launched. Add --launch to submit the job.")
        return 0

    import baseline
    from huggingface_hub import run_uv_job

    token = baseline.load_token()
    if not token:
        print("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.", file=sys.stderr)
        return 2
    job = run_uv_job(
        str(TRAIN_SCRIPT),
        script_args=script_args,
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": token},
        labels={"app": "lexsi-hackathon", "round": args.revision},
        namespace=args.namespace,
        token=token,
    )
    print(f"\nLaunched job {job.id}: {getattr(job, 'url', '')}")
    print(f"Logs: hf jobs logs {job.id}    Stop: hf jobs cancel {job.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
