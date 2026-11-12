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


def parse_timeout_hours(timeout_text: str) -> float:
    """'2h', '90m', '3600s' or '3600' (seconds) -> hours."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", timeout_text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"bad timeout {timeout_text!r}")
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    hours_per_unit = {"s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}
    return amount * hours_per_unit[unit]


def estimate_cost(flavor: str, timeout: str, price_per_hour: float | None = None) -> tuple[float, float]:
    """(USD per hour, worst-case USD for the full timeout)."""
    if price_per_hour is not None:
        price = price_per_hour
    else:
        price = PRICES.get(flavor)
    if price is None:
        raise SystemExit(f"No known price for flavor {flavor!r}: pass --price-per-hour.")
    return price, round(price * parse_timeout_hours(timeout), 2)


def split_training_arguments(command_line: list[str]) -> tuple[list[str], list[str]]:
    """Split at "--": (arguments for this script, arguments passed on to train_lora.py)."""
    if "--" not in command_line:
        return command_line, []
    separator_index = command_line.index("--")
    return command_line[:separator_index], command_line[separator_index + 1:]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    if argv is None:
        command_line = list(sys.argv[1:])
    else:
        command_line = list(argv)
    launch_arguments, training_arguments = split_training_arguments(command_line)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # The HF Job.
    parser.add_argument("--flavor", default="l4x1", help="l4x1 (24 GB); fallback a10g-small (24 GB)")
    parser.add_argument("--timeout", default="2h", help="job time limit, e.g. 2h or 90m (never omit it)")
    parser.add_argument("--price-per-hour", type=float, default=None, help="USD/h for a flavor not in PRICES")
    parser.add_argument("--namespace", default=None, help="HF namespace that runs (and pays for) the job")
    parser.add_argument("--launch", action="store_true", help="really submit the job (spends credits)")

    # The data in and the adapter out.
    parser.add_argument("--dataset", default="mitushigoyal5/shipwright-sft")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--output-repo", default="mitushigoyal5/shipwright-lora")
    parser.add_argument("--revision", default="round-1", help="branch of the adapter repo")
    return parser.parse_args(launch_arguments), training_arguments


def main(argv: list[str] | None = None) -> int:
    options, extra_training_arguments = parse_args(argv)
    price_per_hour, worst_case_cost = estimate_cost(options.flavor, options.timeout, options.price_per_hour)
    training_script_arguments = [
        "--dataset", options.dataset,
        "--dataset-revision", options.dataset_revision,
        "--output-repo", options.output_repo,
        "--revision", options.revision,
        *extra_training_arguments,
    ]
    print(f"script:    {TRAIN_SCRIPT.relative_to(ROOT)} {' '.join(training_script_arguments)}")
    print(f"flavor:    {options.flavor} at ${price_per_hour:.2f}/h")
    print(f"timeout:   {options.timeout}")
    print(f"max cost:  ${worst_case_cost:.2f} (price x timeout; billed only for the time the job runs)")
    print(f"namespace: {options.namespace or 'token owner'}")
    if not options.launch:
        print("\nPlan only. Nothing was launched. Add --launch to submit the job.")
        return 0
    return launch_job(options, training_script_arguments)


def launch_job(options: argparse.Namespace, training_script_arguments: list[str]) -> int:
    import baseline
    from huggingface_hub import run_uv_job

    token = baseline.load_token()
    if not token:
        print("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.", file=sys.stderr)
        return 2
    from shipwright.observability import instrument, retry

    external_call_name = f"{options.flavor} {TRAIN_SCRIPT.name}"
    with instrument.external("hf_job_launch", external_call_name, timeout=options.timeout) as call_record:
        # retry.LAUNCH: retried only when the request surely did not start a (paid) job
        job = retry.call_with_retry(
            run_uv_job,
            str(TRAIN_SCRIPT),
            script_args=training_script_arguments,
            flavor=options.flavor,
            timeout=options.timeout,
            secrets={"HF_TOKEN": token},
            labels={"app": "lexsi-hackathon", "round": options.revision},
            namespace=options.namespace,
            token=token,
            policy=retry.LAUNCH,
            what="hf_job_launch",
        )
        call_record["job_id"] = job.id
    print(f"\nLaunched job {job.id}: {getattr(job, 'url', '')}")
    print(f"Logs: hf jobs logs {job.id}    Stop: hf jobs cancel {job.id}")
    return 0


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="train", meta=True))
