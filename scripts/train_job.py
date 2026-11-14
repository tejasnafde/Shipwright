"""Launch scripts/train_lora.py as an HF Job (GPU, billed to the HF credits of the namespace).

Without --launch this only prints the plan and the cost estimate; nothing runs and nothing is billed. With --launch it
submits the job with an explicit timeout (HF Jobs stop after 30 minutes by default, which kills training). The loop
(scripts/loop.py) launches the same job through shipwright.loop.jobs; this is the by-hand way.

  .venv/bin/python scripts/train_job.py --dataset-revision round-1 --revision round-1          # plan only
  .venv/bin/python scripts/train_job.py --dataset-revision round-1 --revision round-1 --launch # submit

Arguments after "--" go to train_lora.py, e.g. "-- --epochs 2 --lr 1e-4".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright import cloud  # noqa: E402
from shipwright.loop import jobs  # noqa: E402


def split_training_arguments(command_line: list[str]) -> tuple[list[str], list[str]]:
    """Split at "--": (arguments for this script, arguments passed on to train_lora.py)."""
    if "--" not in command_line:
        return command_line, []
    separator_index = command_line.index("--")
    return command_line[:separator_index], command_line[separator_index + 1:]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    command_line = list(sys.argv[1:] if argv is None else argv)
    launch_arguments, training_arguments = split_training_arguments(command_line)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flavor", default="l40sx1", help="an L4 (24 GB) was too tight for 185 examples x 3 epochs")
    parser.add_argument("--timeout", default="45m", help="job time limit, e.g. 45m or 1h (never omit it)")
    parser.add_argument("--price-per-hour", type=float, default=None, help="USD/h for a flavor not in cloud.GPU_PRICES")
    parser.add_argument("--namespace", default=None, help="HF namespace that runs (and pays for) the job")
    parser.add_argument("--launch", action="store_true", help="really submit the job (spends credits)")
    parser.add_argument("--dataset", default=cloud.SFT_REPO)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--output-repo", default=cloud.LORA_REPO)
    parser.add_argument("--revision", required=True, help="branch of the adapter repo, e.g. round-12")
    return parser.parse_args(launch_arguments), training_arguments


def main(argv: list[str] | None = None) -> int:
    options, training_arguments = parse_args(argv)
    job = jobs.TrainJob(dataset_revision=options.dataset_revision, revision=options.revision, flavor=options.flavor,
                        timeout=options.timeout, dataset=options.dataset, output_repo=options.output_repo,
                        training_arguments=training_arguments, namespace=options.namespace)
    price = cloud.gpu_price(options.flavor, options.price_per_hour)
    print(f"script:    {jobs.TRAIN_SCRIPT.relative_to(ROOT)} {' '.join(job.script_arguments())}")
    print(f"flavor:    {options.flavor} at ${price:.2f}/h")
    print(f"timeout:   {options.timeout}")
    print(f"max cost:  ${cloud.worst_case_gpu_usd(options.flavor, options.timeout, options.price_per_hour):.2f}"
          " (price x timeout; billed only for the time the job runs)")
    print(f"namespace: {options.namespace or 'token owner'}")
    if not options.launch:
        print("\nPlan only. Nothing was launched. Add --launch to submit the job.")
        return 0
    job_id = jobs.launch_train(job)
    print(f"\nLaunched job {job_id}\nLogs: hf jobs logs {job_id}    Stop: hf jobs cancel {job_id}")
    return 0


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="train", meta=True))
