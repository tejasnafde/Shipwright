"""Launch the deploy gate (scripts/gate_eval.py --in-job) as an HF Job, by hand.

The loop (scripts/loop.py) launches the same job through shipwright.loop.jobs. Without --launch this prints the plan and
the cost estimate only: nothing is uploaded, nothing runs, nothing is billed. With --launch it uploads the code (git
archive HEAD, plus the two system configs, which need not be committed) to <runs-repo>/gate/round-N/code/ and submits
gate_eval.py with an explicit timeout.

Costs: the GPU (flavor price x run time, at most x timeout) and the user simulator calls on the HF router.

  .venv/bin/python scripts/gate_job.py --round 12 --current configs/systems/base.yaml \\
      --candidate configs/systems/r11-r7-fixes.yaml --user-model <USER_MODEL>   # plan only
  ... --launch                                                            # submit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright import cloud  # noqa: E402
from shipwright.llm.vertex import ROUTER_USER_MODEL  # noqa: E402
from shipwright.loop import jobs  # noqa: E402

# User simulator cost per tau2 run (USD), an estimate: about $0.0026 per run measured, doubled for long runs.
USER_COST_PER_RUN = {"hf_router": 0.005, "vertex": 0.0025}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--current", type=Path, required=True, help="system config, e.g. configs/systems/base.yaml")
    parser.add_argument("--candidate", type=Path, required=True, help="system config of the candidate")
    parser.add_argument("--user-model", required=True,
                        help=f"user simulator, no default (the loop uses its config's; the team's: {ROUTER_USER_MODEL})")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gate.yaml")
    parser.add_argument("--target-label", nargs="*", default=None, help="override the candidate's targets")
    parser.add_argument("--serve-mode", choices=("lora", "merged"), default=None)
    parser.add_argument("--train-eval", action="store_true",
                        help="no gate: only the candidate on --task-ids of the train split (the next round's failures)")
    parser.add_argument("--bfcl-screen", action="store_true", help="no gate: BFCL only, on the candidate")
    parser.add_argument("--task-ids", nargs="*", default=None, help="with --train-eval")
    parser.add_argument("--trials", type=int, default=1, help="with --train-eval")
    parser.add_argument("--flavor", default="a10g-large")
    parser.add_argument("--timeout", default="60m", help="job time limit; a gate needs about 40 minutes of running time")
    parser.add_argument("--price-per-hour", type=float, default=None, help="USD/h for a flavor not in cloud.GPU_PRICES")
    parser.add_argument("--runs-repo", default=cloud.RUNS_REPO)
    parser.add_argument("--namespace", default=None, help="HF namespace that runs (and pays for) the job")
    parser.add_argument("--launch", action="store_true", help="upload the code and submit the job (spends credits)")
    options = parser.parse_args(argv)
    if options.train_eval and not options.task_ids:
        parser.error("--train-eval needs --task-ids")
    return options


def job_from_options(options: argparse.Namespace) -> jobs.GateJob:
    mode = "gate"
    if options.train_eval:
        mode = "train_eval"
    if options.bfcl_screen:
        mode = "bfcl_screen"
    return jobs.GateJob(round_number=options.round, current=options.current.resolve(),
                        candidate=options.candidate.resolve(), user_model=options.user_model, flavor=options.flavor,
                        timeout=options.timeout, mode=mode, task_ids=list(options.task_ids or []), trials=options.trials,
                        runs_repo=options.runs_repo, config=options.config.resolve(),
                        target_labels=options.target_label, serve_mode=options.serve_mode, namespace=options.namespace)


def print_plan(job: jobs.GateJob, options: argparse.Namespace) -> bool:
    """The plan and the cost estimate. Returns whether the job needs a Vertex token."""
    from shipwright.heal.gate import load_config
    from shipwright.heal.systems import load_system, served_name
    from shipwright.llm import vertex

    gate_config = load_config(job.config)
    systems = {"current": load_system(job.current, ROOT), "candidate": load_system(job.candidate, ROOT)}
    if job.mode == "train_eval":
        total_runs = len(job.task_ids) * job.trials
    elif job.mode == "bfcl_screen":
        total_runs = 0
    else:
        total_runs = 2 * gate_config["heldout"]["num_tasks"] * gate_config["k"]
    user_on_vertex = vertex.is_vertex_model(job.user_model)
    classifier_model = (gate_config.get("classify") or {}).get("classifier_model") or ""
    classifier_on_vertex = job.mode == "gate" and vertex.is_vertex_model(classifier_model)
    if user_on_vertex:
        user_cost = total_runs * USER_COST_PER_RUN["vertex"]
    else:
        user_cost = total_runs * USER_COST_PER_RUN["hf_router"]

    print(f"{job.mode} round {job.round_number}  (gate config status: {gate_config['status']})")
    for role, system in systems.items():
        adapter = system.get("adapter")
        detail = f", adapter {adapter['repo']}@{adapter['revision']}" if adapter else ""
        print(f"  {role:9}  {system['name']} -> served as {served_name(system)}{detail}")
    print(f"targets:   {job.target_labels if job.target_labels is not None else systems['candidate']['targets']}")
    print(f"tau2 runs: {total_runs}, user simulator {job.user_model}, about ${user_cost:.2f}")
    if classifier_on_vertex:
        print(f"classify:  AgentTune FailureClassifier, {classifier_model} on Vertex, about $0.10")
    if user_on_vertex or classifier_on_vertex:
        print("           a Vertex token is minted now, valid about 1 h, with no refresh")
    price = cloud.gpu_price(job.flavor, options.price_per_hour)
    print(f"flavor:    {job.flavor} at ${price:.2f}/h, timeout {job.timeout}, max GPU cost "
          f"${cloud.worst_case_gpu_usd(job.flavor, job.timeout, options.price_per_hour):.2f}")
    print(f"results:   https://huggingface.co/datasets/{job.runs_repo}/tree/main/"
          f"{job.results_prefix(systems['candidate']['name'])}")
    return user_on_vertex or classifier_on_vertex


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    job = job_from_options(options)
    needs_vertex_token = print_plan(job, options)
    if not options.launch:
        print("\nPlan only. Nothing was uploaded or launched. Add --launch to submit the job.")
        return 0
    job_id = jobs.launch_gate(job, needs_vertex_token)
    print(f"\nLaunched job {job_id}\nLogs: hf jobs logs {job_id}    Stop: hf jobs cancel {job_id}")
    return 0


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="gate", meta=True))
