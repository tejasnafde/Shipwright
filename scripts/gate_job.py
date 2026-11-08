"""Launch the deploy gate (scripts/gate_eval.py --in-job) as an HF Job.

Without --launch this prints the plan and the cost estimate only: nothing is uploaded,
nothing runs, nothing is billed. With --launch it

1. packs the committed code (git archive HEAD: src, scripts, configs, patches, pyproject.toml)
   and uploads it to <runs-repo>/gate/round-N/code/ (the repo is private, so the job
   cannot clone it),
2. submits gate_eval.py with an explicit timeout (HF Jobs stop after 30 minutes by default).

Costs: the GPU (flavor price x run time, at most x timeout) and the user simulator calls on
the HF router (both from the team's HF credits).

  .venv/bin/python scripts/gate_job.py --round 1 --current configs/systems/base.yaml \\
      --candidate configs/systems/r1-tool.yaml --user-model <USER_MODEL>            # plan only
  ... --launch                                                                       # submit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_job import estimate_cost  # noqa: E402

GATE_SCRIPT = ROOT / "scripts" / "gate_eval.py"
RUNS_REPO = "mitushigoyal5/shipwright-runs"
BUNDLE_PATHS = ["src", "scripts", "configs", "patches", "pyproject.toml"]
# User simulator cost per tau2 run on the HF router (USD). The 20-task baseline spent about
# $0.0026 per run with gpt-oss-120b on cerebras; doubled for longer runs. Estimate only.
USER_COST_PER_RUN = 0.005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--current", type=Path, required=True, help="system config, e.g. configs/systems/base.yaml")
    p.add_argument("--candidate", type=Path, required=True, help="system config of the candidate")
    p.add_argument("--user-model", required=True,
                   help="user simulator, litellm id via the HF router (no default: the team has not chosen it)")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "gate.yaml")
    p.add_argument("--target-label", nargs="*", default=None, help="override the candidate's targets")
    p.add_argument("--flavor", default="a100-large")
    p.add_argument("--timeout", default="2h", help="job time limit, e.g. 2h or 90m (never omit it)")
    p.add_argument("--price-per-hour", type=float, default=None, help="USD/h for a flavor not in PRICES")
    p.add_argument("--runs-repo", default=RUNS_REPO)
    p.add_argument("--serve-mode", choices=("lora", "merged"), default=None)
    p.add_argument("--namespace", default=None, help="HF namespace that runs (and pays for) the job")
    p.add_argument("--launch", action="store_true", help="upload the code and submit the job (spends credits)")
    return p.parse_args(argv)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def script_args(args: argparse.Namespace, bundle: str) -> list[str]:
    out = ["--in-job", "--round", str(args.round), "--config", rel(args.config),
           "--current", rel(args.current), "--candidate", rel(args.candidate),
           "--user-model", args.user_model, "--runs-repo", args.runs_repo, "--code-bundle", bundle]
    if args.target_label is not None:
        out += ["--target-label", *args.target_label]
    if args.serve_mode:
        out += ["--serve-mode", args.serve_mode]
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from shipwright.gate import load_config
    from shipwright.systems import load_system, served_name

    cfg = load_config(args.config)
    systems = {who: load_system(p, ROOT) for who, p in (("current", args.current), ("candidate", args.candidate))}
    price, worst = estimate_cost(args.flavor, args.timeout, args.price_per_hour)
    held = cfg["heldout"]
    runs = 2 * held["num_tasks"] * cfg["k"]
    sha = git("rev-parse", "--short", "HEAD")
    bundle = f"gate/round-{args.round}/code/shipwright-{sha}.tar.gz"
    targets = args.target_label if args.target_label is not None else systems["candidate"]["targets"]

    print(f"gate round {args.round}  (config status: {cfg['status']})")
    for who, s in systems.items():
        extra = []
        if s["adapter"]:
            extra.append(f"adapter {s['adapter']['repo']}@{s['adapter']['revision']}")
        if s["tool_patches"]:
            extra.append(f"patches {s['tool_patches']['ids'] if isinstance(s['tool_patches']['ids'], str) else len(s['tool_patches']['ids'])}")
        print(f"  {who:9}  {s['name']} -> served as {served_name(s)} {', '.join(extra)}")
    print(f"targets:   {targets or 'NONE (the gate will reject)'}")
    print(f"held-out:  {held['domain']} {held['task_split']}, {held['num_tasks']} tasks x k={cfg['k']}"
          f" x 2 systems = {runs} tau2 runs")
    print(f"user sim:  {args.user_model} via the HF router, about ${runs * USER_COST_PER_RUN:.2f}"
          f" (${USER_COST_PER_RUN}/run estimate)")
    if (cfg.get("bfcl") or {}).get("enabled"):
        b = cfg["bfcl"]
        print(f"BFCL:      {', '.join(b['categories'])}, up to {b.get('limit_per_category')} each, on the GPU")
    print(f"script:    {rel(GATE_SCRIPT)} {' '.join(script_args(args, bundle))}")
    print(f"flavor:    {args.flavor} at ${price:.2f}/h")
    print(f"timeout:   {args.timeout}")
    print(f"max cost:  ${worst:.2f} GPU (price x timeout; billed only for the time the job runs)"
          f" + about ${runs * USER_COST_PER_RUN:.2f} user simulator")
    print(f"results:   https://huggingface.co/datasets/{args.runs_repo}/tree/main/gate/round-{args.round}/"
          f"{systems['candidate']['name']}")
    print(f"code:      git HEAD {sha} -> {bundle}")
    print(f"namespace: {args.namespace or 'token owner'}")
    dirty = git("status", "--porcelain", "--", *BUNDLE_PATHS)
    if dirty:
        print("\nwarning: uncommitted changes in the bundle paths are NOT included (git archive HEAD)")
    if not args.launch:
        print("\nPlan only. Nothing was uploaded or launched. Add --launch to submit the job.")
        return 0

    import baseline
    from huggingface_hub import HfApi, run_uv_job

    token = baseline.load_token()
    if not token:
        print("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    api.create_repo(args.runs_repo, repo_type="dataset", exist_ok=True, private=False)
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "code.tar.gz"
        subprocess.check_call(["git", "-C", str(ROOT), "archive", "--format=tar.gz", "-o", str(tar), "HEAD",
                               "--", *BUNDLE_PATHS])
        api.upload_file(path_or_fileobj=str(tar), path_in_repo=bundle, repo_id=args.runs_repo,
                        repo_type="dataset", commit_message=f"gate round {args.round}: code {sha}")
    job = run_uv_job(
        str(GATE_SCRIPT),
        script_args=script_args(args, bundle),
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": token},
        labels={"app": "lexsi-hackathon", "round": f"round-{args.round}", "step": "gate"},
        namespace=args.namespace,
        token=token,
    )
    print(f"\nLaunched job {job.id}: {getattr(job, 'url', '')}")
    print(f"Logs: hf jobs logs {job.id}    Stop: hf jobs cancel {job.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
