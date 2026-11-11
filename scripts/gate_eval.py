# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub>=1.0",
# ]
# ///
"""Deploy gate: evaluate current vs candidate on the held-out split, decide, publish.

Three modes:

  --in-job           Inside an HF Job (launched by scripts/gate_job.py). Downloads the code
                     bundle, installs the harness (scripts/setup_tau2.sh), starts vLLM with
                     the base model and the candidate LoRA, runs tau2 for both systems with
                     k trials (agent: http://localhost:8000/v1, user simulator: HF router or Vertex),
                     runs BFCL, classifies failures, computes metrics, decides and pushes
                     everything to <runs-repo>/gate/round-N/<candidate>/ plus one row to
                     the dataset's metrics.jsonl.
  --local-dry-run    Offline: fake LLM (scripts/baseline.py --dry-run), fixture BFCL data,
                     a fake BFCL model. No vLLM, no network, no token. Exercises the pipeline.
  --current-run DIR --candidate-run DIR
                     Offline: decide from two run dirs that already have failures.jsonl.

Outputs (output dir): <system>/ run dirs (traces.jsonl, failures.jsonl, bfcl.json, ...),
metrics_current.json, metrics_candidate.json, gate_decision.json, gate_inputs.json and
metrics.jsonl (one row per gate run).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS_REPO = "mitushigoyal5/shipwright-runs"
VLLM_PORT = 8000
API_BASE = f"http://localhost:{VLLM_PORT}/v1"
JOB_WORKDIR = Path("/tmp/shipwright")
DRY_BFCL = ROOT / "tests" / "fixtures" / "bfcl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, required=True, help="loop round number (results go to gate/round-N/<candidate>/)")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "gate.yaml")
    p.add_argument("--current", type=Path, default=None, help="system config of the current system")
    p.add_argument("--candidate", type=Path, default=None, help="system config of the candidate")
    p.add_argument("--user-model", default=None,
                   help="user simulator: HF router litellm id or vertex/<publisher>/<model> (required for"
                        " --in-job; no default). In a job, vertex/ uses $SHIPWRIGHT_VERTEX_TOKEN (set by gate_job.py)")
    p.add_argument("--target-label", nargs="*", default=None, help="override the candidate's targets")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--runs-repo", default=RUNS_REPO)
    p.add_argument("--no-push", action="store_true", help="in a job: keep results local (debug)")
    p.add_argument("--serve-mode", choices=("lora", "merged"), default=None, help="override serve.mode")
    p.add_argument("--no-merge-fallback", action="store_true",
                   help="do not fall back to a merged adapter if vLLM fails to start with LoRA")
    p.add_argument("--num-tasks", type=int, default=None, help="override heldout.num_tasks (debug only)")
    # modes
    p.add_argument("--in-job", action="store_true")
    p.add_argument("--code-bundle", default=None, help="path in --runs-repo of the code tarball (job mode)")
    p.add_argument("--bootstrapped", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--local-dry-run", action="store_true")
    p.add_argument("--current-run", type=Path, default=None)
    p.add_argument("--candidate-run", type=Path, default=None)
    args = p.parse_args(argv)
    if bool(args.current_run) != bool(args.candidate_run):
        p.error("--current-run and --candidate-run go together")
    decide_only = bool(args.current_run)
    if sum([args.in_job, args.local_dry_run, decide_only]) != 1:
        p.error("choose one mode: --in-job, --local-dry-run, or --current-run/--candidate-run")
    if not decide_only and not (args.current and args.candidate):
        p.error("--current and --candidate system configs are required")
    if args.in_job and not args.user_model:
        p.error("--user-model is required (no default)")
    # Config paths are repo-relative (the job re-runs this script from the unpacked bundle).
    for key in ("config", "current", "candidate"):
        path = getattr(args, key)
        if path is not None and not path.is_absolute():
            setattr(args, key, ROOT / path)
    return args


# ---------------------------------------------------------------- shared steps

def run(cmd: list[str], log: Path | None = None, cwd: Path | None = None, env: dict | None = None) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if log is None:
        return subprocess.call(cmd, cwd=cwd, env=env)
    with open(log, "w") as f:
        return subprocess.call(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)


def heldout_task_ids(cfg: dict, num_tasks: int | None = None) -> list[str]:
    """The held-out ids of configs/splits.yaml (never the hidden ones); --num-tasks takes the first N."""
    from shipwright.heal import splits

    ids = splits.heldout_ids()
    if len(ids) != cfg["heldout"]["num_tasks"]:
        raise SystemExit(f"configs/gate.yaml heldout.num_tasks is {cfg['heldout']['num_tasks']}, "
                         f"configs/splits.yaml has {len(ids)} held-out ids")
    return ids[:num_tasks] if num_tasks is not None else ids


def baseline_cmd(cfg: dict, system: dict, patch_ids: list[str], out: Path, args: argparse.Namespace) -> list[str]:
    from shipwright.heal.systems import agent_model_id, baseline_args

    held = cfg["heldout"]
    cmd = [
        sys.executable, str(ROOT / "scripts" / "baseline.py"),
        "--domain", held["domain"], "--task-split", held["task_split"],
        "--task-ids", *heldout_task_ids(cfg, args.num_tasks),
        "--trials", str(cfg["k"]), "--seed", str(cfg["seed"]),
        "--agent-model", agent_model_id(system),
        "--agent-temperature", str(cfg.get("agent_temperature", 0.0)),
        "--max-steps", str(cfg.get("max_steps", 100)),
        "--max-concurrency", str(cfg.get("max_concurrency", 4)),
        "--output-dir", str(out),
        *baseline_args(system, patch_ids, ROOT),
    ]
    if args.local_dry_run:
        cmd.append("--dry-run")
    else:
        cmd += ["--agent-api-base", API_BASE, "--user-model", args.user_model]
    return cmd


def classify_cmd(cfg: dict, run_dir: Path) -> list[str]:
    model = (cfg.get("classify") or {}).get("classifier_model")
    cmd = [sys.executable, str(ROOT / "scripts" / "classify.py"), str(run_dir)]
    return cmd + (["--classifier-model", model] if model else ["--no-llm"])


def run_bfcl(cfg: dict, system: dict, out: Path, args: argparse.Namespace) -> dict | None:
    from shipwright.heal import bfcl_eval
    from shipwright.heal.systems import served_name

    bcfg = cfg.get("bfcl") or {}
    if not bcfg.get("enabled"):
        return None
    if args.local_dry_run:
        root, revision = DRY_BFCL, "fixture"

        def chat(messages: list[dict]) -> str:
            return "Sorry, I cannot help with that."  # fake model: never calls a tool
    else:
        root, revision = bfcl_eval.download(bcfg["revision"], bcfg["categories"], os.environ.get("HF_TOKEN")), bcfg["revision"]
        chat = bfcl_eval.make_chat(API_BASE, served_name(system))
    samples = bfcl_eval.load_samples(root, bcfg["categories"], revision, bcfg.get("limit_per_category"))
    result = bfcl_eval.evaluate(samples, chat)
    result["revision"] = revision
    (out / "bfcl.json").write_text(json.dumps(result, indent=1))
    print(f"BFCL {system['name']}: {result['accuracy_points']} points on {result['num_samples']} samples")
    return {k: v for k, v in result.items() if k != "rows"}


def metrics_for(run_dir: Path, cfg: dict, name: str, bfcl: dict | None) -> dict | None:
    from shipwright.heal.eval_metrics import compute_metrics, load_run

    try:
        m = compute_metrics(load_run(run_dir), cfg["k"], system=name)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"metrics for {name} failed: {exc}", file=sys.stderr)
        return None
    if bfcl is None and (run_dir / "bfcl.json").exists():
        bfcl = {k: v for k, v in json.loads((run_dir / "bfcl.json").read_text()).items() if k != "rows"}
    m["bfcl"] = bfcl
    return m


def missing_metrics(name: str, reason: str) -> dict:
    """Metrics for a system whose run failed: every check on it fails, so the gate rejects."""
    return {"system": name, "error": reason, "num_valid_runs": 0, "num_tasks_complete": 0,
            "pass_1": None, "pass_k": None, "task_ids": [], "failed_run_labels": []}


def decide_and_write(out: Path, cfg: dict, round_: int, metrics: dict[str, dict], systems: dict,
                     targets: list[str]) -> dict:
    from shipwright.heal.gate import decide, metrics_row

    for who in ("current", "candidate"):
        (out / f"metrics_{who}.json").write_text(json.dumps(metrics[who], indent=1, default=str))
    decision = decide(metrics["current"], metrics["candidate"], cfg, targets)
    decision["round"] = round_
    decision["systems"] = systems
    (out / "gate_decision.json").write_text(json.dumps(decision, indent=1, default=str))
    row = metrics_row(round_, decision, systems)
    with open(out / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(f"\nGATE round {round_}: {decision['decision'].upper()}")
    for c in decision["checks"]:
        print(f"  [{'ok' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    for w in decision["warnings"]:
        print(f"  warning: {w}")
    return decision


def load_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    """Gate config, the two systems and their resolved patch ids."""
    from shipwright.heal.gate import load_config
    from shipwright.heal.systems import load_system, resolve_patch_ids
    from shipwright.heal.tool_fix import load_ledger

    cfg = load_config(args.config)
    if args.serve_mode:
        cfg.setdefault("serve", {})["mode"] = args.serve_mode
    systems = {"current": load_system(args.current, ROOT), "candidate": load_system(args.candidate, ROOT)}
    if systems["current"]["name"] == systems["candidate"]["name"]:
        raise SystemExit("current and candidate need different names")
    patch_ids = {}
    for who, s in systems.items():
        rows = load_ledger(ROOT / s["tool_patches"]["ledger"]) if s["tool_patches"] else []
        patch_ids[who] = resolve_patch_ids(s, rows)
    return cfg, systems, patch_ids, {who: {**s, "resolved_patch_ids": patch_ids[who]} for who, s in systems.items()}


def default_out(round_: int, tag: str) -> Path:
    return ROOT / "runs" / f"gate-round-{round_}-{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


# ---------------------------------------------------------------- vLLM (job mode)

def download_adapter(adapter: dict, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(adapter["repo"], revision=adapter["revision"], local_dir=str(dest),
                                  token=os.environ.get("HF_TOKEN")))


MERGE_CODE = """
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, adapter, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged", out)
"""


def merge_adapter(base: str, adapter_dir: Path, out: Path, vllm_spec: str, log: Path) -> Path:
    """peft merge_and_unload in vLLM's own env (same torch and transformers)."""
    cmd = ["uv", "run", "--no-project", "--python", "3.12", "--with", vllm_spec, "--with", "peft",
           "python", "-c", MERGE_CODE, base, str(adapter_dir), str(out)]
    if run(cmd, log=log) != 0:
        raise RuntimeError(f"adapter merge failed, see {log}")
    return out


def vllm_cmd(server: dict, scfg: dict, lora_paths: dict[str, Path], model_path: str) -> list[str]:
    cmd = ["uv", "run", "--no-project", "--python", "3.12", "--with", scfg["vllm"],
           "vllm", "serve", model_path, "--served-model-name", server["served_base"],
           "--port", str(VLLM_PORT), "--dtype", "bfloat16",
           "--max-model-len", str(scfg.get("max_model_len", 8192)),
           "--gpu-memory-utilization", str(scfg.get("gpu_memory_utilization", 0.85))]
    if lora_paths:
        cmd += ["--enable-lora", "--max-lora-rank", str(scfg.get("max_lora_rank", 16)),
                "--max-loras", str(len(lora_paths)), "--lora-modules",
                *[f"{name}={path}" for name, path in lora_paths.items()]]
    return cmd


class VLLMServer:
    def __init__(self, cmd: list[str], expect: list[str], log: Path, timeout_s: int):
        self.cmd, self.expect, self.log, self.timeout_s = cmd, expect, log, timeout_s
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "VLLMServer":
        import httpx

        print("+ " + " ".join(self.cmd), flush=True)
        self._f = open(self.log, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self._f, stderr=subprocess.STDOUT)
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.close()
                raise RuntimeError(f"vLLM exited with {self.proc.returncode}, see {self.log}")
            try:
                r = httpx.get(f"{API_BASE}/models", timeout=5)
                names = {m["id"] for m in r.json().get("data", [])}
                if r.status_code == 200 and set(self.expect) <= names:
                    print(f"vLLM ready: {sorted(names)}", flush=True)
                    return self
            except Exception:  # noqa: BLE001 - not up yet
                pass
            time.sleep(10)
        self.close()
        raise RuntimeError(f"vLLM not ready after {self.timeout_s}s, see {self.log}")

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._f.close()

    def __exit__(self, *exc) -> None:
        self.close()


def serve_and_run(cfg: dict, systems: dict, patch_ids: dict, out: Path, args: argparse.Namespace) -> dict:
    """Start vLLM per the serve plan, run tau2 and BFCL for each system. Returns status per system."""
    from shipwright.heal.systems import serve_plan, served_name

    scfg = cfg.get("serve") or {}
    mode = scfg.get("mode", "lora")
    by_name = {s["name"]: (who, s) for who, s in systems.items()}
    status: dict[str, dict] = {}
    adapters_dir = Path("/tmp/adapters")

    def run_plan(plan: list[dict]) -> None:
        for i, server in enumerate(plan):
            lora_paths = {n: download_adapter(a, adapters_dir / n) for n, a in server["loras"].items()}
            model_path = server["base"]
            if server["merged"]:
                adapter_dir = download_adapter(server["merged"], adapters_dir / server["served_base"])
                model_path = str(merge_adapter(server["base"], adapter_dir, Path(f"/tmp/merged/{server['served_base']}"),
                                               scfg["vllm"], out / f"merge-{i}.log"))
            expect = [server["served_base"], *lora_paths]
            with VLLMServer(vllm_cmd(server, scfg, lora_paths, model_path), expect,
                            out / f"vllm-{i}.log", int(scfg.get("startup_timeout_s", 1200))):
                for name in server["systems"]:
                    who, system = by_name[name]
                    run_dir = out / name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    code = run(baseline_cmd(cfg, system, patch_ids[who], run_dir, args), log=run_dir / "baseline.log")
                    bfcl = None
                    try:
                        bfcl = run_bfcl(cfg, system, run_dir, args)
                    except Exception as exc:  # noqa: BLE001 - recorded; the gate rejects without BFCL
                        print(f"BFCL failed for {name}: {exc}", file=sys.stderr)
                    status[name] = {"baseline_exit": code, "bfcl": bfcl, "served_as": served_name(system)}

    try:
        run_plan(serve_plan(list(systems.values()), mode))
        status["_serve_mode"] = {"mode": mode}
    except RuntimeError as exc:
        has_lora = any(s.get("adapter") for s in systems.values())
        if mode != "lora" or not has_lora or args.no_merge_fallback or status:
            raise
        print(f"vLLM with LoRA failed ({exc}); falling back to merged adapters", file=sys.stderr)
        run_plan(serve_plan(list(systems.values()), "merged"))
        status["_serve_mode"] = {"mode": "merged", "lora_error": str(exc)[:500]}
    return status


# ---------------------------------------------------------------- push

def push(out: Path, args: argparse.Namespace, row_file: Path, candidate: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.runs_repo, repo_type="dataset", exist_ok=True, private=False)
    prefix = f"gate/round-{args.round}/{candidate}"
    from shipwright.observability import instrument

    with instrument.external("hf_hub_upload", f"datasets/{args.runs_repo}/{prefix}"):
        api.upload_folder(folder_path=str(out), repo_id=args.runs_repo, repo_type="dataset", path_in_repo=prefix,
                          commit_message=f"gate round {args.round}: results",
                          ignore_patterns=["**/tau2/**"])  # tau2's own results.json duplicates traces.jsonl
    # Append this run's row to the dataset-level metrics.jsonl.
    try:
        old = Path(hf_hub_download(args.runs_repo, "metrics.jsonl", repo_type="dataset", token=token)).read_text()
    except (EntryNotFoundError, RepositoryNotFoundError):
        old = ""
    merged = out / "_metrics_all.jsonl"
    merged.write_text(old + ("" if not old or old.endswith("\n") else "\n") + row_file.read_text())
    with instrument.external("hf_hub_upload", f"datasets/{args.runs_repo}/metrics.jsonl"):
        api.upload_file(path_or_fileobj=str(merged), path_in_repo="metrics.jsonl", repo_id=args.runs_repo,
                        repo_type="dataset", commit_message=f"metrics.jsonl: gate round {args.round}")
    print(f"pushed to https://huggingface.co/datasets/{args.runs_repo}/tree/main/{prefix}")


# ---------------------------------------------------------------- modes

def bootstrap(argv: list[str], args: argparse.Namespace) -> None:
    """Job mode, first pass: fetch the code bundle, install the harness, re-run in its venv."""
    from huggingface_hub import hf_hub_download

    if not args.code_bundle:
        raise SystemExit("--code-bundle is required in job mode")
    tar = hf_hub_download(args.runs_repo, args.code_bundle, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    if JOB_WORKDIR.exists():
        shutil.rmtree(JOB_WORKDIR)
    JOB_WORKDIR.mkdir(parents=True)
    with tarfile.open(tar) as t:
        t.extractall(JOB_WORKDIR, filter="data")
    if run(["bash", str(JOB_WORKDIR / "scripts" / "setup_tau2.sh")]) != 0:
        raise SystemExit("harness setup failed")
    python = str(JOB_WORKDIR / ".venv" / "bin" / "python")
    os.execv(python, [python, str(JOB_WORKDIR / "scripts" / "gate_eval.py"), *argv, "--bootstrapped"])


def _expected_reject() -> None:
    try:
        from shipwright.observability import log
    except ImportError:
        return
    log.expected_exit(1, "the gate rejected the candidate")


def _attach(out: Path) -> None:
    """log.jsonl, errors.json and repro.json in the gate output dir (when run through log.run_script)."""
    try:
        from shipwright.observability import log
    except ImportError:
        return
    log.attach_run_dir(out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.in_job and not args.bootstrapped:
        bootstrap(argv, args)  # does not return

    from shipwright.heal.gate import load_config

    if args.current_run:  # decide only
        cfg = load_config(args.config)
        out = (args.output_dir or default_out(args.round, "decide")).resolve()
        out.mkdir(parents=True, exist_ok=True)
        _attach(out)
        names = {"current": args.current_run.name, "candidate": args.candidate_run.name}
        targets = args.target_label or []
        metrics = {who: metrics_for(d, cfg, names[who], None) or missing_metrics(names[who], "no metrics")
                   for who, d in (("current", args.current_run), ("candidate", args.candidate_run))}
        decision = decide_and_write(out, cfg, args.round, metrics,
                                    {who: {"run_dir": str(d)} for who, d in
                                     (("current", args.current_run), ("candidate", args.candidate_run))}, targets)
        print(f"\nWrote {out}")
        if decision["decision"] != "promote":
            _expected_reject()
        return 0 if decision["decision"] == "promote" else 1

    cfg, systems, patch_ids, record = load_inputs(args)
    targets = args.target_label if args.target_label is not None else systems["candidate"]["targets"]
    tag = "dry" if args.local_dry_run else "job"
    out = (args.output_dir or (Path("/tmp/gate-out") if args.in_job else default_out(args.round, tag))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _attach(out)
    (out / "gate_inputs.json").write_text(json.dumps(
        {"round": args.round, "config": cfg, "systems": record, "targets": targets,
         "user_model": args.user_model, "mode": tag}, indent=1))

    if args.local_dry_run:
        status = {}
        for who, s in systems.items():
            run_dir = out / s["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            code = run(baseline_cmd(cfg, s, patch_ids[who], run_dir, args), log=run_dir / "baseline.log")
            status[s["name"]] = {"baseline_exit": code, "bfcl": run_bfcl(cfg, s, run_dir, args)}
    else:
        try:
            status = serve_and_run(cfg, systems, patch_ids, out, args)
        except Exception as exc:  # noqa: BLE001 - still decide (reject) and publish the logs
            print(f"serving or running failed: {exc}", file=sys.stderr)
            status = {"_error": str(exc)[:1000]}

    metrics = {}
    for who, s in systems.items():
        run_dir, st = out / s["name"], status.get(s["name"]) or {}
        if st.get("baseline_exit") != 0 or not (run_dir / "traces.jsonl").exists():
            metrics[who] = missing_metrics(s["name"], f"tau2 run failed (exit {st.get('baseline_exit')})")
            continue
        if run(classify_cmd(cfg, run_dir), log=run_dir / "classify.log") != 0:
            metrics[who] = missing_metrics(s["name"], "classify failed")
            continue
        metrics[who] = metrics_for(run_dir, cfg, s["name"], st.get("bfcl")) or missing_metrics(s["name"], "no metrics")
    (out / "run_status.json").write_text(json.dumps(status, indent=1, default=str))
    decide_and_write(out, cfg, args.round, metrics, record, targets)

    if args.in_job and not args.no_push:
        push(out, args, out / "metrics.jsonl", systems["candidate"]["name"])
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    if "--in-job" in sys.argv and "--bootstrapped" not in sys.argv:
        sys.exit(main())  # first pass in an HF Job: the harness (and shipwright.log) is not installed yet
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="gate"))
