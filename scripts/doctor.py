"""Is this machine and account ready to run the loop? Checks every integration the loop relies on, before any money is spent.

    .venv/bin/python scripts/doctor.py                 # free: accounts, tokens, configs, data, every command the loop builds
    .venv/bin/python scripts/doctor.py --cloud         # also a vLLM start on an HF GPU with the adapter (about $0.1)

Every check below exists because the thing it checks broke a real run (AGENTS.md, "Known traps"): a token without the Jobs
scope, a training row the chat template rejects, a launcher that does not take an option, vLLM that needs a compiler. The loop
(scripts/run_loop.py) runs the free checks before its first step and stops if one fails. Each check says what it saw, so a
red line names its own fix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUNS_REPO = "mitushigoyal5/shipwright-runs"
ADAPTER_REPO = "mitushigoyal5/shipwright-lora"


@dataclass
class Check:
    name: str
    status: str  # "ok", "FAIL" or "skipped"
    detail: str


def ok(name: str, detail: str = "") -> Check:
    return Check(name, "ok", detail)


def fail(name: str, detail: str) -> Check:
    return Check(name, "FAIL", detail)


def skipped(name: str, detail: str) -> Check:
    return Check(name, "skipped", detail)


def git_is_clean_and_pushed() -> Check:
    name = "git: clean tree, HEAD pushed"
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if dirty:
        return fail(name, "uncommitted changes: the cloud image and the gate bundle are built from HEAD")
    unpushed = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--count", "@{upstream}..HEAD"], capture_output=True, text=True)
    if unpushed.returncode != 0 or unpushed.stdout.strip() != "0":
        return fail(name, "HEAD is not pushed: push it so the Cloud Build makes its image")
    return ok(name)


def hf_token_scopes() -> Check:
    """The token can read and launch Jobs (a missing job.write cost a day) and call the router."""
    name = "HF token: Jobs and router"
    import os

    import baseline

    token = baseline.load_token() or os.environ.get("HF_TOKEN")
    if not token:
        return fail(name, "no HF token: put HF_TOKEN in .env or set SWITCHBOARD_SECRETS_DIR")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        who = api.whoami()["name"]
        list(api.list_jobs())
    except Exception as error:  # noqa: BLE001 - the message is the point
        return fail(name, f"{type(error).__name__}: {str(error)[:200]} (the token needs the Jobs permissions)")
    import httpx

    response = httpx.post("https://router.huggingface.co/v1/chat/completions", timeout=60, headers={"Authorization": f"Bearer {token}"},
                          json={"model": "openai/gpt-oss-120b:deepinfra", "max_tokens": 50,
                                "messages": [{"role": "user", "content": "Say ok."}]})
    if response.status_code != 200:
        return fail(name, f"the router answered {response.status_code}: the token needs Inference Providers access")
    return ok(name, f"account {who}")


def vertex_token() -> Check:
    name = "Vertex: a token from gcloud"
    from shipwright.llm import vertex

    try:
        token = vertex.provider().get()
    except Exception as error:  # noqa: BLE001
        return fail(name, f"{type(error).__name__}: {str(error)[:200]} (gcloud --configuration=lexsi auth login)")
    return ok(name) if token else fail(name, "an empty token")


def gcloud_project() -> Check:
    name = "gcloud: the project is reachable"
    from shipwright.control_room import gcs_runs

    try:
        gcs_runs.list_remote_runs("ishank")
    except Exception as error:  # noqa: BLE001
        return fail(name, f"{type(error).__name__}: {str(error)[:200]}")
    return ok(name)


def system_configs_are_valid() -> Check:
    name = "every system config loads"
    from shipwright.heal import systems

    bad = []
    for path in sorted((ROOT / "configs" / "systems").glob("*.yaml")):
        try:
            systems.load_system(path, ROOT)
        except Exception as error:  # noqa: BLE001
            bad.append(f"{path.name}: {error}")
    return fail(name, "; ".join(bad)) if bad else ok(name)


def adapter_exists(revision: str) -> Check:
    name = f"the adapter {ADAPTER_REPO}@{revision} is on the Hub"
    from huggingface_hub import HfApi

    try:
        found = HfApi().file_exists(ADAPTER_REPO, "adapter_config.json", revision=revision)
    except Exception as error:  # noqa: BLE001
        return fail(name, f"{type(error).__name__}: {str(error)[:150]}")
    return ok(name) if found else fail(name, "adapter_config.json is not there")


def runs_dataset_reachable() -> Check:
    name = "the runs dataset answers"
    from huggingface_hub import HfApi

    try:
        HfApi().list_repo_files(RUNS_REPO, repo_type="dataset")
    except Exception as error:  # noqa: BLE001
        return fail(name, f"{type(error).__name__}: {str(error)[:150]}")
    return ok(name)


def launchers_accept_the_loops_commands() -> Check:
    """Each launcher script parses the exact command the loop builds for it (plan only, nothing launches)."""
    name = "the launcher commands parse"
    import argparse as argparse_module

    import run_loop
    import train_job

    problems = []
    try:
        chosen = run_loop.recipe_module.Recipe(epochs=2.0, learning_rate=1e-4)
        command = run_loop.train_command(argparse_module.Namespace(round=99), chosen)
        train_job.parse_args(command[2:])
    except SystemExit as error:
        problems.append(f"train_job: {error}")
    plans = [
        ["--round", "99", "--current", "configs/systems/base.yaml", "--candidate", "configs/systems/r5-weight.yaml",
         "--user-model", run_loop.run_round.GATE_USER_MODEL, "--flavor", "l40sx1", "--timeout", "60m"],
        ["--round", "99", "--current", "configs/systems/base.yaml", "--candidate", "configs/systems/r5-weight.yaml",
         "--user-model", run_loop.run_round.GATE_USER_MODEL, "--bfcl-screen", "--flavor", "l40sx1", "--timeout", "30m"],
        ["--round", "99", "--current", "configs/systems/base.yaml", "--candidate", "configs/systems/r5-weight.yaml",
         "--user-model", run_loop.run_round.GATE_USER_MODEL, "--train-eval", "--trials", "1", "--task-ids", "0", "1",
         "--flavor", "l40sx1", "--timeout", "40m"],
    ]
    for plan in plans:
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "gate_job.py"), *plan], capture_output=True, text=True)
        if completed.returncode != 0:
            problems.append(f"gate_job {' '.join(plan[7:9])}: {(completed.stderr or completed.stdout)[-200:]}")
    return fail(name, " | ".join(problems)) if problems else ok(name)


def sft_rows_render_with_the_real_template(dataset: Path | None = None) -> Check:
    """Every SFT row is accepted by Tiny Aya's chat template (the opening greeting once broke 166 of 185 rows)."""
    name = "SFT rows render with the real chat template"
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return skipped(name, "transformers is not installed here")
    candidates = sorted((ROOT / "runs").glob("r*-sft/sft.jsonl"), key=lambda path: path.stat().st_mtime)
    sft_path = dataset or (candidates[-1] if candidates else None)
    if sft_path is None:
        return skipped(name, "no SFT data on disk yet")
    tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
    rows = [json.loads(line) for line in sft_path.read_text().splitlines() if line.strip()]
    bad = 0
    for row in rows:
        try:
            tokenizer.apply_chat_template(row["messages"], tokenize=False)
        except Exception:  # noqa: BLE001
            bad += 1
    return fail(name, f"{bad} of {len(rows)} rows in {sft_path.name} are rejected") if bad else ok(name, f"{len(rows)} rows in {sft_path.parent.name}")


def guards_behave() -> Check:
    """The new guards on a recorded conversation: a repeat is caught, a first lookup is not."""
    name = "the tool guards behave on a fixed conversation"
    import json as json_module

    from shipwright.llm import guards

    def call(call_id: str) -> dict:
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "get_order_details", "arguments": json_module.dumps({"order_id": "#W1"})}}]}

    messages = [
        {"role": "user", "content": "email a@b.c, order #W1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {
            "name": "find_user_id_by_email", "arguments": json_module.dumps({"email": "a@b.c"})}}]},
        {"role": "tool", "tool_call_id": "a", "content": "u_1"},
        call("b"), {"role": "tool", "tool_call_id": "b", "content": '{"order_id": "#W1", "status": "pending", "items": []}'}]
    first = guards.first_violation(messages[:3], guards.ToolCall("get_order_details", {"order_id": "#W1"}))
    repeat = guards.first_violation(messages, guards.ToolCall("get_order_details", {"order_id": "#W1"}))
    if first is not None or repeat is None or repeat.rule != "repeated_call":
        return fail(name, f"first={first}, repeat={repeat}")
    return ok(name)


def bfcl_replay_is_disjoint_from_the_gate() -> Check:
    name = "the BFCL replay set does not overlap the gate's samples"
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(ROOT / "tests" / "test_bfcl_replay.py")],
                               capture_output=True, text=True, cwd=ROOT)
    return ok(name) if completed.returncode == 0 else fail(name, completed.stdout[-300:])


def vllm_starts_on_a_gpu(adapter_revision: str) -> Check:
    """The only paid check: vLLM 0.32.0 with the adapter on an HF GPU, answering a request (scripts/vllm_smoke.py)."""
    name = "vLLM starts with the adapter on an HF GPU"
    import os

    environment = {**os.environ}
    completed = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "hf"), "jobs", "uv", "run", "--flavor", "a10g-large", "--timeout", "25m",
         "--secrets", "HF_TOKEN", str(ROOT / "scripts" / "vllm_smoke.py"), "--adapter-repo", ADAPTER_REPO,
         "--adapter-revision", adapter_revision], capture_output=True, text=True, env=environment)
    output = completed.stdout + completed.stderr
    return ok(name) if "SMOKE PASS" in output else fail(name, output[-400:])


FREE_CHECKS: list[Callable[[], Check]] = [
    git_is_clean_and_pushed, hf_token_scopes, vertex_token, gcloud_project, runs_dataset_reachable, system_configs_are_valid,
    launchers_accept_the_loops_commands, guards_behave, sft_rows_render_with_the_real_template,
]


def run_checks(cloud: bool = False, adapter_revision: str = "round-5", extra: list[Callable[[], Check]] | None = None) -> list[Check]:
    checks = []
    for check in [*FREE_CHECKS, lambda: adapter_exists(adapter_revision), bfcl_replay_is_disjoint_from_the_gate, *(extra or [])]:
        try:
            checks.append(check())
        except Exception as error:  # noqa: BLE001 - a crashing check is a failed check
            checks.append(fail(getattr(check, "__name__", "check"), f"crashed: {type(error).__name__}: {str(error)[:200]}"))
    if cloud:
        checks.append(vllm_starts_on_a_gpu(adapter_revision))
    return checks


def print_checks(checks: list[Check]) -> None:
    for check in checks:
        print(f"  [{check.status:7}] {check.name}" + (f": {check.detail}" if check.detail else ""))
    failed = [check for check in checks if check.status == "FAIL"]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks fine." + (f" {len(failed)} FAILED." if failed else " Ready."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cloud", action="store_true", help="also start vLLM with the adapter on an HF GPU (about $0.1)")
    parser.add_argument("--adapter-revision", default="round-5")
    options = parser.parse_args()
    checks = run_checks(options.cloud, options.adapter_revision)
    print_checks(checks)
    return 1 if any(check.status == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="run", meta=True))
