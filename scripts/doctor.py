"""Is this machine and account ready to run the loop? Checks every integration the loop relies on, before any money is spent.

    shipwright doctor            # free: accounts, tokens, configs, data, every command the loop builds
    shipwright doctor --cloud    # also a vLLM start on an HF GPU with the adapter (about $0.1)

Every check below exists because the thing it checks broke a real run (AGENTS.md, "Known traps"): a token without the Jobs
scope, a training row the chat template rejects, a launcher that does not take an option, vLLM that needs a compiler. `shipwright
run --launch` runs the free checks before it starts the cloud execution and starts nothing if one fails. Each check says what it saw, so a
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

from shipwright import cloud  # noqa: E402

RUNS_REPO = cloud.RUNS_REPO
ADAPTER_REPO = cloud.LORA_REPO


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
    token = cloud.hf_token()
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
        gcs_runs.list_remote_runs(cloud.OWNERS[1])
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


def round_configs_are_valid() -> Check:
    """configs/rounds/*.yaml: a cap a GPU job could pass, or a short gate timeout, fails here before the cloud execution
    starts."""
    name = "every round config is valid"
    import yaml

    from shipwright.loop import config as config_module

    bad = []
    for path in sorted((ROOT / "configs" / "rounds").glob("*.yaml")):
        problems = config_module.validate(yaml.safe_load(path.read_text()))
        bad += [f"{path.name}: {problem}" for problem in problems]
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
    """The by-hand launchers parse the commands the docs give, and the gate command names the bundled system configs (plan
    only, nothing launches)."""
    name = "the launcher commands parse"
    import gate_job
    import train_job

    from shipwright.loop import jobs

    problems = []
    try:
        train_job.parse_args(["--dataset-revision", "round-99", "--revision", "round-99", "--", "--epochs", "2"])
    except SystemExit as error:
        problems.append(f"train_job: {error}")
    base = ["--round", "99", "--current", "configs/systems/base.yaml", "--candidate", "configs/systems/r8-weight.yaml",
            "--user-model", "openai/openai/gpt-oss-120b:deepinfra"]
    for extra in ([], ["--bfcl-screen"], ["--train-eval", "--task-ids", "0", "1"]):
        try:
            job = gate_job.job_from_options(gate_job.parse_args(base + extra))
            arguments = jobs.gate_script_arguments(job, "gate/round-99/code/x.tar.gz")
            if "configs/systems/generated/r8-weight.json" not in arguments:
                problems.append(f"gate_job {extra}: the candidate is not the bundled config")
            flagged = [argument for argument in arguments if Path(argument).suffix in jobs.HF_LOCAL_FILE_SUFFIXES]
            if flagged:
                problems.append(f"gate_job {extra}: huggingface_hub would treat {flagged} as local files (round 12)")
        except SystemExit as error:
            problems.append(f"gate_job {extra}: {error}")
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


def observation_format_and_repair_behave() -> Check:
    """The student's view of a product listing is shortened, an id of the wrong kind is rerouted, and an over-long request is
    shortened before anything is dropped: the three harness behaviours the loop relies on (gate 8, replay of 9,085 calls)."""
    name = "tool-result shortening and id repair behave"
    import json as json_module

    from shipwright.llm import context, guards, prompted_tools

    listing = {"name": "Shelf", "product_id": "P2", "variants": {
        "111": {"item_id": "111", "options": {"c": "red"}, "available": True, "price": 1.0},
        "222": {"item_id": "222", "options": {"c": "blue"}, "available": False, "price": 2.0}}}
    shown = prompted_tools.observation_text("get_product_details", json_module.dumps(listing))
    if '"222"' in shown or "available_variants" not in shown:
        return fail(name, f"a listing was not shortened: {shown[:120]}")
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {
                   "name": "get_product_details", "arguments": '{"product_id": "P2"}'}}]},
               {"role": "tool", "tool_call_id": "a", "content": json_module.dumps(listing)}]
    fixed = guards.repaired_call(history, guards.ToolCall("get_product_details", {"product_id": "111"}), {"get_item_details"})
    if fixed != guards.ToolCall("get_item_details", {"item_id": "111"}):
        return fail(name, f"an item id given to get_product_details was not rerouted: {fixed}")
    bulky = {"role": "user", "content": "Tool result for get_product_details: " + json_module.dumps(
        {**listing, "variants": {str(n): {"item_id": str(n), "options": {"c": "x" * 40}, "available": True, "price": 1.0}
                                 for n in range(40)}}, separators=(",", ":"))}
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "hi"}, bulky, bulky, bulky, bulky]
    fitted, info = context.fit_messages(messages, budget=context.count_messages(messages) - 50)
    if info["results_shortened"] < 1 or context.count_messages(fitted) > context.count_messages(messages) - 50:
        return fail(name, f"an over-long request was not shortened to fit: {info}")
    return ok(name)


def state_reminders_behave() -> Check:
    """The per-turn status note (llm/reminders.py): says the customer is not identified before a lookup, names the user id after
    one, repeats the last tool error, and goes into the request only, never into the conversation it was built from."""
    name = "the state reminders behave on a fixed conversation"
    from shipwright.llm import reminders

    customer_turn = {"role": "user", "content": "hi, I want to return an order"}
    before_lookup = reminders.status_note([customer_turn])
    if before_lookup is None or "not identified" not in before_lookup:
        return fail(name, f"no 'not identified' note before a lookup: {before_lookup}")
    lookup = {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {
        "name": "find_user_id_by_email", "arguments": '{"email": "a@b.c"}'}}]}
    after_lookup = reminders.status_note([customer_turn, lookup, {"role": "tool", "tool_call_id": "a", "content": "sara_doe_1"}])
    if after_lookup is None or "sara_doe_1" not in after_lookup:
        return fail(name, f"the user id is not named after a lookup: {after_lookup}")
    failed = reminders.status_note([customer_turn, lookup, {"role": "tool", "tool_call_id": "a", "content": "Error: user not found"}])
    if failed is None or "Error: user not found" not in failed:
        return fail(name, f"the last tool error is not repeated: {failed}")
    request = [{"role": "system", "content": "policy"}, dict(customer_turn)]
    extended = reminders.add_to_request(request, before_lookup)
    if request[-1]["content"] != "hi, I want to return an order" or before_lookup not in extended[-1]["content"]:
        return fail(name, "the note leaked into the original messages or did not reach the request")
    return ok(name)


def capacity_fixes_behave() -> Check:
    """The 2026-11-15 fixes for context overflow, on fixed inputs: the short policy replaces only tau2's policy block (its
    instructions mention "<policy>" too) and keeps the exact transfer line; a request that drops old turns still says who
    the customer is; an 11th lookup in a row is refused; a simulated customer that talks as the agent is asked again."""
    name = "the short policy, the trim note, the lookup limit and the simulator re-ask behave"
    import json as json_module
    from types import SimpleNamespace

    from shipwright.llm import context, guards, patches, prompted_tools

    system = ("<instructions>\nHelp according to the <policy> provided below.\n</instructions>\n<policy>\n# Retail agent "
              "policy\n\nLong rules.\n\nRespond in English.\n</policy>")
    short = prompted_tools.condensed_system(system)
    if "provided below" not in short or "Long rules" in short or "PLEASE HOLD ON." not in short or not short.endswith(
            "Respond in English.\n</policy>"):
        return fail(name, f"the short policy is not in place of tau2's block: {short[:200]!r}")
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "lookup"},
                {"role": "user", "content": 'Tool result for find_user_id_by_email: "sara_doe_1"'},
                {"role": "assistant", "content": "x " * 400}, {"role": "user", "content": "y " * 400},
                {"role": "assistant", "content": "ok?"}, {"role": "user", "content": "yes"}]
    fitted, info = context.fit_messages(messages, budget=300)
    if not info["dropped_messages"] or not any("authenticated as user id sara_doe_1" in m["content"] for m in fitted):
        return fail(name, f"a trimmed request lost the customer's user id: {info}")
    history = [{"role": "user", "content": "hi"}]
    for number in range(guards.LOOKUP_STREAK_LIMIT):
        history += [{"role": "assistant", "content": None, "tool_calls": [{"id": str(number), "type": "function", "function": {
            "name": "get_item_details", "arguments": json_module.dumps({"item_id": str(number)})}}]},
            {"role": "tool", "tool_call_id": str(number), "content": "{}"}]
    if guards.check_lookup_streak(history, guards.ToolCall("get_item_details", {"item_id": "x"})) is None:
        return fail(name, "an 11th lookup in a row was not refused")
    agent_like = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="I'm sorry, but I can't verify your account without the phone number."))])
    if not patches.is_agent_voiced_reply(agent_like):
        return fail(name, "a simulated customer speaking as the agent was not caught")
    return ok(name)


def observation_v3_compacts_records() -> Check:
    """Observation format 3: an order record arrives without empty fields and with a one-line address, every id and price kept."""
    name = "observation format 3 compacts an order record"
    import json as json_module

    from shipwright.llm import prompted_tools

    if prompted_tools.OBSERVATION_VERSION != 3:
        return fail(name, f"OBSERVATION_VERSION is {prompted_tools.OBSERVATION_VERSION}, this check expects 3 (update it with the format)")
    order = {"order_id": "#W1", "user_id": "u1", "status": "delivered", "fulfillments": [], "notes": None,
             "address": {"address1": "1 Main St", "address2": "", "city": "Austin", "country": "USA", "state": "TX", "zip": "78701"},
             "items": [{"name": "Shelf", "product_id": "P2", "item_id": "111", "price": 1.5, "options": {"c": "red"}}]}
    shown = prompted_tools.observation_text("get_order_details", json_module.dumps(order))
    parsed = json_module.loads(shown)
    if "fulfillments" in parsed or "notes" in parsed or isinstance(parsed.get("address"), dict):
        return fail(name, f"empty fields or a multi-line address survived: {shown[:200]}")
    for kept in ("#W1", "u1", "delivered", "111", "1.5", "Austin"):
        if kept not in shown:
            return fail(name, f"{kept} was lost from the shown record: {shown[:200]}")
    return ok(name)


def one_command_round_runs_offline() -> Check:
    """`shipwright run --dry-run` on the round 10 config with fake models and fake HF jobs (nothing is billed): the config's
    recipe block, the reminders switch on the candidate, the stages and the promote step all run once. About 15 seconds."""
    name = "shipwright run --dry-run runs the round 10 config offline"
    command = [sys.executable, str(ROOT / "scripts" / "loop.py"), "run", "--config", "configs/rounds/r10-fixes.yaml",
               "--dry-run", "--rounds", "1"]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    chain_file = ROOT / "runs" / "dry-chain-r10" / "chain.json"
    if completed.returncode != 0 or not chain_file.exists():
        return fail(name, (completed.stderr or completed.stdout)[-300:])
    chain = json.loads(chain_file.read_text())
    if chain.get("status") != "promoted":
        return fail(name, f"the dry chain ended {chain.get('status')}: {chain.get('reason')}")
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
    git_is_clean_and_pushed, hf_token_scopes, vertex_token, gcloud_project, runs_dataset_reachable, system_configs_are_valid, round_configs_are_valid,
    launchers_accept_the_loops_commands, guards_behave, observation_format_and_repair_behave, state_reminders_behave, observation_v3_compacts_records,
    capacity_fixes_behave,
    one_command_round_runs_offline, sft_rows_render_with_the_real_template,
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
