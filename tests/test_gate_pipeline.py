import json
import sys
from pathlib import Path

import pytest

from shipwright import bfcl_eval, systems
from shipwright.classify import classify_run
from shipwright.tool_fix import load_ledger, select_patches

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))

import gate_eval  # noqa: E402
import gate_job  # noqa: E402
import train_job  # noqa: E402

BASE = {"name": "base", "model": "CohereLabs/tiny-aya-global"}


# ---------------------------------------------------------------- systems

def test_validate_system_defaults_and_rules():
    s = systems.validate_system(dict(BASE))
    assert s["adapter"] is None and s["tool_patches"] is None and s["targets"] == [] and s["route"] == "none"
    with pytest.raises(ValueError):
        systems.validate_system({**BASE, "model": "CohereLabs/c4ai-command-r7b-12-2024"})  # not allowed
    with pytest.raises(ValueError):
        systems.validate_system({**BASE, "name": "Bad Name"})
    with pytest.raises(ValueError):
        systems.validate_system({**BASE, "adapter": {"repo": "x/y"}})
    with pytest.raises(ValueError):
        systems.validate_system({**BASE, "tool_patches": {"ids": "all"}})


def test_repo_system_configs_load():
    for name in ("base", "r1-tool", "r1-weight"):
        s = systems.load_system(ROOT / "configs" / "systems" / f"{name}.yaml", ROOT)
        assert s["name"] == name
    w = systems.load_system(ROOT / "configs" / "systems" / "r1-weight.yaml", ROOT)
    assert w["adapter"] == {"repo": "mitushigoyal5/shipwright-lora", "revision": "round-1"}
    assert systems.served_name(w) == "tiny-aya-r1-weight"
    assert systems.agent_model_id(w) == "openai/tiny-aya-r1-weight"


def test_served_names_stay_prompted():
    from shipwright.prompted_tools import is_prompted_model

    for cfg in (BASE, {**BASE, "name": "r2", "adapter": {"repo": "a/b", "revision": "round-2"}}):
        assert is_prompted_model(systems.agent_model_id(systems.validate_system(cfg)))


def test_select_patches():
    rows = [{"id": "a", "status": "applied"}, {"id": "b", "status": "proposed"},
            {"id": "c", "status": "rolled_back"}]
    assert [r["id"] for r in select_patches(rows, "applied")] == ["a"]
    picked = select_patches(rows, ["c", "b"])
    assert [r["id"] for r in picked] == ["b", "c"]  # ledger order
    assert all(r["status"] == "applied" for r in picked)
    assert rows[1]["status"] == "proposed"  # input untouched
    with pytest.raises(KeyError):
        select_patches(rows, ["zzz"])


def test_resolve_patch_ids_and_baseline_args():
    s = systems.load_system(ROOT / "configs" / "systems" / "r1-tool.yaml", ROOT)
    ledger = load_ledger(ROOT / "patches" / "ledger.jsonl")
    ids = systems.resolve_patch_ids(s, ledger)
    assert ids == [r["id"] for r in ledger if r["status"] == "applied"]
    args = systems.baseline_args(s, ids, ROOT)
    assert args[0] == "--tool-patches" and args[2] == "--tool-patch-ids" and args[3:] == ids
    assert systems.baseline_args(systems.validate_system(dict(BASE)), [], ROOT) == []


def test_serve_plan_lora_and_merged():
    cur = systems.validate_system(dict(BASE))
    cand = systems.validate_system({**BASE, "name": "r1-weight",
                                    "adapter": {"repo": "mitushigoyal5/shipwright-lora", "revision": "round-1"}})
    [srv] = systems.serve_plan([cur, cand], "lora")
    assert srv["served_base"] == "CohereLabs/tiny-aya-global"
    assert srv["loras"] == {"tiny-aya-r1-weight": cand["adapter"]} and srv["systems"] == ["base", "r1-weight"]
    merged = systems.serve_plan([cur, cand], "merged")
    assert [p["served_base"] for p in merged] == ["CohereLabs/tiny-aya-global", "tiny-aya-r1-weight"]
    assert merged[0]["merged"] is None and merged[1]["merged"] == cand["adapter"]
    with pytest.raises(ValueError):
        systems.serve_plan([cur, {**cand, "model": "CohereLabs/aya-expanse-8b"}])


def test_vllm_cmd_enables_lora_only_with_adapters():
    scfg = {"vllm": "vllm==0.32.0", "max_lora_rank": 16, "max_model_len": 8192}
    server = {"served_base": "CohereLabs/tiny-aya-global"}
    cmd = gate_eval.vllm_cmd(server, scfg, {"tiny-aya-r1-weight": Path("/tmp/a")}, "CohereLabs/tiny-aya-global")
    assert "--enable-lora" in cmd and "tiny-aya-r1-weight=/tmp/a" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"
    assert "--enable-lora" not in gate_eval.vllm_cmd(server, scfg, {}, "CohereLabs/tiny-aya-global")


# ---------------------------------------------------------------- BFCL

def test_bfcl_fixture_scoring():
    samples = bfcl_eval.load_samples(FIX / "bfcl", ["simple", "multiple", "irrelevance"], "fixture", None)
    assert len(samples) == 4
    good = {
        "triangle": '```json\n{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}\n```',
        "Factorial": '{"name": "math.factorial", "arguments": {"number": 5}}',
        "weather": '```json\n{"name": "get_weather", "arguments": {"city": "paris"}}\n```',
    }

    def chat(messages):
        assert messages[0]["role"] == "system" and "To call a tool" in messages[0]["content"]
        text = messages[-1]["content"]
        return next((v for k, v in good.items() if k in text), "Paris is the capital of France.")

    res = bfcl_eval.evaluate(samples, chat)
    assert res["accuracy_points"] == 100.0 and res["num_request_errors"] == 0

    def bad(messages):
        raise RuntimeError("down")

    res = bfcl_eval.evaluate(samples, bad)
    assert res["num_request_errors"] == 4
    assert res["per_category_points"]["simple"] == 0.0


def test_bfcl_rejects_parallel_categories():
    with pytest.raises(ValueError):
        bfcl_eval.load_samples(FIX / "bfcl", ["parallel"], "fixture", None)


# ---------------------------------------------------------------- scripts

def test_train_job_prices():
    for flavor, price in {"a10g-large": 1.50, "l40sx1": 1.80, "a100-large": 2.50, "rtx-pro-6000": 2.75}.items():
        assert train_job.PRICES[flavor] == price
    assert train_job.estimate_cost("a100-large", "2h") == (2.50, 5.0)


def test_gate_job_plan_only(capsys):
    rc = gate_job.main(["--round", "1", "--current", "configs/systems/base.yaml",
                        "--candidate", "configs/systems/r1-weight.yaml", "--user-model", "openai/u"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "a100-large at $2.50/h" in out and "max cost:  $5.00" in out
    assert "Nothing was uploaded or launched" in out
    assert "tiny-aya-r1-weight" in out


def test_gate_job_requires_user_model():
    with pytest.raises(SystemExit):
        gate_job.parse_args(["--round", "1", "--current", "a", "--candidate", "b"])


def test_gate_eval_modes():
    with pytest.raises(SystemExit):
        gate_eval.parse_args(["--round", "1"])  # no mode
    with pytest.raises(SystemExit):  # job mode needs the user model
        gate_eval.parse_args(["--round", "1", "--in-job", "--current", "a", "--candidate", "b"])
    a = gate_eval.parse_args(["--round", "1", "--local-dry-run", "--current", "configs/systems/base.yaml",
                              "--candidate", "configs/systems/r1-tool.yaml"])
    assert a.current == ROOT / "configs" / "systems" / "base.yaml"


def _run_dir(path: Path, traces: list[dict]) -> Path:
    path.mkdir(parents=True)
    (path / "traces.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
    rows = [r for r in (classify_run(t) for t in traces) if r]
    (path / "failures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "run_config.json").write_text(json.dumps({"task_split": "test"}))
    return path


def test_decide_only_rejects_small_runs_and_appends_metrics(tmp_path):
    t = [json.loads((FIX / f"{n}.json").read_text()) for n in ("passed_run", "wrong_tool_delivered_order")]
    cur = _run_dir(tmp_path / "cur", t)
    cand = _run_dir(tmp_path / "cand", t)
    out = tmp_path / "out"
    args = ["--round", "1", "--current-run", str(cur), "--candidate-run", str(cand), "--output-dir", str(out),
            "--target-label", "tool_example_copied"]
    assert gate_eval.main(args) == 1  # reject
    assert gate_eval.main(args) == 1
    decision = json.loads((out / "gate_decision.json").read_text())
    assert decision["decision"] == "reject"
    failed = {c["name"] for c in decision["checks"] if not c["passed"]}
    assert {"valid_runs", "target", "bfcl"} <= failed  # 1 trial per task, no BFCL in these dirs
    rows = (out / "metrics.jsonl").read_text().splitlines()
    assert len(rows) == 2 and json.loads(rows[0])["round"] == 1


def test_local_dry_run_end_to_end(tmp_path):
    out = tmp_path / "dry"
    rc = gate_eval.main(["--round", "0", "--local-dry-run", "--current", "configs/systems/base.yaml",
                         "--candidate", "configs/systems/r1-tool.yaml", "--num-tasks", "1",
                         "--output-dir", str(out)])
    assert rc == 0
    for f in ("gate_decision.json", "metrics_current.json", "metrics_candidate.json", "metrics.jsonl",
              "gate_inputs.json", "base/traces.jsonl", "r1-tool/failures.jsonl", "r1-tool/bfcl.json"):
        assert (out / f).exists(), f
    d = json.loads((out / "gate_decision.json").read_text())
    assert d["decision"] == "reject"  # 3 runs per system is far below the floor
    assert d["metrics_candidate"]["active_tool_patches"]
    assert d["metrics_current"]["num_runs"] == 3
