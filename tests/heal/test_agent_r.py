"""Agent-R (arxiv 2501.11425): MCTS tree, pairs, critique, splicing, masks. Offline."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from shipwright.heal import agent_r, sft_data

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


# ---- tiny tau2-shaped conversations ----


def greet():
    return [{"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "I want to return my order."}]


def call(name, args, result, cid="c1"):
    return [{"role": "assistant", "content": None,
             "tool_calls": [{"id": cid, "name": name, "arguments": args, "requestor": "assistant"}]},
            {"role": "tool", "id": cid, "content": result, "requestor": "assistant"}]


def say(text, reply):
    return [{"role": "assistant", "content": text}, {"role": "user", "content": reply}]


FIND = call("find_user_id_by_email", {"email": "a@b.c"}, "user_1")
GOOD_ORDER = call("get_order_details", {"order_id": "#W1"}, '{"status": "delivered"}')
BAD_ORDER = call("get_order_details", {"order_id": "#W0000000"}, "Error: Order not found")
RETURN = call("return_delivered_order_items", {"order_id": "#W1"}, '{"status": "return requested"}')
DONE = say("Your return is requested.", "Thanks ###STOP###")
GIVE_UP = say("I cannot find that order.", "ok ###STOP###")


def test_split_steps_and_keys_ignore_ids():
    msgs = greet() + FIND + DONE
    steps = agent_r.split_steps(msgs)
    assert [len(s) for s in steps] == [2, 2, 2]
    assert [m for s in steps for m in s] == msgs
    other_id = call("find_user_id_by_email", {"email": "a@b.c"}, "user_1", cid="zz")
    assert agent_r.step_key(FIND) == agent_r.step_key(other_id)
    assert agent_r.step_key(FIND) != agent_r.step_key(BAD_ORDER)


def test_tree_insert_backup_and_uct_selection():
    tree = agent_r.Tree("7", children=2)
    good = greet() + FIND + GOOD_ORDER + RETURN + DONE
    bad = greet() + FIND + BAD_ORDER + GIVE_UP
    p_good = tree.insert(0, good, 1.0)
    p_bad = tree.insert(0, bad, 0.0)
    assert agent_r.common_prefix(p_good, p_bad) == 2  # greeting and find step are shared
    assert tree.root.visits == 2 and tree.root.q == 0.5
    shared = tree.nodes[p_good[1]]
    assert shared.visits == 2 and len(shared.children) == 2
    assert tree.prefix(p_good[1]) == greet() + FIND
    # Root and the shared nodes are not expanded yet: select returns the root first.
    assert tree.select() is tree.root
    tree.root.expansions = 2
    tree.nodes[p_good[0]].expansions = 2
    shared.expansions = 2
    # Now UCT descends into the higher-Q branch (the good one).
    assert tree.select().id == p_good[2]
    with pytest.raises(ValueError):
        tree.insert(p_good[2], bad, 0.0)  # rollout does not start with that node's prefix


def test_run_search_expands_with_children_rollouts_and_respects_budget():
    tree = agent_r.Tree("1", children=2)
    calls = []

    def rollout(prefix, n):
        calls.append((len(prefix), n))
        out = []
        for i in range(n):
            tail = (GOOD_ORDER + RETURN + DONE) if i == 0 else (BAD_ORDER + GIVE_UP)
            base = prefix if prefix else greet() + FIND
            out.append({"messages": base + tail, "reward": 1.0 if i == 0 else 0.0, "simulation_id": f"s{len(calls)}{i}"})
        return out

    counts = agent_r.run_search(tree, rollout, simulations=6)
    assert counts["rollouts"] == 6 and calls[0] == (0, 2)
    assert all(n <= 2 for _, n in calls)
    assert len(tree.leaves) == 6
    assert json.loads(json.dumps(tree.to_json()))["nodes"][0]["visits"] == 6


def test_select_pairs_uses_adjacent_good_path_and_thresholds():
    leaves = [
        {"path": [1, 2, 3, 4], "reward": 0.0, "simulation_id": "bad"},
        {"path": [1, 2, 5, 6], "reward": 1.0, "simulation_id": "near"},
        {"path": [1, 7, 8], "reward": 1.0, "simulation_id": "far"},
        {"path": [1, 2, 9], "reward": 0.9, "simulation_id": "almost"},
    ]
    pairs = agent_r.select_pairs(leaves, alpha=1.0, beta=0.2)
    by_bad = {p["bad"]["simulation_id"]: p for p in pairs}
    assert by_bad["bad"]["good"]["simulation_id"] == "near" and by_bad["bad"]["t"] == 2
    assert "almost" not in by_bad  # 0.9 is below alpha 1.0, but only 0.1 below a good leaf (< beta)
    assert "almost" in {p["bad"]["simulation_id"] for p in agent_r.select_pairs(leaves, alpha=1.0, beta=0.05)}
    assert not agent_r.select_pairs(leaves, alpha=1.0, beta=1.5)
    # alpha 0.5: "almost" is good too; it shares as much prefix as "near" and is shorter, so it wins
    assert {p["good"]["simulation_id"] for p in agent_r.select_pairs(leaves, alpha=0.5)} == {"almost"}
    assert len(agent_r.select_good(leaves, alpha=1.0)) == 2


@pytest.mark.parametrize("text,verdict", [
    ("Reasons... Judgment: Bad", "bad"),
    ("judgement: good", "good"),
    ("**Judgment:** Uncertain", "uncertain"),
    ("Judgment: Good\nwait, actually Judgment: Bad", "bad"),
    ("no verdict here", "uncertain"),
])
def test_parse_judgment(text, verdict):
    assert agent_r.parse_judgment(text) == verdict


def test_find_transition_modes():
    bad_steps = agent_r.split_steps(greet() + FIND + BAD_ORDER + GIVE_UP)
    seen = []

    def judge(history, step):
        seen.append(len(history))
        return "Judgment: Bad" if "#W0000000" in json.dumps(step) else "Judgment: Good"

    tr = agent_r.find_transition(bad_steps, 2, "model", judge)
    assert tr["t_prime"] == 2 and tr["found"] and seen == [2]
    assert agent_r.find_transition(bad_steps, 2, "direct")["t_prime"] == 3
    assert agent_r.find_transition(bad_steps, 2, "divergence")["t_prime"] == 2
    none_bad = agent_r.find_transition(bad_steps, 2, "model", lambda h, s: "Judgment: Good")
    assert none_bad == {**none_bad, "t_prime": 3, "found": False}
    msgs = agent_r.critic_messages("POLICY", bad_steps[:2], bad_steps[2])
    assert "Judgment: <Good or Bad or Uncertain>" in msgs[0]["content"]
    assert "get_order_details" in msgs[1]["content"] and "Order not found" in msgs[1]["content"]


def test_build_revision_inline_and_turn():
    bad = agent_r.split_steps(greet() + FIND + BAD_ORDER + GIVE_UP)
    good = agent_r.split_steps(greet() + FIND + GOOD_ORDER + RETURN + DONE)
    rev = agent_r.build_revision(bad, good, t=2, t_prime=3, thought="I made a mistake.")
    # prefix (greet, find) + bad order + give-up text (its user stop removed) + good suffix
    assert rev["n_context_assistant"] == 4
    assert not any(agent_r.is_user_stop(m) for m in rev["messages"][:-1])
    first_good = [m for m in rev["messages"] if m["role"] == "assistant"][4]
    assert first_good["content"] == "I made a mistake." and first_good["tool_calls"][0]["name"] == "get_order_details"
    assert good[2][0]["content"] is None  # the source run is not changed
    turn = agent_r.build_revision(bad, good, t=2, t_prime=2, thought="Oops.", fmt="turn")
    i = turn["n_context_assistant"]
    asst = [m for m in turn["messages"] if m["role"] == "assistant"]
    assert asst[i]["content"] == "Oops." and asst[i + 1]["tool_calls"][0]["name"] == "get_order_details"
    assert {"role": "user", "content": "OK."} in turn["messages"]
    with pytest.raises(ValueError):
        agent_r.build_revision(bad, good, t=3, t_prime=2, thought="x")


def test_revision_mask_and_windows_keep_bad_steps_masked():
    conv = [{"role": "system", "content": "S"}]
    for i in range(12):
        conv += [{"role": "assistant", "content": f"a{i} " + "x" * 300}, {"role": "user", "content": f"u{i} " + "y" * 300}]
    conv.append({"role": "assistant", "content": "final"})
    mask = sft_data.revision_mask(conv, n_context_assistant=8)
    assert mask[0] is True and mask[1] is True and mask[conv.index(conv[-1])] is False
    whole = sft_data.masked_examples(conv, mask)
    assert len(whole) == 1 and sum(not m for m in whole[0][1]) == 5
    windows = sft_data.masked_examples(conv, mask, budget=300)
    assert len(windows) > 1
    trained = []
    for msgs, m in windows:
        assert len(msgs) == len(m) and m[-1] is False
        for x, masked in zip(msgs, m):
            if not masked:
                assert x["role"] == "assistant"
                trained.append(x["content"].split()[0])
            if x["role"] == "assistant" and x["content"].split()[0] in {f"a{i}" for i in range(8)}:
                assert masked  # a bad / prefix step is never a target
    assert set(trained) == {"a8", "a9", "a10", "a11", "final"}


def test_eta_mix_and_general_records():
    general = sft_data.general_records([
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]},
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "user", "content": "no answer"}]},
    ] * 10)
    assert len(general) == 20 and general[0]["messages"][0]["role"] == "user"
    agent = [{"messages": [], "source": "revision"}] * 4
    mixed, info = sft_data.eta_mix(agent, general, eta=0.2)
    assert info["general_wanted"] == 16 and len(mixed) == 20 and info["eta_actual"] == 0.2


def _load_agenttune_events(monkeypatch):
    """AgentTune's real EventLog without torch (agenttune.agentic imports torch)."""
    import agenttune

    path = Path(agenttune.__file__).parent / "agentic" / "events.py"
    spec = importlib.util.spec_from_file_location("agenttune_events_for_test", path)
    events = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, events)  # dataclasses look their module up here
    spec.loader.exec_module(events)
    pkg = types.ModuleType("agenttune.agentic")
    pkg.Event, pkg.EventKind, pkg.EventLog = events.Event, events.EventKind, events.EventLog
    monkeypatch.setitem(sys.modules, "agenttune.agentic", pkg)


def test_train_lora_passes_agent_r_mask_through_agenttune(monkeypatch):
    pytest.importorskip("agenttune")
    _load_agenttune_events(monkeypatch)
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_lora

    cls = train_lora.make_eventlog_class()
    messages = [{"role": "system", "content": "S"}, {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "u"}, {"role": "assistant", "content": "bad"},
                {"role": "user", "content": "err"}, {"role": "assistant", "content": "I made a mistake.\nfix"}]
    mask = [True, True, True, True, True, False]
    row = train_lora.to_eventlog(cls, messages, mask).as_dataset_rows("sft")[0]
    assert row["messages"] == messages and row["loss_mask"] == mask
    plain = train_lora.to_eventlog(cls, messages).as_dataset_rows("sft")[0]
    assert plain["loss_mask"] == [True, False, True, False, True, False]  # unchanged default
    with pytest.raises(ValueError):
        train_lora.context_events(messages, mask[:-1])


def test_dry_run_search_revise_build_end_to_end(tmp_path):
    """The whole Agent-R data path on tau2 with the noisy-oracle fake actor (no network)."""
    pytest.importorskip("tau2")
    search, rev, sft = tmp_path / "search", tmp_path / "rev", tmp_path / "sft"
    run = lambda *a: subprocess.run([PY, *a], cwd=ROOT, capture_output=True, text=True, timeout=600)  # noqa: E731
    r = run("scripts/agent_r_search.py", "--dry-run", "--user-model", "dry-user", "--task-ids", "0",
            "--simulations", "8", "--output-dir", str(search))
    assert r.returncode == 0, r.stderr[-2000:]
    summary = json.loads((search / "search_summary.json").read_text())
    assert summary["rollouts"] == 8 and (search / "trees" / "0.json").exists()
    traces = [json.loads(x) for x in (search / "traces.jsonl").read_text().splitlines()]
    assert all(t["mcts"]["path"] for t in traces)
    tree = json.loads((search / "trees" / "0.json").read_text())
    assert any(n["depth"] >= 1 and n["expansions"] for n in tree["nodes"]), "no rollout from a replayed prefix"

    r = run("scripts/agent_r_revise.py", str(search), "--dry-run", "--iteration", "3", "--output-dir", str(rev))
    assert r.returncode == 0, r.stderr[-2000:]
    rs = json.loads((rev / "revise_summary.json").read_text())
    revisions = [json.loads(x) for x in (rev / "revisions.jsonl").read_text().splitlines() if x.strip()]
    assert rs["revisions"] == len(revisions)
    for x in revisions:
        assert x["t"] <= x["t_prime"] and x["thought"] in agent_r.REVISION_THOUGHTS

    r = run("scripts/build_sft.py", "--agent-r", str(rev), "--estimate-tokens", "--output-dir", str(sft))
    assert r.returncode == 0, r.stderr[-2000:]
    records = [json.loads(x) for x in (sft / "sft.jsonl").read_text().splitlines()]
    for rec in records:
        if rec["source"] == "revision":
            assert len(rec["loss_mask"]) == len(rec["messages"]) and not rec["loss_mask"][-1]
            first_trained = rec["messages"][rec["loss_mask"].index(False)]["content"]
            assert any(first_trained.startswith(t) for t in agent_r.REVISION_THOUGHTS)
