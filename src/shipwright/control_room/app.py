"""Control-room: read-only replay of a round, step by step. Reads only the run-dir
files the harness scripts already produce - it never calls a teacher API, a trainer,
or the gate itself (docs/system-design.html, "Demo: the loop, round by round").

Unlike a single `runs/round-N/` folder, each step here writes into its own run dir
(`scripts/baseline.py` -> `runs/baseline-<ts>/`, `scripts/build_sft.py` -> its own
`--output-dir`, `scripts/gate_eval.py` -> `runs/gate-round-N-<mode>-<ts>/`), so this
app asks for up to three dirs instead of one round number - the same three a person
tracks by hand when running a round for real (docs/baseline.md, docs/training.md,
docs/gate.md).

Run: streamlit run src/shipwright/control_room/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _candidate_dirs(prefix: str) -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted((p.name for p in RUNS_DIR.glob(f"{prefix}*") if p.is_dir()), reverse=True)


def _pick_dir(label: str, prefix: str, key: str) -> Path | None:
    """A dropdown of runs/<prefix>* dirs, plus a manual path for anything elsewhere."""
    options = ["(none)", *_candidate_dirs(prefix), "(enter a path)"]
    choice = st.sidebar.selectbox(label, options, key=f"{key}_choice")
    if choice == "(none)":
        return None
    if choice == "(enter a path)":
        manual = st.sidebar.text_input(f"{label} path", key=f"{key}_manual")
        return Path(manual) if manual else None
    return RUNS_DIR / choice


def render_run(baseline_dir: Path, round_num: int | None) -> None:
    traces = _read_jsonl(baseline_dir / "traces.jsonl")
    failures = _read_jsonl(baseline_dir / "failures.jsonl")
    by_task = {str(f.get("task_id")): f for f in failures}

    st.subheader("1. Run - student trace")
    for t in traces:
        tid = str(t.get("task_id"))
        label = f"task {tid}, trial {t.get('trial')} - reward {t.get('reward')}"
        with st.expander(label):
            for turn in t.get("agent_turns") or []:
                st.markdown(f"**turn {turn.get('message_index')}**")
                st.write(turn.get("raw_output") or "(no raw output captured)")
                for tc in turn.get("parsed_tool_calls") or []:
                    st.code(json.dumps(tc, indent=2), language="json")

    st.subheader("2. Detect")
    st.table([{"task_id": f.get("task_id"), "outcome_failed": f.get("outcome_failed"),
              "detected_by": "+".join(f.get("detected_by") or [])} for f in failures])

    st.subheader("3. Classify")
    st.table([{"task_id": f.get("task_id"), "labels": ",".join(f.get("labels") or []),
              "agenttune_root_cause": f.get("agenttune_root_cause"), "source": f.get("source"),
              "explanation": f.get("explanation")} for f in failures])

    st.subheader("4. Route")
    st.table([{"task_id": f.get("task_id"), "fault_layer": f.get("fault_layer"),
              "route": f.get("route")} for f in failures])

    return by_task


def render_generate(baseline_dir: Path, round_num: int | None, sft_dir: Path | None) -> None:
    st.subheader("5. Generate / patch")

    ledger = _read_jsonl(Path("patches") / "ledger.jsonl")
    if round_num is not None:
        ledger = [e for e in ledger if e.get("round") == round_num]
    if ledger:
        st.write("Tool-fault branch (no training, verified immediately):")
        st.table([{k: e.get(k) for k in ("id", "tool", "field", "before", "after", "status")} for e in ledger])
    else:
        st.caption("No tool patches recorded for this round in patches/ledger.jsonl.")

    if sft_dir is not None:
        kept = _read_jsonl(sft_dir / "kept.jsonl")
        summary = _read_json(sft_dir / "teacher_summary.json")
        if kept:
            st.write("Model-fault branch - student vs teacher trace:")
            for ex in kept[:20]:
                col1, col2 = st.columns(2)
                col1.markdown(f"**student (task {ex.get('task_id')})**")
                col1.write(ex.get("rejected") or ex.get("student") or "(not captured)")
                col2.markdown("**teacher**")
                col2.write(ex.get("chosen") or ex.get("teacher") or "(not captured)")
        if summary:
            st.caption("Teacher run summary:")
            st.json(summary)
    else:
        st.caption("No teacher-run dir given - weight-fix generation not shown.")


def render_curate_and_train(baseline_dir: Path, sft_dir: Path | None) -> None:
    st.subheader("6. Curate / train")

    round_status = _read_json(baseline_dir / "round_status.json")
    if round_status:
        st.write("Retrain trigger (AgentTune's RetrainingTrigger, scripts/round_status.py):")
        st.json(round_status)
    else:
        st.caption("No round_status.json in this run dir yet - run scripts/round_status.py.")

    if sft_dir is not None:
        stats = _read_json(sft_dir / "sft_stats.json")
        if stats:
            st.write("SFT data built (scripts/build_sft.py):")
            st.json(stats)
        training_meta = _read_json(sft_dir / "training_meta.json")
        if training_meta:
            st.write("LoRA training run (scripts/train_lora.py):")
            st.json(training_meta)


def render_gate(gate_dir: Path | None) -> dict | None:
    st.subheader("7. Gate")
    decision = _read_json(gate_dir / "gate_decision.json") if gate_dir else None
    if decision:
        st.json(decision)
    else:
        st.caption("No gate-eval dir given - run scripts/gate_eval.py and point this app at its output.")
    return decision


def render_outcome(decision: dict | None) -> None:
    st.subheader("8. Outcome")
    if not decision:
        st.info("No gate decision to show yet.")
        return
    reason = "; ".join(decision.get("reasons") or [])
    if decision.get("decision") == "promote":
        st.success(f"Promoted. {reason}")
    else:
        st.error(f"Rejected: old system kept. {reason}")


def main() -> None:
    st.set_page_config(page_title="Shipwright control room", layout="wide")
    st.title("Shipwright: the loop, round by round")

    st.sidebar.header("Run dirs for this round")
    baseline_dir = _pick_dir("Baseline run dir (traces + failures)", "baseline-", "baseline")
    sft_dir = _pick_dir("Teacher/SFT output dir (optional)", "sft-", "sft")
    gate_dir = _pick_dir("Gate-eval output dir (optional)", "gate-", "gate")
    round_num = st.sidebar.number_input("Round number (for patches/ledger.jsonl)", min_value=0, step=1, value=1)

    if baseline_dir is None or not baseline_dir.exists():
        st.warning(
            f"Pick a baseline run dir in the sidebar (looked under {RUNS_DIR}). "
            "Run scripts/baseline.py first, for real or with --dry-run."
        )
        return

    render_run(baseline_dir, round_num)
    render_generate(baseline_dir, round_num, sft_dir)
    render_curate_and_train(baseline_dir, sft_dir)
    decision = render_gate(gate_dir)
    render_outcome(decision)

    if gate_dir is not None:
        st.divider()
        st.subheader("metrics.jsonl, one row per gate-eval run in this dir")
        st.table(_read_jsonl(gate_dir / "metrics.jsonl"))


if __name__ == "__main__":
    main()
