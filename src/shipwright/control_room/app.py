"""Control-room: read-only replay of a round, step by step. Reads only the run-dir
files the harness scripts already produce - it never calls a teacher API, a trainer,
or the gate itself (docs/system-design.html, "Demo: the loop, round by round").

Unlike a single `runs/round-N/` folder, each step here writes into its own run dir
(`scripts/baseline.py` -> `runs/baseline-<ts>/`, `scripts/build_sft.py` -> its own
`--output-dir`, `scripts/gate_eval.py` -> `runs/gate-round-N-<mode>-<ts>/`), so this
app asks for up to three dirs instead of one round number - the same three a person
tracks by hand when running a round for real (docs/baseline.md, docs/training.md,
docs/gate.md).

Layout: a persistent header (round outcome + KPI tiles + a pipeline stepper showing
where the loop is and whether each stage is healthy) stays on screen while five tabs
hold the full detail per stage, so a viewer always has "is this working" at a glance
and can still drill into every row behind it - nothing summarized away.

Run: streamlit run src/shipwright/control_room/app.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import streamlit as st

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"

CSS = """
<style>
.cr-pipeline{display:flex;align-items:flex-start;margin:0.25rem 0 1.25rem;overflow-x:auto;
  padding-bottom:4px;}
.cr-node{display:flex;flex-direction:column;align-items:center;min-width:88px;flex-shrink:0;}
.cr-dot{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:0.95rem;border:2px solid transparent;
  color:#fff;}
.cr-dot.pending{background:transparent;border-color:#9aa0a6;color:#9aa0a6;}
.cr-dot.info{background:#2563eb;}
.cr-dot.warn{background:#d97706;}
.cr-dot.good{background:#16a34a;}
.cr-dot.bad{background:#dc2626;}
.cr-label{margin-top:6px;font-size:0.78rem;font-weight:600;text-align:center;
  line-height:1.15;}
.cr-caption{font-size:0.68rem;color:#8a8f98;text-align:center;max-width:96px;
  line-height:1.2;margin-top:2px;}
.cr-line{flex:1 1 20px;height:2px;background:#9aa0a6;margin-top:17px;min-width:16px;}
.cr-line.done{background:#2563eb;}
.cr-badge{display:inline-block;padding:0.3rem 0.9rem;border-radius:999px;font-weight:700;
  font-size:0.85rem;letter-spacing:0.02em;}
.cr-badge.good{background:rgba(22,163,74,0.15);color:#16a34a;border:1px solid #16a34a;}
.cr-badge.bad{background:rgba(220,38,38,0.15);color:#dc2626;border:1px solid #dc2626;}
.cr-badge.pending{background:rgba(154,160,166,0.15);color:#9aa0a6;border:1px solid #9aa0a6;}
.cr-sidecaption{color:#8a8f98;font-size:0.8rem;margin-top:-0.5rem;margin-bottom:0.75rem;}
</style>
"""

STAGES = ("Run", "Detect", "Classify", "Route", "Fix / Generate", "Train", "Gate", "Outcome")


# ---------------------------------------------------------------- data loading

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
    """A dropdown of runs/<prefix>* dirs (most recent selected by default), plus a
    manual path for anything elsewhere."""
    candidates = _candidate_dirs(prefix)
    options = ["(none)", *candidates, "(enter a path)"]
    default_index = 1 if candidates else 0
    choice = st.sidebar.selectbox(label, options, index=default_index, key=f"{key}_choice")
    if choice == "(none)":
        return None
    if choice == "(enter a path)":
        manual = st.sidebar.text_input(f"{label} path", key=f"{key}_manual")
        return Path(manual) if manual else None
    return RUNS_DIR / choice


class RoundData:
    """Everything read from disk for one round, loaded once per render."""

    def __init__(self, baseline_dir: Path, sft_dir: Path | None, gate_dir: Path | None, round_num: int):
        self.baseline_dir = baseline_dir
        self.sft_dir = sft_dir
        self.gate_dir = gate_dir
        self.round_num = round_num

        self.traces = _read_jsonl(baseline_dir / "traces.jsonl")
        self.failures = _read_jsonl(baseline_dir / "failures.jsonl")
        self.failure_summary = _read_json(baseline_dir / "failure_summary.json")
        self.round_status = _read_json(baseline_dir / "round_status.json")
        self.run_summary = _read_json(baseline_dir / "summary.json")

        ledger_all = _read_jsonl(Path("patches") / "ledger.jsonl")
        self.ledger = [e for e in ledger_all if e.get("round") == round_num]

        self.sft_kept = _read_jsonl(sft_dir / "kept.jsonl") if sft_dir else []
        self.teacher_summary = _read_json(sft_dir / "teacher_summary.json") if sft_dir else None
        self.sft_stats = _read_json(sft_dir / "sft_stats.json") if sft_dir else None
        self.training_meta = _read_json(sft_dir / "training_meta.json") if sft_dir else None

        self.gate_decision = _read_json(gate_dir / "gate_decision.json") if gate_dir else None
        self.metrics_rows = _read_jsonl(gate_dir / "metrics.jsonl") if gate_dir else []

    # ------------------------------------------------- derived counts

    @property
    def num_tasks(self) -> int:
        return len({str(t.get("task_id")) for t in self.traces})

    @property
    def num_trials(self) -> int:
        return len(self.traces)

    @property
    def num_passed(self) -> int:
        return sum(1 for t in self.traces if (t.get("reward") or 0) >= 1)

    @property
    def pass_rate(self) -> float | None:
        return self.num_passed / self.num_trials if self.num_trials else None

    @property
    def label_counts(self) -> Counter:
        c: Counter = Counter()
        for f in self.failures:
            c.update(f.get("labels") or [])
        return c

    @property
    def route_counts(self) -> Counter:
        return Counter(f.get("route") for f in self.failures if f.get("route"))

    @property
    def gate_checks(self) -> list[dict]:
        return (self.gate_decision or {}).get("checks") or []

    @property
    def gate_checks_passed(self) -> tuple[int, int]:
        checks = [c for c in self.gate_checks if not c.get("skipped")]
        return sum(1 for c in checks if c.get("passed")), len(checks)


# ---------------------------------------------------------------- stage status

def _stage_status(data: RoundData) -> list[dict]:
    """One {name, state, caption} per STAGES entry. state: pending|info|warn|good|bad."""
    out = []

    if data.traces:
        out.append({"state": "info", "caption": f"{data.num_tasks} tasks x {data.num_trials // max(data.num_tasks, 1)} trials"})
    else:
        out.append({"state": "pending", "caption": "no traces yet"})

    n_detected = sum(1 for f in data.failures if f.get("outcome_failed") or "agenttune_detector" in (f.get("detected_by") or []))
    if data.traces:
        out.append({"state": "warn" if n_detected else "good", "caption": f"{n_detected} failed" if n_detected else "all passed"})
    else:
        out.append({"state": "pending", "caption": "-"})

    if data.failures:
        top = data.label_counts.most_common(1)
        out.append({"state": "info", "caption": f"top: {top[0][0]}" if top else "classified"})
    else:
        out.append({"state": "pending" if data.traces else "pending", "caption": "none to classify" if data.traces else "-"})

    if data.failures:
        rc = data.route_counts
        out.append({"state": "info", "caption": " / ".join(f"{k}:{v}" for k, v in rc.items()) or "-"})
    else:
        out.append({"state": "pending", "caption": "-"})

    n_patches = len(data.ledger)
    n_teacher = len(data.sft_kept)
    if n_patches or n_teacher:
        out.append({"state": "info", "caption": f"{n_patches} patches, {n_teacher} corrections"})
    else:
        out.append({"state": "pending", "caption": "none generated"})

    if data.round_status is not None:
        should = data.round_status.get("should_train")
        out.append({"state": "warn" if should else "good", "caption": "training" if should else "buffer not full"})
    elif data.training_meta:
        out.append({"state": "info", "caption": "trained"})
    else:
        out.append({"state": "pending", "caption": "no decision yet"})

    if data.gate_decision is not None:
        decision = data.gate_decision.get("decision")
        passed, total = data.gate_checks_passed
        out.append({"state": "good" if decision == "promote" else "bad", "caption": f"{passed}/{total} checks"})
    else:
        out.append({"state": "pending", "caption": "not run"})

    if data.gate_decision is not None:
        decision = data.gate_decision.get("decision")
        out.append({"state": "good" if decision == "promote" else "bad",
                    "caption": "promoted" if decision == "promote" else "rejected"})
    else:
        out.append({"state": "pending", "caption": "-"})

    return out


def render_pipeline(data: RoundData) -> None:
    statuses = _stage_status(data)
    icons = {"pending": "", "info": "", "warn": "!", "good": "OK", "bad": "X"}
    parts = ['<div class="cr-pipeline">']
    for i, (name, status) in enumerate(zip(STAGES, statuses)):
        state = status["state"]
        parts.append(
            f'<div class="cr-node"><div class="cr-dot {state}">{icons[state]}</div>'
            f'<div class="cr-label">{name}</div>'
            f'<div class="cr-caption">{status["caption"]}</div></div>'
        )
        if i < len(STAGES) - 1:
            done = state not in ("pending",)
            parts.append(f'<div class="cr-line {"done" if done else ""}"></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_header(data: RoundData) -> None:
    decision = data.gate_decision
    outcome_state = "pending"
    outcome_text = "ROUND IN PROGRESS"
    if decision is not None:
        outcome_state = "good" if decision.get("decision") == "promote" else "bad"
        outcome_text = "PROMOTED" if outcome_state == "good" else "REJECTED"

    top = st.columns([2, 1, 1, 1, 1])
    with top[0]:
        st.markdown(f"### Round {data.round_num}")
        st.markdown(f'<span class="cr-badge {outcome_state}">{outcome_text}</span>', unsafe_allow_html=True)
    with top[1]:
        st.metric("Trials run", data.num_trials)
    with top[2]:
        pr = data.pass_rate
        st.metric("Pass rate", f"{pr:.0%}" if pr is not None else "-")
    with top[3]:
        st.metric("Failures found", len(data.failures))
    with top[4]:
        passed, total = data.gate_checks_passed
        st.metric("Gate checks", f"{passed}/{total}" if total else "-")

    render_pipeline(data)


# ---------------------------------------------------------------- tabs

def tab_run_trace(data: RoundData) -> None:
    if not data.traces:
        st.info("No traces yet - run scripts/baseline.py (for real, or with --dry-run).")
        return
    st.caption(f"{data.num_tasks} tasks, {data.num_trials} trials, from `{data.baseline_dir.name}/traces.jsonl`.")
    for t in data.traces:
        tid = str(t.get("task_id"))
        passed = (t.get("reward") or 0) >= 1
        badge = "good" if passed else "bad"
        label = f"task {tid} · trial {t.get('trial')} · reward {t.get('reward')}"
        with st.expander(label):
            st.markdown(f'<span class="cr-badge {badge}">{"PASSED" if passed else "FAILED"}</span>', unsafe_allow_html=True)
            for turn in t.get("agent_turns") or []:
                st.markdown(f"**turn {turn.get('message_index')}**")
                st.write(turn.get("raw_output") or "(no raw output captured)")
                for tc in turn.get("parsed_tool_calls") or []:
                    st.code(json.dumps(tc, indent=2), language="json")


def tab_classify_route(data: RoundData) -> None:
    if not data.failures:
        st.success("No failures to classify - every traced trial passed.") if data.traces else \
            st.info("No traces yet.")
        return

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Detect**")
        st.dataframe(
            [{"task_id": f.get("task_id"), "outcome_failed": f.get("outcome_failed"),
              "detected_by": "+".join(f.get("detected_by") or [])} for f in data.failures],
            width="stretch", hide_index=True,
        )
    with cols[1]:
        st.markdown("**Label distribution**")
        if data.label_counts:
            st.bar_chart(dict(data.label_counts))
        else:
            st.caption("No labels yet.")

    st.markdown("**Classify**")
    st.dataframe(
        [{"task_id": f.get("task_id"), "labels": ", ".join(f.get("labels") or []),
          "agenttune_root_cause": f.get("agenttune_root_cause"), "source": f.get("source"),
          "explanation": f.get("explanation")} for f in data.failures],
        width="stretch", hide_index=True,
    )

    st.markdown("**Route**")
    rc = data.route_counts
    if rc:
        route_cols = st.columns(len(rc))
        for col, (route, n) in zip(route_cols, rc.most_common()):
            col.metric(route, n)
    st.dataframe(
        [{"task_id": f.get("task_id"), "fault_layer": f.get("fault_layer"), "route": f.get("route")}
         for f in data.failures],
        width="stretch", hide_index=True,
    )


def tab_fix_train(data: RoundData) -> None:
    st.markdown("### Tool-fault branch - patch ledger")
    if data.ledger:
        st.caption("No training, verified and applied immediately (self_heal/router.py's tool-fault path).")
        st.dataframe(
            [{k: e.get(k) for k in ("id", "tool", "field", "before", "after", "status")} for e in data.ledger],
            width="stretch", hide_index=True,
        )
    else:
        st.caption(f"No tool patches recorded for round {data.round_num} in patches/ledger.jsonl.")

    st.divider()
    st.markdown("### Model-fault branch - teacher generation")
    if data.sft_dir is None:
        st.caption("No teacher/SFT output dir selected in the sidebar.")
    else:
        if data.sft_kept:
            for ex in data.sft_kept[:20]:
                col1, col2 = st.columns(2)
                col1.markdown(f"**student (task {ex.get('task_id')})**")
                col1.write(ex.get("rejected") or ex.get("student") or "(not captured)")
                col2.markdown("**teacher**")
                col2.write(ex.get("chosen") or ex.get("teacher") or "(not captured)")
        else:
            st.caption("No teacher corrections kept in this dir.")
        if data.teacher_summary:
            with st.expander("Teacher run summary"):
                st.json(data.teacher_summary)

    st.divider()
    st.markdown("### Retrain trigger & training")
    if data.round_status:
        should = data.round_status.get("should_train")
        st.markdown(
            f'<span class="cr-badge {"warn" if should else "good"}">'
            f'{"SHOULD TRAIN" if should else "BUFFER NOT FULL"}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"AgentTune's RetrainingTrigger: {data.round_status.get('reason')}")
        with st.expander("Full retrain-trigger status"):
            st.json(data.round_status)
    else:
        st.caption("No round_status.json - run scripts/round_status.py.")

    if data.sft_stats:
        st.markdown("**SFT data built** (`scripts/build_sft.py`)")
        st.json(data.sft_stats)
    if data.training_meta:
        st.markdown("**LoRA training run** (`scripts/train_lora.py`)")
        st.json(data.training_meta)


def tab_gate_outcome(data: RoundData) -> None:
    if data.gate_decision is None:
        st.info("No gate-eval dir selected, or it has no gate_decision.json yet.")
        return

    decision = data.gate_decision
    approved = decision.get("decision") == "promote"
    reason = "; ".join(decision.get("reasons") or [])

    if approved:
        st.success(f"**PROMOTED.** {reason}")
    else:
        st.error(f"**REJECTED - old system kept.** {reason}")

    st.markdown("**Checks**")
    for check in data.gate_checks:
        passed = check.get("passed")
        skipped = check.get("skipped")
        icon = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        state = "pending" if skipped else ("good" if passed else "bad")
        cols = st.columns([1, 5])
        cols[0].markdown(f'<span class="cr-badge {state}">{icon}</span>', unsafe_allow_html=True)
        cols[1].markdown(f"**{check.get('name')}** - {check.get('detail')}")

    verdicts = decision.get("verdicts")
    if verdicts:
        with st.expander("AgentTune / AuditKIT verdict detail"):
            st.json(verdicts)

    with st.expander("Full gate decision (raw)"):
        st.json(decision)


def tab_history(data: RoundData) -> None:
    if not data.metrics_rows:
        st.info("No metrics.jsonl rows in the selected gate-eval dir.")
        return

    st.caption(f"{len(data.metrics_rows)} row(s) from `{data.gate_dir.name}/metrics.jsonl`.")
    chart_rows = []
    for row in data.metrics_rows:
        m = row.get("metrics") or {}
        chart_rows.append({
            "round": row.get("round"),
            "current pass_1": (m.get("current") or {}).get("pass_1"),
            "candidate pass_1": (m.get("candidate") or {}).get("pass_1"),
        })
    if any(r.get("current pass_1") is not None or r.get("candidate pass_1") is not None for r in chart_rows):
        st.markdown("**pass^1, current vs candidate, by round**")
        st.line_chart({r["round"]: {"current": r["current pass_1"], "candidate": r["candidate pass_1"]}
                       for r in chart_rows}, x_label="round")

    st.dataframe(
        [{"round": r.get("round"), "decision": r.get("decision"), "reasons": "; ".join(r.get("reasons") or [])}
         for r in data.metrics_rows],
        width="stretch", hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="Shipwright control room", layout="wide")
    st.title("Shipwright: the loop, round by round")
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.header("Run dirs for this round")
    st.sidebar.markdown(
        '<p class="cr-sidecaption">Each step writes its own run dir - pick the three that make up one round.</p>',
        unsafe_allow_html=True,
    )
    baseline_dir = _pick_dir("Baseline run dir", "baseline-", "baseline")
    sft_dir = _pick_dir("Teacher/SFT output dir (optional)", "sft-", "sft")
    gate_dir = _pick_dir("Gate-eval output dir (optional)", "gate-", "gate")
    round_num = st.sidebar.number_input("Round number (for patches/ledger.jsonl)", min_value=0, step=1, value=1)

    if baseline_dir is None or not baseline_dir.exists():
        st.warning(
            f"Pick a baseline run dir in the sidebar (looked under `{RUNS_DIR}`). "
            "Run `scripts/baseline.py` first, for real or with `--dry-run`."
        )
        return

    data = RoundData(baseline_dir, sft_dir, gate_dir, int(round_num))

    render_header(data)
    st.divider()

    tabs = st.tabs(["Run & Trace", "Classify & Route", "Fix & Train", "Gate & Outcome", "History"])
    with tabs[0]:
        tab_run_trace(data)
    with tabs[1]:
        tab_classify_route(data)
    with tabs[2]:
        tab_fix_train(data)
    with tabs[3]:
        tab_gate_outcome(data)
    with tabs[4]:
        tab_history(data)


if __name__ == "__main__":
    main()
