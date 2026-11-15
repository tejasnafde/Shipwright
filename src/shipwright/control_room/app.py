"""Control-room: read-only replay of a round, step by step. Reads only the run-dir
files the harness scripts already produce - it never calls a teacher API, a trainer,
or the gate itself (docs/system-design.html, "Observability").

The sidebar offers a few named scenarios (a real round, a staged walkthrough), each a fixed
set of run dirs, so a judge never aligns paths by hand. "Custom (advanced)" exposes the
per-dir pickers: each step writes its own run dir (`scripts/baseline.py` ->
`runs/baseline-<ts>/`, `scripts/build_sft.py` -> its own `--output-dir`,
`scripts/gate_eval.py` -> `runs/gate-round-N-<mode>-<ts>/`), the same dirs a person tracks
by hand when running a round for real (docs/pipeline/baseline.md, docs/pipeline/training.md,
docs/pipeline/gate.md).

Layout: a persistent header (round outcome + KPI tiles + a pipeline stepper showing
where the loop is and whether each stage is healthy) stays on screen while six tabs
hold the full detail per stage. Traces render as chat bubbles (customer, agent reasoning,
tool call, observation); every raw file stays one expander away for auditing.

Run: streamlit run src/shipwright/control_room/app.py
"""

from __future__ import annotations

import datetime
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from shipwright.control_room import gcs_runs, hub_runs, rounds

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"
LEDGER_PATH = RUNS_DIR.parent / "patches" / "ledger.jsonl"



@dataclass
class Scenario:
    """A named, fixed set of run dirs (names under runs/) the sidebar can load in one click."""
    baseline: str
    sft: str | None
    gate: str | None
    after: str | None
    task_id: str
    round_num: int
    summary: str


# Listed only when all of their dirs exist. The staged one is written by scripts/demo_data.py
# and every one of its dirs carries demo.json, so the STAGED banner shows wherever it's read.
SCENARIOS = {
    "Round 1 (real): the prompt leak": Scenario(
        baseline="r1-baseline-train", sft="r1-teacher-t1", gate=None, after=None, task_id="", round_num=1,
        summary="Real run on all 74 train tasks. Tiny Aya passed 3; the top failure is `invented_id`: "
                "28 traces copied `#W1234567` from the prompt's own few-shot example. "
                "Teacher replay kept 17 corrections. No gate yet: no adapter is trained.",
    ),
    "Staged walkthrough: teacher fix + gate rejection": Scenario(
        baseline="baseline-demo-round1", sft="sft-demo-fix", gate="gate-demo-regression",
        after="baseline-demo-round2", task_id="42", round_num=1,
        summary="Hand-written data, not a measured result: one task failing, its teacher fix, "
                "the same task passing, and the gate rejecting a candidate for a real pass^1 regression.",
    ),
}
NO_SCENARIO = "(pick runs by hand below)"

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
.cr-demo-banner{background:#fff4e0;border:1px solid #d97706;border-radius:8px;padding:10px 14px;
  margin-bottom:12px;font-size:0.85rem;color:#7a4a0a;}
.cr-demo-banner b{color:#9a5b00;}
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


@dataclass
class RunInfo:
    name: str
    kind: str  # "real", "dry" or "demo"
    owner: str  # "tejas", "mitushi", "ishank", or "this machine" when the run was not made on Cloud Run
    created: str  # "MM-DD HH:MM" or ""
    passed_text: str  # "3/74 passed" or ""


KIND_RANK = {"real": 0, "dry": 1, "demo": 2}
KIND_LABEL = {"real": "REAL", "dry": "DRY RUN", "demo": "STAGED"}

# What each dropdown lists: a dir qualifies by the files it holds, not by its name, because
# cloud runs are named by their owner (r1-baseline-train) and not by a fixed prefix.
ROLE_MARKER_FILES = {
    "baseline": ("traces.jsonl",),
    "sft": ("kept.jsonl", "sft_stats.json"),
    "gate": ("gate_decision.json",),
}


def _describe_run(run_dir: Path) -> RunInfo:
    repro = _read_json(run_dir / "repro.json") or {}
    arguments = repro.get("argv") or []
    if (run_dir / "demo.json").exists():
        kind = "demo"
    elif any("dry" in str(argument) for argument in arguments) or run_dir.name.startswith("dry"):
        kind = "dry"
    else:
        kind = "real"
    owner = repro.get("owner") or "this machine"
    created = (repro.get("created_at") or "")[5:16].replace("T", " ")
    passed_text = ""
    summary = _read_json(run_dir / "summary.json") or {}
    if summary.get("num_runs") is not None and summary.get("num_passed") is not None:
        passed_text = f"{summary['num_passed']}/{summary['num_runs']} passed"
    return RunInfo(run_dir.name, kind, owner, created, passed_text)


def _run_label(info: RunInfo) -> str:
    """Run name first: a narrow sidebar dropdown cuts the label off at the right."""
    parts = [info.name, KIND_LABEL[info.kind]]
    if info.kind != "demo":
        parts.append(info.owner)
    if info.passed_text:
        parts.append(info.passed_text)
    if info.created:
        parts.append(info.created + " UTC")
    return " · ".join(parts)


def _candidate_runs(role: str) -> list[RunInfo]:
    """Run dirs holding this role's files: real runs first, then dry, then demo, newest first."""
    if not RUNS_DIR.exists():
        return []
    marker_files = ROLE_MARKER_FILES[role]
    infos = []
    for run_dir in RUNS_DIR.iterdir():
        if run_dir.is_dir() and any((run_dir / marker).exists() for marker in marker_files):
            infos.append(_describe_run(run_dir))
    infos.sort(key=lambda info: info.created, reverse=True)
    infos.sort(key=lambda info: KIND_RANK[info.kind])
    return infos


def _demo_note(run_dir: Path | None) -> str | None:
    """A run dir generated by scripts/demo_data.py carries demo.json saying why it's
    staged and what it's standing in for - never a real measured result. Returns the
    note to show, or None for a real run dir."""
    if run_dir is None:
        return None
    marker = _read_json(run_dir / "demo.json")
    return marker.get("note") if marker else None


def render_demo_banner(run_dir: Path | None) -> None:
    note = _demo_note(run_dir)
    if note:
        st.markdown(f'<div class="cr-demo-banner"><b>STAGED EXAMPLE, not a measured result.</b> '
                    f'{note}</div>', unsafe_allow_html=True)


def _pick_dir(label: str, role: str, key: str, container=None, default_to_real_run: bool = True) -> Path | None:
    """A dropdown of the run dirs that hold this role's files, each labelled REAL, DRY RUN
    or DEMO with its owner. Defaults to the newest REAL run, never to a dry or demo one.
    Renders into ``container`` (default: the sidebar itself)."""
    box = container if container is not None else st.sidebar
    infos = _candidate_runs(role)
    labels = {info.name: _run_label(info) for info in infos}
    options = ["(none)", *labels, "(enter a path)"]
    state_key = f"{key}_choice"
    # Only set a default `index` on first render: once session_state holds a value for this key
    # (including one "Show staged demo" set programmatically), passing `index` too makes
    # Streamlit warn about a conflicting default.
    kwargs = {}
    if state_key not in st.session_state:
        has_real_run = bool(infos) and infos[0].kind == "real"
        kwargs["index"] = 1 if has_real_run and default_to_real_run else 0
    choice = box.selectbox(label, options, key=state_key, format_func=lambda name: labels.get(name, name), **kwargs)
    if choice == "(none)":
        return None
    if choice == "(enter a path)":
        manual = box.text_input(f"{label} path", key=f"{key}_manual")
        return Path(manual) if manual else None
    return RUNS_DIR / choice


class RoundData:
    """Everything read from disk for one round, loaded once per render."""

    def __init__(self, baseline_dir: Path, sft_dir: Path | None, gate_dir: Path | None, round_num: int | None):
        self.baseline_dir = baseline_dir
        self.sft_dir = sft_dir
        self.gate_dir = gate_dir
        self.round_num = round_num

        self.traces = _read_jsonl(baseline_dir / "traces.jsonl")
        self.failures = _read_jsonl(baseline_dir / "failures.jsonl")
        self.failure_summary = _read_json(baseline_dir / "failure_summary.json")
        self.round_status = _read_json(baseline_dir / "round_status.json")
        self.run_summary = _read_json(baseline_dir / "summary.json")

        # Tool patches, split by how they relate to this run: the ones it ran with (its
        # run_config.json names them) and the ones proposed from its own failures.
        ledger = _read_jsonl(LEDGER_PATH)
        run_config = _read_json(baseline_dir / "run_config.json") or {}
        active_patch_ids = set(run_config.get("active_tool_patches") or [])
        simulation_ids = {t.get("simulation_id") for t in self.traces if t.get("simulation_id")}
        self.active_patches = [row for row in ledger if row.get("id") in active_patch_ids]
        self.proposed_patches = [row for row in ledger
                                 if simulation_ids.intersection(row.get("source_failures") or [])]

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
        out.append({"state": "pending", "caption": "none to classify" if data.traces else "-"})

    if data.failures:
        rc = data.route_counts
        out.append({"state": "info", "caption": " / ".join(f"{k}:{v}" for k, v in rc.items()) or "-"})
    else:
        out.append({"state": "pending", "caption": "-"})

    n_patches = len(data.proposed_patches)
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


def render_run_kind_banner(run_dir: Path) -> None:
    """Say plainly what the selected baseline run is: real, dry run, or staged demo."""
    info = _describe_run(run_dir)
    if info.kind == "demo":
        return  # render_demo_banner already explains it
    if info.kind == "dry":
        st.warning(f"**Practice run.** `{info.name}` used a fake model to test the plumbing. "
                   "Its numbers mean nothing.")
    else:
        st.success(f"This is a **real run** (`{info.name}`, made by {info.owner}"
                   + (f" on {info.created} UTC" if info.created else "") + ")"
                   + (f": {info.passed_text.replace('passed', 'tasks passed')}" if info.passed_text else "") + ".")


def render_reading_guide() -> None:
    first_visit = "guide_seen" not in st.session_state
    st.session_state["guide_seen"] = True
    with st.expander("Start here: what you are looking at", expanded=first_visit):
        st.markdown(
            "**The idea.** A small model (Tiny Aya, the *student*) answers customer-support tasks. When it fails, we find "
            "out *why*, a stronger model (the *teacher*) shows the right way on those same tasks, and we train the student "
            "on that. A **gate** then tests the retrained student on tasks it has never seen and decides whether it "
            "is allowed to replace the current one. If it is refused, the next round starts from what went wrong.\n\n"
            "**How to use this page**\n\n"
            "1. In the sidebar, pick a **round**. That fills in every run below for you, and the sidebar says in words "
            "what each one is. You only open *Choose runs by hand* for something unusual.\n"
            "2. Open **What changed?** first: what the retrained student does better, what it does worse, "
            "and a task-by-task list. Click any task to read the two conversations next to each other.\n"
            "3. **Follow a Task** walks one training task through the loop: why it failed, what the teacher did, "
            "what the student does now.\n"
            "4. The other tabs hold the detail: every conversation, why runs failed, what we trained on, "
            "the gate's tests, and the history.\n\n"
            "**Labels.** REAL = real models on real tasks. DRY RUN = a fake model, only testing the plumbing. "
            "STAGED = hand-written example data, never a measured result.")


def render_header(data: RoundData) -> None:
    render_run_kind_banner(data.baseline_dir)
    render_demo_banner(data.baseline_dir)
    decision = data.gate_decision
    outcome_state = "pending"
    outcome_text = "NOT TESTED BY THE GATE YET"
    if decision is not None:
        outcome_state = "good" if decision.get("decision") == "promote" else "bad"
        outcome_text = "GATE SAID YES" if outcome_state == "good" else "GATE SAID NO"

    top = st.columns([2, 1, 1, 1, 1])
    with top[0]:
        st.markdown(f"### Round {data.round_num}" if data.round_num is not None else "### The runs you picked")
        st.markdown(f'<span class="cr-badge {outcome_state}">{outcome_text}</span>', unsafe_allow_html=True)
    with top[1]:
        st.metric("Conversations run", data.num_trials)
    with top[2]:
        pr = data.pass_rate
        st.metric("Passed", f"{data.num_passed} ({pr:.0%})" if pr is not None else "-")
    with top[3]:
        st.metric("Failed", len(data.failures))
    with top[4]:
        passed, total = data.gate_checks_passed
        st.metric("Gate tests passed", f"{passed} of {total}" if total else "-")

    render_pipeline(data)


# ---------------------------------------------------------------- shared trace rendering
# Used by Run & Trace, Fix & Train (student vs teacher) and Follow a Task (before/after) -
# one chat-style renderer so a trace looks the same wherever it's shown. Callers never put a
# trace inside an expander: the raw-JSON expander under it could not nest.

USER_AVATAR = ":material/person:"
AGENT_AVATAR = ":material/support_agent:"
TOOL_AVATAR = ":material/database:"
FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)


def _markdown_safe(text: object) -> str:
    """Streamlit markdown reads `$...$` as LaTeX; customer messages are full of prices."""
    return str(text).replace("$", "\\$")


def _truncate(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars total)"


def _short_value(value: object, limit: int = 120) -> str:
    """One argument value as plain text for an inline code span."""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    return _truncate(text, limit).replace("`", "'")


def _render_tool_call(tool_call: dict, heading: str = "Tool call") -> None:
    """The tool name, then one line per argument - not the raw JSON the model emitted."""
    st.markdown(f":blue-badge[:material/build: {heading}] **`{tool_call.get('name')}`**")
    arguments = tool_call.get("arguments")
    if isinstance(arguments, dict):
        if arguments:
            lines = [f"- {name}: `{_short_value(value)}`" for name, value in arguments.items()]
            st.markdown("\n".join(lines))
        else:
            st.caption("no arguments")
    elif arguments:
        # Malformed arguments (src/shipwright/llm/patches.py keeps them as a string).
        st.caption("arguments could not be parsed:")
        st.text(_truncate(str(arguments), 300))


def _turn_reasoning(trace: dict) -> dict[int, str]:
    """message index -> the agent's own reasoning for a tool-call turn: the provider's
    reasoning field if present, else any prose the model wrote around its JSON tool call."""
    reasoning_by_message = {}
    for turn in trace.get("agent_turns") or []:
        if not turn.get("parsed_tool_calls"):
            continue  # a text reply is shown as the reply itself
        text = turn.get("reasoning") or ""
        if not text:
            prose = FENCED_BLOCK.sub("", turn.get("raw_output") or "").strip()
            if prose and not prose.startswith("{"):
                text = prose
        if text and turn.get("message_index") is not None:
            reasoning_by_message[turn["message_index"]] = text
    return reasoning_by_message


def _render_chat(messages: list[dict], reasoning_by_message: dict[int, str]) -> None:
    """One chat bubble per tau2 message: customer, agent (reasoning in muted italics, then
    the tool call or the reply), and the tool's observation (green, red if it errored)."""
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(_markdown_safe(message.get("content") or ""))
        elif role == "tool":
            with st.chat_message("tool", avatar=TOOL_AVATAR):
                if message.get("error"):
                    st.markdown(":red-badge[:material/error: Observation: error]")
                else:
                    st.markdown(":green-badge[:material/check: Observation]")
                st.text(_truncate(str(message.get("content") or "")))
        elif role == "assistant":
            with st.chat_message("assistant", avatar=AGENT_AVATAR):
                reasoning = reasoning_by_message.get(index)
                if reasoning:
                    st.caption(f"*Reasoning: {_markdown_safe(_truncate(reasoning, 400))}*")
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    for tool_call in tool_calls:
                        _render_tool_call(tool_call)
                else:
                    st.markdown(_markdown_safe(_truncate(str(message.get("content") or "(no text)"))))


def _trace_for_task(traces: list[dict], task_id: str) -> dict | None:
    return next((t for t in traces if str(t.get("task_id")) == task_id), None)


def _trace_passed(trace: dict) -> bool:
    return (trace.get("reward") or 0) >= 1


def _render_task_trace(trace: dict, key: str) -> None:
    """The chat view of one trace, then the raw record (token counts, unparsed model output,
    tool schemas) one click below it. ``key`` keeps the expanders apart when one page shows
    several traces."""
    _render_trace_chat(trace)
    with st.expander("🔍 View raw JSON trace", key=f"raw_{key}"):
        st.code(json.dumps(trace, indent=2, ensure_ascii=False), language="json")


def _render_trace_chat(trace: dict) -> None:
    if _trace_passed(trace):
        st.badge("Passed", icon=":material/check_circle:", color="green")
    else:
        st.badge("Failed", icon=":material/cancel:", color="red")
    st.caption(f"trial {trace.get('trial')}, reward {trace.get('reward')}")
    messages = trace.get("messages")
    if messages:
        _render_chat(messages, _turn_reasoning(trace))
        return
    # Older or hand-built traces without a `messages` list (e.g. some --dry-run fixtures):
    # fall back to the coarser agent_turns view instead of showing nothing.
    for turn in trace.get("agent_turns") or []:
        with st.chat_message("assistant", avatar=AGENT_AVATAR):
            tool_calls = turn.get("parsed_tool_calls") or []
            if tool_calls:
                for tool_call in tool_calls:
                    _render_tool_call(tool_call)
            else:
                st.markdown(_markdown_safe(turn.get("raw_output") or "(no raw output captured)"))


def _first_tool_call(trace: dict | None) -> dict | None:
    """The first tool call a trace makes - from `messages` if present, else `agent_turns`."""
    if not trace:
        return None
    for m in trace.get("messages") or []:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return m["tool_calls"][0]
    for turn in trace.get("agent_turns") or []:
        calls = turn.get("parsed_tool_calls") or []
        if calls:
            return calls[0]
    return None


def _render_scalar_tiles(values: dict, max_tiles: int = 8) -> None:
    """The top-level numbers and short strings of a stats file as metric tiles; the full
    file stays one click away for auditing."""
    scalars = [(key, value) for key, value in values.items()
               if isinstance(value, (int, float, str)) and not isinstance(value, bool) and len(str(value)) <= 24]
    scalars = scalars[:max_tiles]
    if scalars:
        columns = st.columns(min(len(scalars), 4))
        for position, (key, value) in enumerate(scalars):
            if isinstance(value, float):
                value = f"{value:.3g}"
            columns[position % len(columns)].metric(key.replace("_", " "), value)


# ---------------------------------------------------------------- tabs

def tab_run_trace(data: RoundData) -> None:
    if not data.traces:
        st.info("No traces yet - run scripts/baseline.py (for real, or with --dry-run).")
        return
    infra_errors = sum(1 for t in data.traces if t.get("error"))
    columns = st.columns(4)
    columns[0].metric("Trials", data.num_trials)
    columns[1].metric("Passed", data.num_passed)
    columns[2].metric("Failed", data.num_trials - data.num_passed)
    columns[3].metric("Infra errors", infra_errors, help="Simulator or provider errors, not the student's fault.")

    st.dataframe(
        [{"task": t.get("task_id"), "trial": t.get("trial"), "reward": t.get("reward"),
          "passed": _trace_passed(t), "termination": t.get("termination_reason"),
          "messages": t.get("num_messages") or len(t.get("messages") or []),
          "error": t.get("error_type") or t.get("error")} for t in data.traces],
        width="stretch", hide_index=True,
    )
    st.caption(f"from `{data.baseline_dir.name}/traces.jsonl`")

    trial_labels = [f"{'PASS' if _trace_passed(t) else 'FAIL'} · task {t.get('task_id')} · trial {t.get('trial')}"
                    for t in data.traces]
    chosen = st.selectbox("Open a trial", range(len(data.traces)), format_func=lambda index: trial_labels[index],
                          key="run_trace_trial")
    with st.container(border=True):
        _render_task_trace(data.traces[chosen], key="run_trace")


ROUTE_MEANING = {
    "tool_fix": "patch the tool description, no training",
    "weight_fix": "teacher replay, then LoRA retrain",
    "no_fix": "environment or infra error, not the model's fault",
}


def tab_classify_route(data: RoundData) -> None:
    if not data.failures:
        if data.traces:
            st.success("No failures to classify - every traced trial passed.")
        else:
            st.info("No traces yet.")
        return

    label_counts = data.label_counts
    route_counts = data.route_counts
    source_counts = Counter(f.get("source") for f in data.failures if f.get("source"))
    top_label, top_count = label_counts.most_common(1)[0] if label_counts else ("-", 0)
    st.info(
        f"- **{len(data.failures)} failed runs** detected out of {data.num_trials}.\n"
        f"- **Top cause:** `{top_label}`, in {top_count} of them.\n"
        f"- **Classified by:** " + ", ".join(f"{source} {count}" for source, count in source_counts.most_common())
        + " (deterministic rules first, an LLM only for what no rule explains)."
    )

    st.markdown("#### Route: which fix each failure gets")
    if route_counts:
        route_columns = st.columns(len(route_counts))
        for column, (route, count) in zip(route_columns, route_counts.most_common()):
            column.metric(route, count, help=ROUTE_MEANING.get(route))
            column.caption(ROUTE_MEANING.get(route, ""))

    st.markdown("#### Why runs failed")
    st.bar_chart(
        {"failed runs": dict(label_counts.most_common())},
        horizontal=True, x_label="failed runs", y_label="label",
    )

    st.markdown("#### Every failure")
    st.dataframe(
        [{"task": f.get("task_id"), "labels": ", ".join(f.get("labels") or []),
          "route": f.get("route"), "fault layer": f.get("fault_layer"), "source": f.get("source"),
          "detected by": "+".join(f.get("detected_by") or []),
          "AgentTune root cause": f.get("agenttune_root_cause"),
          "explanation": f.get("explanation")} for f in data.failures],
        width="stretch", hide_index=True,
    )


def _proposed_dates(patches: list[dict]) -> str:
    """'Nov 08' or 'Nov 08 to Nov 10': when these patches entered the ledger as proposed."""
    dates = sorted({entry["at"][:10] for row in patches for entry in row.get("history") or []
                    if entry.get("status") == "proposed" and entry.get("at")})
    if not dates:
        return "(date unknown)"
    labels = [datetime.date.fromisoformat(date).strftime("%b %d") for date in (dates[0], dates[-1])]
    return labels[0] if labels[0] == labels[1] else f"{labels[0]} to {labels[1]}"


def tab_fix_train(data: RoundData) -> None:
    render_demo_banner(data.sft_dir)
    tool_column, weight_column = st.columns(2)
    with tool_column:
        st.markdown("### Tool fix: patch ledger")
        st.info(
            "- **Generated by:** a ~1ms regex that swaps example ids for placeholders first; "
            "gpt-oss-120b only for what the regex misses, under a strict prompt.\n"
            "- **Approved by:** a human, because a tool edit changes every run from now on.\n"
            "- **Undo:** one-step rollback from `patches/ledger.jsonl`."
        )
        active_tiles = st.columns(2)
        active_tiles[0].metric("Active during this run", len(data.active_patches),
                               help="Applied patches this baseline ran with (its run_config.json).")
        active_tiles[1].metric("Proposed from this run", len(data.proposed_patches),
                               help="Ledger rows whose source failures are runs in this baseline.")
        if data.active_patches:
            st.caption(f"The {len(data.active_patches)} active patches were proposed "
                       f"{_proposed_dates(data.active_patches)}, before this run: they shaped it, "
                       "they were not made from it.")
        if data.proposed_patches:
            source_counts = Counter(row.get("source") for row in data.proposed_patches)
            status_counts = Counter(row.get("status") for row in data.proposed_patches)
            st.caption(f"Proposed from this run: {source_counts.get('rules', 0)} by regex rule, "
                       f"{source_counts.get('llm', 0)} by LLM fallback; "
                       + ", ".join(f"{status} {count}" for status, count in status_counts.most_common()) + ".")
    with weight_column:
        st.markdown("### Weight fix: teacher replay")
        st.info(
            "- **Generated by:** the teacher replays tasks the student failed; only passing runs are kept.\n"
            "- **Approved by:** the 8-check gate on held-out tasks, not a person.\n"
            "- **Undo:** the old adapter keeps serving until the gate promotes."
        )
        if data.sft_dir is not None:
            st.metric("Teacher corrections kept", len(data.sft_kept))

    for title, patches in (("Patches proposed from this run's failures", data.proposed_patches),
                           ("Patches active during this run", data.active_patches)):
        if patches:
            st.markdown(f"**{title}**")
            st.dataframe(
                [{"id": row.get("id"), "proposed": _proposed_dates([row]), "source": row.get("source"),
                  "tool": row.get("tool"), "field": row.get("field"), "before": row.get("before"),
                  "after": row.get("after"), "status": row.get("status")} for row in patches],
                width="stretch", hide_index=True,
            )

    st.divider()
    st.markdown("### Student vs teacher, task by task")
    if data.sft_dir is None:
        st.caption("No teacher/SFT output dir in this scenario.")
    elif not data.sft_kept:
        st.caption("No teacher corrections kept in this dir.")
    else:
        kept_task_ids = [str(example.get("task_id")) for example in data.sft_kept]
        chosen_task_id = st.selectbox("Task", kept_task_ids, key="fix_train_task")
        teacher_trace = _trace_for_task(data.sft_kept, chosen_task_id)
        student_trace = _trace_for_task(data.traces, chosen_task_id)
        student_column, teacher_column = st.columns(2)
        with student_column:
            st.markdown("**Student (Tiny Aya), failed**")
            if student_trace:
                _render_task_trace(student_trace, key="fix_student")
            else:
                st.caption("No matching trace for this task id in the baseline dir.")
        with teacher_column:
            st.markdown("**Teacher, kept for training**")
            _render_task_trace(teacher_trace, key="fix_teacher")
    if data.teacher_summary:
        with st.expander("Teacher run summary (raw)"):
            st.json(data.teacher_summary)

    st.divider()
    st.markdown("### Retrain trigger and training")
    if data.round_status:
        if data.round_status.get("should_train"):
            st.warning(f"**Should train.** AgentTune RetrainingTrigger: {data.round_status.get('reason')}")
        else:
            st.success(f"**Buffer not full, no training yet.** AgentTune RetrainingTrigger: {data.round_status.get('reason')}")
        with st.expander("Full retrain-trigger status (raw)"):
            st.json(data.round_status)
    else:
        st.caption("No round_status.json - run scripts/round_status.py (the loop's train stage writes it).")

    if data.sft_stats:
        st.markdown("**SFT data built** (`scripts/build_sft.py`)")
        _render_scalar_tiles(data.sft_stats)
        with st.expander("sft_stats.json (raw)"):
            st.json(data.sft_stats)
    if data.training_meta:
        st.markdown("**LoRA training run** (`scripts/train_lora.py`)")
        _render_scalar_tiles(data.training_meta)
        with st.expander("training_meta.json (raw)"):
            st.json(data.training_meta)


# Failure-label rates: lower is better, so a drop shows green.
LOWER_IS_BETTER_CHECKS = {"target", "non_target"}
CHECK_TITLES = {
    "pass_1": "pass^1 (task success)",
    "pass_k": "pass^k (succeeds all k tries)",
    "target": "Targeted failure rate",
    "non_target": "Other failure rate",
    "bfcl": "BFCL tool-calling score",
    "agenttune_gate": "AgentTune DeploymentGate",
    "comparable": "Same tasks and k",
    "valid_runs": "Enough valid runs",
}


def _format_check_value(check: dict, value: float) -> str:
    if check.get("name") == "bfcl":
        return f"{value:.2f} pts"
    return f"{value:.1%}"


def _render_score_check(check: dict) -> None:
    """One numeric check as a metric tile: the candidate's value, its delta against the
    live system (green when it improved, red when it regressed) and pass/fail."""
    current, candidate, delta = check.get("current"), check.get("candidate"), check.get("delta")
    name = check.get("name")
    if delta is None:
        delta = candidate - current
    if name == "bfcl":
        delta_text = f"{delta:+.2f} pts"
    else:
        delta_text = f"{delta * 100:+.1f} pts"
    with st.container(border=True):
        if check.get("skipped"):
            st.markdown(":gray-badge[SKIPPED]")
        elif check.get("passed"):
            st.markdown(":green-badge[:material/check: PASS]")
        else:
            st.markdown(":red-badge[:material/close: FAIL]")
        st.metric(
            CHECK_TITLES.get(name, name),
            _format_check_value(check, candidate),
            delta=delta_text,
            delta_color="inverse" if name in LOWER_IS_BETTER_CHECKS else "normal",
            help=check.get("detail"),
        )
        st.caption(f"Current {_format_check_value(check, current)} → Candidate {_format_check_value(check, candidate)}")


def tab_gate_outcome(data: RoundData) -> None:
    render_demo_banner(data.gate_dir)
    if data.gate_decision is None:
        st.info("No gate run selected, or it has no gate_decision.json. A baseline alone answers *how bad is "
                "the student and why*; the gate needs a trained candidate.")
        return

    decision = data.gate_decision
    reasons = decision.get("reasons") or []
    if decision.get("decision") == "promote":
        st.success("**PROMOTED.** The candidate replaces the live system.")
    else:
        st.error("**REJECTED. The live system is kept.**\n\n" + "\n".join(f"- {reason}" for reason in reasons))

    st.markdown("#### The gate's tests, in plain words")
    for check in data.gate_checks:
        if check.get("skipped"):
            continue
        passed = bool(check.get("passed"))
        words = (CHECK_PLAIN_PASS if passed else CHECK_PLAIN_FAIL).get(check.get("name"), check.get("detail") or "")
        st.markdown((":green-badge[passed]" if passed else ":red-badge[failed]") + f" {words}")
    st.caption("The exact rule and numbers behind each test are in the tiles and the raw decision below.")

    score_checks = [c for c in data.gate_checks if c.get("current") is not None and c.get("candidate") is not None]
    other_checks = [c for c in data.gate_checks if c not in score_checks]

    st.markdown("#### Measured deltas: candidate vs the live system")
    st.caption("Big number = candidate. Green = better than live, red = worse. Hover a tile for the exact rule.")
    if score_checks:
        for row_start in range(0, len(score_checks), 3):
            columns = st.columns(3)
            for column, check in zip(columns, score_checks[row_start:row_start + 3]):
                with column:
                    _render_score_check(check)
    else:
        st.caption("No numeric checks in this decision.")

    st.markdown("#### Validity and verdict checks")
    for check in other_checks:
        if check.get("skipped"):
            badge = ":gray-badge[SKIPPED]"
        elif check.get("passed"):
            badge = ":green-badge[PASS]"
        else:
            badge = ":red-badge[FAIL]"
        title = CHECK_TITLES.get(check.get("name"), check.get("name"))
        st.markdown(f"{badge} **{title}**: {_markdown_safe(check.get('detail') or '')}")

    verdicts = decision.get("verdicts")
    if verdicts:
        with st.expander("AgentTune / AuditKIT verdict detail (raw)"):
            st.json(verdicts)
    with st.expander("Full gate decision (raw)"):
        st.json(decision)


def tab_follow_task(data: RoundData, after_dir: Path | None, task_id: str) -> None:
    """The headline 'aha' view: why one task failed, the fix, then the before and after
    traces side by side."""
    render_demo_banner(data.baseline_dir)
    if after_dir:
        render_demo_banner(after_dir)

    if not task_id:
        st.info("Pick a task id in the sidebar to follow it through the loop.")
        return

    before_trace = _trace_for_task(data.traces, task_id)
    before_failure = next((f for f in data.failures if str(f.get("task_id")) == task_id), None)
    teacher_trace = _trace_for_task(data.sft_kept, task_id)
    after_traces = _read_jsonl(after_dir / "traces.jsonl") if after_dir else []
    after_trace = _trace_for_task(after_traces, task_id)

    before_column, fix_column, after_column = st.columns(3)
    with before_column:
        st.markdown(f"#### 1. Before: task {task_id}")
        with st.container(border=True):
            if before_failure:
                labels = before_failure.get("labels") or ["unlabeled"]
                st.markdown("**Why it failed:** " + " ".join(f":red-badge[{label}]" for label in labels))
                st.markdown(f"**Route:** `{before_failure.get('route')}`")
                if before_failure.get("explanation"):
                    st.caption(_markdown_safe(before_failure["explanation"]))
            elif before_trace and _trace_passed(before_trace):
                st.success("This task passed in the baseline.")
            else:
                st.caption("No classified failure for this task.")
        if before_trace:
            st.caption(f"from `{data.baseline_dir.name}`")
            _render_task_trace(before_trace, key="follow_before")
        else:
            st.caption("Task not found in this baseline dir.")

    with fix_column:
        st.markdown("#### 2. The fix")
        with st.container(border=True):
            if teacher_trace:
                st.markdown("**Teacher replay:** its own passing run on this task, used as training data.")
                bad_call, good_call = _first_tool_call(before_trace), _first_tool_call(teacher_trace)
                if bad_call:
                    _render_tool_call(bad_call, heading="Student called")
                if good_call:
                    _render_tool_call(good_call, heading="Teacher called")
            else:
                st.caption("No teacher correction for this task in the selected SFT dir; its fix may "
                           "be a tool patch instead (see Fix & Train).")
        if teacher_trace:
            st.caption(f"from `{data.sft_dir.name}/kept.jsonl`")
            _render_task_trace(teacher_trace, key="follow_fix")

    with after_column:
        st.markdown("#### 3. After")
        with st.container(border=True):
            if after_trace and _trace_passed(after_trace):
                st.success("**Passes** with the trained student.")
            elif after_trace:
                st.error("**Still fails** with the trained student.")
            elif after_dir is None:
                st.caption("No \"after\" run for this round yet. It is made at the start of the next round: the "
                           "trained student runs the same training tasks again. Meanwhile **What changed?** compares "
                           "the student before and after on the gate's held-out tasks.")
            else:
                st.caption(f"Task {task_id} is not in `{after_dir.name}`: that run used other tasks (for example the "
                           "held-out ones). Pick a run on the same training tasks, or look at **What changed?**.")
        if after_trace:
            st.caption(f"from `{after_dir.name}`")
            _render_task_trace(after_trace, key="follow_after")


def tab_history(data: RoundData) -> None:
    if not data.metrics_rows:
        st.info("No metrics.jsonl rows: pick a gate run in the Gate & Outcome tab.")
        return

    latest = data.metrics_rows[-1]
    st.info(f"**{len(data.metrics_rows)} round(s) recorded.** Latest: round {latest.get('round')}, "
            f"**{latest.get('decision')}**.")
    chart_rows = []
    for row in data.metrics_rows:
        m = row.get("metrics") or {}
        chart_rows.append({
            "round": row.get("round"),
            "current pass_1": (m.get("current") or {}).get("pass_1"),
            "candidate pass_1": (m.get("candidate") or {}).get("pass_1"),
        })
    if any(r.get("current pass_1") is not None or r.get("candidate pass_1") is not None for r in chart_rows):
        st.markdown("**pass^1, live vs candidate, by round**")
        st.line_chart({r["round"]: {"live": r["current pass_1"], "candidate": r["candidate pass_1"]}
                       for r in chart_rows}, x_label="round")

    st.dataframe(
        [{"round": r.get("round"), "decision": r.get("decision"), "reasons": "; ".join(r.get("reasons") or [])}
         for r in data.metrics_rows],
        width="stretch", hide_index=True,
    )


def render_pull_runs_button(container) -> None:
    """Copy runs the team's cloud jobs uploaded to GCS that are not in runs/ yet. Needs gcloud."""
    if gcs_runs.gcloud_binary() is None:
        return  # e.g. the Hugging Face Space image has no gcloud
    owner_choice = container.selectbox("Pull runs from GCS", ["all", *gcs_runs.OWNERS], key="pull_owner")
    if not container.button("Pull new runs", width="stretch"):
        return
    owners = list(gcs_runs.OWNERS) if owner_choice == "all" else [owner_choice]
    with container, st.spinner("Copying from GCS..."):
        try:
            results = gcs_runs.pull_runs(owners, RUNS_DIR)
        except RuntimeError as error:
            container.error(str(error))
            return
    pulled = [result.run_name for result in results if result.status == "pulled"]
    failed = [f"{result.run_name}: {result.detail}" for result in results if result.status == "failed"]
    message = f"Pulled {len(pulled)} new run(s): {', '.join(pulled) or 'none, already up to date'}"
    if failed:
        message += ". FAILED: " + "; ".join(failed)
    st.session_state["pull_message"] = message
    st.rerun()


CHECK_PLAIN_FAIL = {
    "pass_1": "It solves fewer tasks on a single try than the current system.",
    "pass_k": "It no longer solves the same task every time it tries (the gate asks for 3 out of 3).",
    "target": "The failures we trained it to fix did not go down enough.",
    "non_target": "More runs now fail for reasons we were not trying to fix.",
    "bfcl": "It got worse at general tool-calling outside our retail tasks (the BFCL quiz).",
    "agenttune_gate": "AgentTune's own deployment check saw a regression.",
    "valid_runs": "Too few of its runs finished to judge it fairly (often a serving problem, not the model).",
    "comparable": "The two systems were not tested on the same tasks, so the comparison is not valid.",
}
CHECK_PLAIN_PASS = {
    "pass_1": "It solves at least as many tasks on a single try.",
    "pass_k": "It is at least as consistent from try to try.",
    "target": "The failures we trained it to fix went down.",
    "non_target": "No new kinds of failure appeared.",
    "bfcl": "It kept its general tool-calling skill.",
    "agenttune_gate": "AgentTune's deployment check saw no regression.",
    "valid_runs": "Enough runs finished to judge it.",
    "comparable": "Both systems were tested on the same tasks.",
}


def _group_by_task() -> dict[str, str]:
    """Task id -> how many changes it needs ("0-1 writes" or "2+ writes"), when tau2 is installed here."""
    try:
        from shipwright.heal import slices

        return slices.write_slice_by_task("retail", "test")
    except Exception:  # noqa: BLE001 - tau2 is not in the Space image; the column is simply left out
        return {}


def _system_name(system: object, default: str) -> str:
    """A system's name: a real gate decision has {"name": ...}, the staged demo data has the bare string."""
    if isinstance(system, dict):
        return system.get("name") or default
    return system or default


def _gate_system_traces(gate_dir: Path, decision: dict) -> tuple[list[dict], list[dict], str, str]:
    systems = decision.get("systems") or {}
    live_name = _system_name(systems.get("current"), "base")
    candidate_name = _system_name(systems.get("candidate"), "")
    return (_read_jsonl(gate_dir / live_name / "traces.jsonl"), _read_jsonl(gate_dir / candidate_name / "traces.jsonl"),
            live_name, candidate_name)


def _render_findings(decision: dict) -> None:
    findings = rounds.what_changed(decision)
    better = [finding for finding in findings if finding.kind == "good"]
    worse = [finding for finding in findings if finding.kind == "bad"]
    columns = st.columns(2)
    with columns[0]:
        st.markdown("##### Better")
        if better:
            for finding in better:
                st.markdown(f":green[**+**] {finding.text}")
        else:
            st.caption("Nothing clearly better.")
    with columns[1]:
        st.markdown("##### Worse")
        if worse:
            for finding in worse:
                st.markdown(f":red[**-**] {finding.text}")
        else:
            st.caption("Nothing clearly worse.")


def _render_numbers(decision: dict) -> None:
    current, candidate = decision.get("metrics_current") or {}, decision.get("metrics_candidate") or {}

    def percent(value):
        return "-" if value is None else f"{value:.0%}"

    def number(value, digits=1):
        return "-" if value is None else f"{value:.{digits}f}"

    bfcl_before, bfcl_after = rounds._bfcl_points(current), rounds._bfcl_points(candidate)
    st.dataframe([
        {"": "Tasks solved on one try", "Before": percent(current.get("pass_1")), "After": percent(candidate.get("pass_1"))},
        {"": "Tasks solved on every try", "Before": percent(current.get("pass_k")), "After": percent(candidate.get("pass_k"))},
        {"": "Tool calls that returned an error", "Before": percent(current.get("tool_call_error_rate")),
         "After": percent(candidate.get("tool_call_error_rate"))},
        {"": "Steps per conversation", "Before": number(current.get("avg_steps")), "After": number(candidate.get("avg_steps"))},
        {"": "General tool-calling quiz (BFCL, points)", "Before": number(bfcl_before), "After": number(bfcl_after)},
    ], hide_index=True, width="stretch")


def _render_task_by_task(gate_dir: Path, decision: dict) -> None:
    before_traces, after_traces, live_name, candidate_name = _gate_system_traces(gate_dir, decision)
    if not before_traces or not after_traces:
        st.caption("The per-task conversations are not in this gate folder.")
        return
    changes = rounds.compare_tasks(before_traces, after_traces, _group_by_task())
    changed = [change for change in changes if change.change != "same"]
    st.markdown("#### Task by task")
    st.caption(f"The gate's held-out tasks, run {changes[0].before_trials} times each by the current student "
               f"(`{live_name}`) and by the retrained one (`{candidate_name}`). Only tasks both ran are listed.")
    show_all = st.radio("Show", ["Only tasks that changed", "All tasks"], horizontal=True, key="compare_filter")
    shown = changes if show_all == "All tasks" else changed
    if not shown:
        st.info("No task changed how often it passes.")
        return
    has_group = any(change.group for change in shown)
    rows = []
    for change in shown:
        row = {"Task": change.task_id}
        if has_group:
            row["Needs"] = change.group
        row["Before"] = f"{change.before_passed} of {change.before_trials}"
        row["After"] = f"{change.after_passed} of {change.after_trials}"
        row["What happened"] = rounds.CHANGE_WORDS[change.change]
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")

    st.markdown("#### Read one task side by side")
    task_ids = [change.task_id for change in shown]
    task_id = st.selectbox("Task", task_ids, key="compare_task")
    left, right = st.columns(2)
    for column, traces, title, key in ((left, before_traces, f"Before: `{live_name}`", "cmp_before"),
                                       (right, after_traces, f"After: `{candidate_name}`", "cmp_after")):
        with column:
            st.markdown(f"##### {title}")
            task_traces = [trace for trace in traces if str(trace.get("task_id")) == task_id]
            if not task_traces:
                st.caption("This system has no run of that task.")
                continue
            trial_numbers = [trace.get("trial") for trace in task_traces]
            trial = st.selectbox("Try", trial_numbers, key=f"{key}_trial_{task_id}",
                                 format_func=lambda number, task_traces=task_traces: (
                                     f"try {number}: " + ("passed" if _trace_passed(
                                         next(trace for trace in task_traces if trace.get("trial") == number)) else "failed")))
            _render_task_trace(next(trace for trace in task_traces if trace.get("trial") == trial), key=f"{key}_{task_id}")


def tab_compare(data: RoundData, entry: rounds.RoundEntry | None) -> None:
    """The first thing to read: what the retrained student does differently, in words, then the proof."""
    decision = data.gate_decision
    if decision is None:
        st.info("**No gate results are loaded for this round.** The gate is the test on tasks the student has never "
                "seen, and this tab compares the student before and after it. "
                + ("Use *Update runs* in the sidebar to download this round's gate results from Hugging Face."
                   if entry and entry.outcome != "unfinished" else
                   "This round has not reached the gate yet."))
        return
    round_words = f"Round {data.round_num}" if data.round_num is not None else "This round"
    if decision.get("decision") == "promote":
        st.success(f"**{round_words}: the gate said yes.** The retrained student is allowed to replace the current one.")
    else:
        st.error(f"**{round_words}: the gate said no.** The current student stays. Below: what the retrained student "
                 "did better, what it did worse, and the task-by-task proof.")
    _render_findings(decision)
    st.markdown("#### The numbers")
    _render_numbers(decision)
    _render_task_by_task(data.gate_dir, decision)


# ---------------------------------------------------------------- sidebar: pick a round

@dataclass
class RoundSelection:
    baseline_dir: Path | None
    sft_dir: Path | None
    gate_dir: Path | None
    after_dir: Path | None
    round_num: int | None
    entry: rounds.RoundEntry | None
    scenario: Scenario | None
    choice: str


CHOICE_BY_HAND = "(choose runs by hand)"


def _round_choices() -> dict[str, rounds.RoundEntry]:
    return {entry.label: entry for entry in rounds.discover_rounds(RUNS_DIR)}


def _available_scenarios() -> dict[str, Scenario]:
    """Scenarios whose run dirs are all on disk (the Space image may ship only some)."""
    available = {}
    for name, scenario in SCENARIOS.items():
        needed = [scenario.baseline, scenario.sft, scenario.gate, scenario.after]
        if all((RUNS_DIR / directory).exists() for directory in needed if directory):
            available[name] = scenario
    return available


def _apply_choice() -> None:
    """on_change of the round dropdown: point the four run pickers at that round's dirs, so the pickers always show
    (and can override) exactly what is loaded."""
    choice = st.session_state["scenario_choice"]
    entry = _round_choices().get(choice)
    scenario = SCENARIOS.get(choice)
    if entry is not None:
        st.session_state["baseline_choice"] = entry.before or "(none)"
        st.session_state["sft_choice"] = entry.teacher or "(none)"
        st.session_state["gate_choice"] = entry.gate or "(none)"
        st.session_state["after_choice"] = entry.after or "(none)"
    elif scenario is not None:
        st.session_state["baseline_choice"] = scenario.baseline
        st.session_state["sft_choice"] = scenario.sft or "(none)"
        st.session_state["gate_choice"] = scenario.gate or "(none)"
        st.session_state["after_choice"] = scenario.after or "(none)"
    st.session_state.pop("task_id_choice", None)


def _default_choice(choices: dict[str, rounds.RoundEntry], scenarios: dict[str, Scenario]) -> str:
    decided = [label for label, entry in choices.items() if entry.outcome != "unfinished"]
    if decided:
        return decided[0]  # the newest round the gate has decided
    if choices:
        return next(iter(choices))
    return next(iter(scenarios), CHOICE_BY_HAND)


def _status_line(done: bool, role: str, meaning: str, run: Path | None, missing: str) -> None:
    mark = ":green[**done**]" if done else ":gray[**missing**]"
    if done and run is not None:
        st.sidebar.markdown(f"{mark} **{role}**: {meaning}  \n`{run.name}`")
    else:
        st.sidebar.markdown(f"{mark} **{role}**: {meaning}  \n{missing}")


def render_loaded_runs(selection: RoundSelection) -> None:
    """The four roles in words, so nobody has to know which folder goes where."""
    st.sidebar.markdown("##### The four runs it loaded")
    _status_line(selection.baseline_dir is not None, "Before", "the student on the training tasks",
                 selection.baseline_dir, "pick one under *Choose runs by hand*")
    _status_line(selection.sft_dir is not None, "Corrections", "what the teacher showed, what it is trained on",
                 selection.sft_dir, "none for this round")
    _status_line(selection.gate_dir is not None, "Gate", "the test on tasks it has never seen",
                 selection.gate_dir, "not on this machine: *Update runs* below downloads it")
    _status_line(selection.after_dir is not None, "After", "the retrained student on the same training tasks",
                 selection.after_dir, "not run yet: it runs at the start of the next round")


def render_update_runs(entry: rounds.RoundEntry | None) -> None:
    """Both ways to get runs that were made elsewhere: the Cloud Run jobs (GCS) and the gate job (Hugging Face)."""
    box = st.sidebar.expander("Update runs")
    box.caption("Runs made on the cloud are not on this machine until you pull them.")
    if gcs_runs.gcloud_binary() is not None:
        render_pull_runs_button(box)
    if entry is not None and entry.outcome != "unfinished":
        if box.button(f"Download round {entry.number}'s gate results (Hugging Face)", width="stretch", key="pull_gate"):
            with st.spinner("Downloading..."):
                try:
                    hub_runs.pull_gate_results(entry.number, RUNS_DIR, entry.candidate)
                except Exception as error:  # noqa: BLE001 - shown to the person, not a crash
                    box.error(f"Could not download them: {error}")
                    return
            st.session_state["pull_message"] = f"Downloaded the gate results of round {entry.number}."
            _apply_choice()
            st.rerun()
    if "pull_message" in st.session_state:
        st.sidebar.success(st.session_state.pop("pull_message"))


def select_round() -> RoundSelection:
    st.sidebar.header("1. Pick a round")
    choices, scenarios = _round_choices(), _available_scenarios()
    names = [*choices, *scenarios, CHOICE_BY_HAND]
    if "scenario_choice" not in st.session_state or st.session_state["scenario_choice"] not in names:
        st.session_state["scenario_choice"] = _default_choice(choices, scenarios)
        _apply_choice()
    choice = st.sidebar.selectbox("Which round do you want to look at?", names, key="scenario_choice",
                                  on_change=_apply_choice,
                                  help="Picking a round fills in the four runs below. A round is one pass of the loop.")
    entry, scenario = choices.get(choice), scenarios.get(choice)
    if scenario:
        st.sidebar.caption(scenario.summary)

    status_slot = st.sidebar.container()
    st.sidebar.header("2. Something unusual?")
    advanced = st.sidebar.expander("Choose runs by hand")
    advanced.caption("Only if you want to mix runs from different rounds. The four boxes are the four roles above.")
    baseline_dir = _pick_dir("Before: the student on the training tasks", "baseline", "baseline", container=advanced)
    sft_dir = _pick_dir("Corrections: the teacher's runs or the training data", "sft", "sft", container=advanced,
                        default_to_real_run=False)
    gate_dir = _pick_dir("Gate: the held-out test", "gate", "gate", container=advanced, default_to_real_run=False)
    after_dir = _pick_dir("After: the retrained student, same tasks as Before", "baseline", "after", container=advanced,
                          default_to_real_run=False)
    if after_dir is not None and after_dir == baseline_dir:
        advanced.warning("After is the same run as Before, so it is ignored.")
        after_dir = None

    expected = (entry.before, entry.teacher, entry.gate, entry.after) if entry else None
    loaded = tuple(path.name if path else None for path in (baseline_dir, sft_dir, gate_dir, after_dir))
    round_num = entry.number if entry and expected == loaded else None
    if scenario and loaded == (scenario.baseline, scenario.sft, scenario.gate, scenario.after):
        round_num = scenario.round_num

    selection = RoundSelection(baseline_dir, sft_dir, gate_dir, after_dir, round_num, entry, scenario, choice)
    with status_slot:
        render_loaded_runs(selection)
    render_update_runs(entry)
    return selection


def select_task_id(data: RoundData, selection: RoundSelection) -> str:
    task_ids = sorted({str(t.get("task_id")) for t in data.traces if t.get("task_id") is not None},
                      key=lambda task_id: (len(task_id), task_id))
    if not task_ids:
        return ""
    # Default to the scenario's task, else the first task the teacher fixed, else the first failure.
    default_task_id = selection.scenario.task_id if selection.scenario else ""
    if default_task_id not in task_ids:
        candidates = [str(example.get("task_id")) for example in data.sft_kept]
        candidates += [str(failure.get("task_id")) for failure in data.failures]
        default_task_id = next((task_id for task_id in candidates if task_id in task_ids), task_ids[0])
    if st.session_state.get("task_id_choice") not in task_ids:
        st.session_state["task_id_choice"] = default_task_id
    st.sidebar.header("3. Follow one task")
    st.sidebar.caption("Used by the Follow a Task tab.")
    return st.sidebar.selectbox("Training task", task_ids, key="task_id_choice")


def main() -> None:
    st.set_page_config(page_title="Shipwright control room", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Shipwright: a self-healing support agent")
    st.caption("The student fails, we find out why, the teacher shows the right way, we retrain, and a gate decides. "
               "Every number is read from the run's own files.")

    selection = select_round()
    if selection.baseline_dir is None or not selection.baseline_dir.exists():
        st.warning(
            f"Pick a round in the sidebar (looked under `{RUNS_DIR}`). If the list is empty, there are no runs on this "
            "machine yet: open *Update runs* to pull them, or run `scripts/baseline.py` (a real or a `--dry-run`)."
        )
        return

    render_reading_guide()
    header_slot = st.container()
    st.divider()
    tabs = st.tabs(["What changed?", "Follow a Task", "Run & Trace", "Classify & Route", "Fix & Train",
                    "Gate & Outcome", "History"])
    data = RoundData(selection.baseline_dir, selection.sft_dir, selection.gate_dir, selection.round_num)
    task_id = select_task_id(data, selection)

    with header_slot:
        render_header(data)
    with tabs[0]:
        tab_compare(data, selection.entry)
    with tabs[1]:
        tab_follow_task(data, selection.after_dir, task_id)
    with tabs[2]:
        tab_run_trace(data)
    with tabs[3]:
        tab_classify_route(data)
    with tabs[4]:
        tab_fix_train(data)
    with tabs[5]:
        tab_gate_outcome(data)
    with tabs[6]:
        tab_history(data)


if __name__ == "__main__":
    main()
