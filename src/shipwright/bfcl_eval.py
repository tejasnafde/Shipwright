"""BFCL v3 regression check for the gate, scored with AuditKIT.

AuditKIT loads the data (``auditkit.load_bfcl``) and scores each reply
(``ToolCallF1(arg_match=bfcl_arg_match)``, BFCL's accepted-value matching). Tiny Aya has no
native tool calling, so ``ak.ToolCallAdapter`` (native ``tool_calls``) does not fit. This
module is the adapter: it renders each BFCL question with our prompted-tools system prompt
(``prompted_tools.convert_messages``), sends it to the served model, and parses the reply's
```json block with ``prompted_tools.parse_tool_call``. The parsed call goes to AuditKIT as
the trace.

Our protocol allows one tool call per reply, so only single-call categories are used:
``simple``, ``multiple`` and the no-call ``irrelevance`` category.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from shipwright.context import REPLY_TOKENS
from shipwright.prompted_tools import convert_messages, parse_tool_call

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
SUPPORTED = ("simple", "multiple", "irrelevance")


def bfcl_files(root: Path, category: str) -> tuple[Path, Path | None]:
    q = Path(root) / f"BFCL_v3_{category}.json"
    a = Path(root) / "possible_answer" / f"BFCL_v3_{category}.json"
    return q, (a if a.exists() else None)


def download(revision: str, categories: list[str], token: str | None = None) -> Path:
    """Download the category files at a pinned revision (HF Hub, free)."""
    from huggingface_hub import snapshot_download

    patterns = [f"BFCL_v3_{c}.json" for c in categories] + [f"possible_answer/BFCL_v3_{c}.json" for c in categories]
    return Path(snapshot_download(BFCL_REPO, repo_type="dataset", revision=revision,
                                  allow_patterns=patterns, token=token))


def load_samples(root: Path, categories: list[str], revision: str, limit: int | None) -> list:
    import auditkit as ak

    samples = []
    for cat in categories:
        if cat not in SUPPORTED:
            raise ValueError(f"BFCL category {cat!r} not supported (one call per reply): use {SUPPORTED}")
        q, a = bfcl_files(root, cat)
        for s in ak.load_bfcl(str(q), str(a) if a else None, revision=revision, limit=limit):
            s.metadata["bfcl_category"] = cat
            samples.append(s)
    return samples


def sample_messages(sample) -> list[dict]:
    """The prompted request for one BFCL sample (system prompt with the compact tool list)."""
    msgs = list(sample.metadata.get("messages") or [{"role": "user", "content": sample.input}])
    return convert_messages(msgs, sample.tools or [])


def score_reply(sample, reply: str | None) -> dict:
    """AuditKIT ToolCallF1 (BFCL matcher) on one reply. Returns exact match and the call."""
    import auditkit as ak

    call = parse_tool_call(reply)
    turns = [[{"name": call[0], "arguments": call[1]}]] if call else []
    scores = ak.ToolCallF1(arg_match=ak.bfcl_arg_match).score(sample, reply or "", {"trace": {"tool_calls": turns}})
    by_name = {s.name.split("_custom_")[0]: s.value for s in scores}
    return {"id": sample.metadata.get("bfcl_id"), "category": sample.metadata.get("bfcl_category"),
            "exact": by_name.get("tool_call_exact", 0.0), "f1": by_name.get("tool_call_f1", 0.0),
            "call": {"name": call[0], "arguments": call[1]} if call else None}


def evaluate(samples: list, chat: Callable[[list[dict]], str]) -> dict:
    """Run every sample through ``chat`` (messages -> reply text) and score it."""
    rows = []
    for s in samples:
        try:
            reply = chat(sample_messages(s))
        except Exception as exc:  # noqa: BLE001 - a failed request scores as no call, and is counted
            rows.append({"id": s.metadata.get("bfcl_id"), "category": s.metadata.get("bfcl_category"),
                         "exact": 0.0, "f1": 0.0, "call": None, "error": str(exc)[:200]})
            continue
        rows.append(score_reply(s, reply))
    per_cat: dict[str, list[float]] = {}
    for r in rows:
        per_cat.setdefault(r["category"], []).append(r["exact"])
    acc = sum(r["exact"] for r in rows) / len(rows) if rows else None
    return {
        "accuracy_points": round(100 * acc, 2) if acc is not None else None,
        "per_category_points": {c: round(100 * sum(v) / len(v), 2) for c, v in per_cat.items()},
        "num_samples": len(rows),
        "num_request_errors": sum(1 for r in rows if r.get("error")),
        "rows": rows,
    }


def make_chat(api_base: str, model: str, timeout: float = 120):
    import httpx

    client = httpx.Client(timeout=timeout)

    def chat(messages: list[dict]) -> str:
        r = client.post(f"{api_base}/chat/completions",
                        json={"model": model, "messages": messages, "temperature": 0, "max_tokens": REPLY_TOKENS})
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""

    return chat
