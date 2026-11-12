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

from shipwright.llm.context import REPLY_TOKENS
from shipwright.llm.prompted_tools import convert_messages, parse_tool_call

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
    import auditkit

    samples = []
    for category in categories:
        if category not in SUPPORTED:
            raise ValueError(f"BFCL category {category!r} not supported (one call per reply): use {SUPPORTED}")
        questions_path, answers_path = bfcl_files(root, category)
        answers_argument = str(answers_path) if answers_path else None
        for sample in auditkit.load_bfcl(str(questions_path), answers_argument, revision=revision, limit=limit):
            sample.metadata["bfcl_category"] = category
            samples.append(sample)
    return samples


def sample_messages(sample) -> list[dict]:
    """The prompted request for one BFCL sample (system prompt with the compact tool list)."""
    messages = list(sample.metadata.get("messages") or [{"role": "user", "content": sample.input}])
    return convert_messages(messages, sample.tools or [])


def score_reply(sample, reply: str | None) -> dict:
    """AuditKIT ToolCallF1 (BFCL matcher) on one reply. Returns exact match and the call."""
    import auditkit

    parsed_call = parse_tool_call(reply)
    tool_call_turns = [[{"name": parsed_call[0], "arguments": parsed_call[1]}]] if parsed_call else []
    scorer = auditkit.ToolCallF1(arg_match=auditkit.bfcl_arg_match)
    scores = scorer.score(sample, reply or "", {"trace": {"tool_calls": tool_call_turns}})
    value_by_name = {score.name.split("_custom_")[0]: score.value for score in scores}
    if parsed_call:
        call = {"name": parsed_call[0], "arguments": parsed_call[1]}
    else:
        call = None
    return {
        "id": sample.metadata.get("bfcl_id"),
        "category": sample.metadata.get("bfcl_category"),
        "exact": value_by_name.get("tool_call_exact", 0.0),
        "f1": value_by_name.get("tool_call_f1", 0.0),
        "call": call,
    }


def evaluate(samples: list, chat: Callable[[list[dict]], str]) -> dict:
    """Run every sample through ``chat`` (messages -> reply text) and score it."""
    rows = []
    for sample in samples:
        try:
            reply = chat(sample_messages(sample))
        except Exception as request_error:  # noqa: BLE001 - a failed request scores as no call, and is counted
            rows.append({
                "id": sample.metadata.get("bfcl_id"),
                "category": sample.metadata.get("bfcl_category"),
                "exact": 0.0,
                "f1": 0.0,
                "call": None,
                "error": str(request_error)[:200],
            })
            continue
        rows.append(score_reply(sample, reply))

    exact_scores_by_category: dict[str, list[float]] = {}
    for row in rows:
        exact_scores_by_category.setdefault(row["category"], []).append(row["exact"])

    if rows:
        accuracy = sum(row["exact"] for row in rows) / len(rows)
        accuracy_points = round(100 * accuracy, 2)
    else:
        accuracy_points = None
    return {
        "accuracy_points": accuracy_points,
        "per_category_points": {
            category: round(100 * sum(scores) / len(scores), 2)
            for category, scores in exact_scores_by_category.items()
        },
        "num_samples": len(rows),
        "num_request_errors": sum(1 for row in rows if row.get("error")),
        "rows": rows,
    }


def make_chat(api_base: str, model: str, timeout: float = 120):
    import httpx

    http_client = httpx.Client(timeout=timeout)

    def chat(messages: list[dict]) -> str:
        from shipwright.observability import retry

        return retry.call_with_retry(once, messages, policy=retry.LLM, what=f"llm_bfcl {model}")

    def once(messages: list[dict]) -> str:
        from shipwright.observability import instrument

        request_body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": REPLY_TOKENS}
        with instrument.external("llm_bfcl", model, role="agent") as call_record:
            response = http_client.post(f"{api_base}/chat/completions", json=request_body)
            call_record["status"] = response.status_code
            response.raise_for_status()
            response_body = response.json()
            call_record.update(instrument.usage_fields(response_body))
            return response_body["choices"][0]["message"].get("content") or ""

    return chat
