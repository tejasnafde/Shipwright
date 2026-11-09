"""Spans and log lines for every external call and every tau2 step (docs/debugging.md).

- ``install_tau2()`` wraps three tau2 functions from the outside (vendored tau2 is not edited):
  ``tau2.runner.batch.run_unit`` (one task run: span ``task <id>``, log context task_id/trial),
  ``LLMAgent.generate_next_message`` (span ``agent_turn``, log context ``turn``) and
  ``Environment.get_response`` (span ``execute_tool <name>``: latency, error flag).
- ``llm_call()`` is used by ``shipwright.llm_patch`` around every litellm completion that tau2
  makes (agent, user simulator, judge; the teacher is the agent of a teacher run).
- ``external()`` times any other call (Vertex token fetch, GCS upload, HF Hub, classifier, judge).

Every call logs: target (model, URL, repo), latency_ms, status, tokens in/out/cached (LLM calls),
retries, error_class and error. Secrets never reach a log line (``shipwright.log.redact``).
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from typing import Any, Callable, Iterator

from shipwright import log, report, tracing

logger = log.get_logger("shipwright.calls")
_lock = threading.Lock()
_installed = False
# Per task run: agent turn counter and the agent's last input and raw output.
_task_state: contextvars.ContextVar[dict | None] = contextvars.ContextVar("shipwright_task_state", default=None)
TRIM = 1000


def _text(msg: Any) -> str | None:
    if msg is None:
        return None
    if isinstance(msg, dict):
        role, content = msg.get("role"), msg.get("content")
    else:
        role, content = getattr(msg, "role", None), getattr(msg, "content", None)
    if isinstance(content, list):
        content = " ".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content)
    return report.trim(f"[{role}] {content}", TRIM)


def usage_fields(response: Any) -> dict[str, Any]:
    from shipwright.costs import cached_tokens

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {}

    def get(key: str) -> Any:
        return usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)

    return {"tokens_in": get("prompt_tokens"), "tokens_out": get("completion_tokens"),
            "tokens_cached": cached_tokens(usage)}


def _raw_output(response: Any) -> str | None:
    try:
        msg = response.choices[0].message
        text = msg.content
        calls = getattr(msg, "tool_calls", None) or []
        if calls:
            text = (text or "") + " " + " ".join(f"{c.function.name}({c.function.arguments})" for c in calls)
        return report.trim(text, TRIM)
    except Exception:  # noqa: BLE001
        return None


def _error_status(exc: BaseException) -> Any:
    return getattr(exc, "status_code", None) or type(exc).__name__


def llm_call(role: str, model: str, kwargs: dict, call: Callable[[], Any], *, retries: Callable[[], int] | None = None
             ) -> Any:
    """Run one LLM call with a span and a log line. Raises what ``call`` raises."""
    state = _task_state.get()
    messages = kwargs.get("messages") or []
    last_input = _text(messages[-1]) if messages else None
    attrs = {"gen_ai.operation.name": "chat", "gen_ai.request.model": model, "shipwright.role": role,
             "shipwright.messages": len(messages), "shipwright.tools": len(kwargs.get("tools") or [])}
    start = time.perf_counter()
    with log.bind(model=model, component=f"llm.{role}"), tracing.span(f"chat {model}", attrs) as sp:
        try:
            if role == "agent":
                if state is not None:
                    state["agent_calls"] = state.get("agent_calls", 0) + 1
                # Fail on the agent's 2nd call, so the report has an earlier output to show.
                if state is None or state["agent_calls"] >= 2:
                    log.fail_point("agent_call")
            response = call()
        except Exception as e:
            ms = round((time.perf_counter() - start) * 1000, 1)
            logger.error(f"llm call failed: {role} {model}", role=role, latency_ms=ms, status=_error_status(e),
                         retries=retries() if retries else 0, last_input=last_input,
                         last_output=(state or {}).get("last_output"), exc_info=True, stack_info=True)
            raise
        ms = round((time.perf_counter() - start) * 1000, 1)
        usage = usage_fields(response)
        out = _raw_output(response)
        finish = None
        try:
            finish = response.choices[0].finish_reason
        except Exception:  # noqa: BLE001
            pass
        tracing.set_attrs(sp, {"gen_ai.usage.input_tokens": usage.get("tokens_in"),
                               "gen_ai.usage.output_tokens": usage.get("tokens_out"),
                               "gen_ai.usage.cache_read.input_tokens": usage.get("tokens_cached"),
                               "gen_ai.response.finish_reasons": finish, "shipwright.latency_ms": ms})
        if role == "agent" and state is not None:
            state["last_input"], state["last_output"] = last_input, out
        logger.info(f"llm call {role}", role=role, latency_ms=ms, status="ok", finish_reason=finish,
                    retries=retries() if retries else 0, **usage)
        return response


@contextlib.contextmanager
def external(kind: str, target: str, **attrs: Any) -> Iterator[dict]:
    """Time an external call. Set ``info["status"]`` (and other fields) inside the block.

    Logs one line (INFO, or ERROR with the stack on an exception) and makes a span ``kind``.
    """
    info: dict[str, Any] = {}
    start = time.perf_counter()
    span_attrs = {"shipwright.kind": kind, "shipwright.target": target,
                  **{f"shipwright.{k}": v for k, v in attrs.items()}}
    with log.bind(component=kind), tracing.span(kind, span_attrs) as sp:
        try:
            yield info
        except Exception as e:
            ms = round((time.perf_counter() - start) * 1000, 1)
            logger.error(f"{kind} failed: {target}", kind=kind, target=target, latency_ms=ms,
                         status=info.get("status") or _error_status(e), **attrs,
                         **{k: v for k, v in info.items() if k != "status"}, exc_info=True)
            raise
        ms = round((time.perf_counter() - start) * 1000, 1)
        tracing.set_attrs(sp, {"shipwright.latency_ms": ms, "shipwright.status": info.get("status")})
        logger.info(f"{kind} {target}", kind=kind, target=target, latency_ms=ms, status=info.get("status", "ok"),
                    **attrs, **{k: v for k, v in info.items() if k != "status"})


# ---------------------------------------------------------------- tau2 hooks

def install_tau2() -> None:
    """Wrap tau2's task, agent turn and tool execution. Idempotent. Only warns if tau2 changed."""
    global _installed
    with _lock:
        if _installed:
            return
        _installed = True
    try:
        import tau2.runner.batch as batch

        real_run_unit = batch.run_unit

        def run_unit(ctx: Any, task: Any, trial: int, seed: int, progress_str: str = "") -> Any:
            task_id = str(getattr(task, "id", ""))
            token = _task_state.set({"agent_calls": 0, "turn": 0})
            try:
                with log.bind(task_id=task_id, trial=trial), tracing.span(
                        f"task {task_id}", {"shipwright.task_id": task_id, "shipwright.trial": trial,
                                            "shipwright.seed": seed}) as sp:
                    tracing.TASK_SPANS[(task_id, trial)] = tracing.current_ids()[1] or ""
                    start = time.perf_counter()
                    logger.info(f"task {task_id} start", seed=seed)
                    try:
                        result = real_run_unit(ctx, task, trial, seed, progress_str)
                    except Exception:
                        logger.error(f"task {task_id} crashed", exc_info=True)
                        raise
                    reward = getattr(getattr(result, "reward_info", None), "reward", None)
                    term = getattr(result, "termination_reason", None)
                    term = getattr(term, "value", term)
                    tracing.set_attrs(sp, {"shipwright.reward": reward, "shipwright.termination_reason": term})
                    if term == "infrastructure_error":
                        tracing.mark_error(sp, str((getattr(result, "info", None) or {}).get("error", term))[:300])
                    logger.info(f"task {task_id} end", reward=reward, termination_reason=term,
                                seconds=round(time.perf_counter() - start, 1))
                    return result
            finally:
                _task_state.reset(token)

        batch.run_unit = run_unit
    except Exception as e:  # noqa: BLE001
        logger.warning("tau2 task hook not installed", error=str(e))

    try:
        from tau2.agent.llm_agent import LLMAgent

        real_next = LLMAgent.generate_next_message

        def generate_next_message(self: Any, message: Any, state: Any) -> Any:
            st = _task_state.get()
            if st is not None:
                st["turn"] = st.get("turn", 0) + 1
            turn = st["turn"] if st is not None else None
            with log.bind(turn=turn), tracing.span("agent_turn", {"shipwright.turn": turn}):
                return real_next(self, message, state)

        LLMAgent.generate_next_message = generate_next_message
    except Exception as e:  # noqa: BLE001
        logger.warning("tau2 agent hook not installed", error=str(e))

    try:
        from tau2.environment.environment import Environment

        real_get_response = Environment.get_response

        def get_response(self: Any, message: Any) -> Any:
            name = getattr(message, "name", "?")
            start = time.perf_counter()
            with log.bind(component="tool"), tracing.span(
                    f"execute_tool {name}", {"gen_ai.tool.name": name, "gen_ai.operation.name": "execute_tool"}) as sp:
                log.fail_point("tool_call")
                resp = real_get_response(self, message)
                ms = round((time.perf_counter() - start) * 1000, 1)
                err = bool(getattr(resp, "error", False))
                if err:
                    tracing.mark_error(sp, str(getattr(resp, "content", ""))[:300])
                logger.info(f"tool {name}", tool=name, latency_ms=ms, status="tool_error" if err else "ok",
                            requestor=getattr(message, "requestor", None),
                            error=report.trim(getattr(resp, "content", None), 300) if err else None)
                return resp

        Environment.get_response = get_response
    except Exception as e:  # noqa: BLE001
        logger.warning("tau2 tool hook not installed", error=str(e))
