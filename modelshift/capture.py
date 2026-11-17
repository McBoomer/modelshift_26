"""One-line instrumentation for production clients.

    from modelshift.capture import instrument_openai
    client = instrument_openai(OpenAI(), TraceStore("traces.db"))

Recording never raises into your app: failures are logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import random
import time

from .core import LLMRequest, LLMResponse, content_to_text
from .providers import Provider
from .trace import Trace, TraceStore

log = logging.getLogger("modelshift")

_PARAM_KEYS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "stop")


def _params(kwargs: dict) -> dict:
    p = {k: kwargs[k] for k in _PARAM_KEYS if k in kwargs}
    if "max_completion_tokens" in p:
        p["max_tokens"] = p.pop("max_completion_tokens")
    return p


def _record_openai(store, kwargs, result, latency_ms, tags):
    msgs = list(kwargs.get("messages", []))
    system_parts = [
        content_to_text(m.get("content")) for m in msgs if m.get("role") in ("system", "developer")
    ]
    rest = [m for m in msgs if m.get("role") not in ("system", "developer")]
    msg = result.choices[0].message
    calls = []
    for tc in getattr(msg, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        calls.append({"name": tc.function.name, "arguments": args})
    store.add(
        Trace(
            request=LLMRequest(
                messages=rest,
                system="\n\n".join(system_parts) or None,
                tools=kwargs.get("tools"),
                response_format=kwargs.get("response_format"),
                params=_params(kwargs),
            ),
            response=LLMResponse(text=msg.content or "", tool_calls=calls, latency_ms=latency_ms),
            provider="openai",
            model=kwargs.get("model", "unknown"),
            tags=dict(tags or {}),
        )
    )


def _record_anthropic(store, kwargs, result, latency_ms, tags):
    system = kwargs.get("system")
    if isinstance(system, list):
        system = content_to_text(system)
    tools = None
    if kwargs.get("tools"):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in kwargs["tools"]
        ]
    text = "".join(getattr(b, "text", "") for b in result.content if b.type == "text")
    calls = [{"name": b.name, "arguments": dict(b.input)} for b in result.content if b.type == "tool_use"]
    store.add(
        Trace(
            request=LLMRequest(
                messages=list(kwargs.get("messages", [])),
                system=system,
                tools=tools,
                params=_params(kwargs),
            ),
            response=LLMResponse(text=text, tool_calls=calls, latency_ms=latency_ms),
            provider="anthropic",
            model=kwargs.get("model", "unknown"),
            tags=dict(tags or {}),
        )
    )


def _wrap(owner, attr, recorder, store, tags, sample_rate):
    original = getattr(owner, attr)

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        if kwargs.get("stream") or random.random() > sample_rate:
            return result
        try:
            recorder(store, kwargs, result, (time.perf_counter() - t0) * 1000, tags)
        except Exception:  # never break production traffic
            log.exception("modelshift: failed to record trace")
        return result

    setattr(owner, attr, wrapped)


def instrument_openai(client, store: TraceStore, tags: dict | None = None, sample_rate: float = 1.0):
    _wrap(client.chat.completions, "create", _record_openai, store, tags, sample_rate)
    return client


def instrument_anthropic(client, store: TraceStore, tags: dict | None = None, sample_rate: float = 1.0):
    _wrap(client.messages, "create", _record_anthropic, store, tags, sample_rate)
    return client


class RecordingProvider(Provider):
    """Wrap any Provider so every call is stored as a trace."""

    def __init__(self, inner: Provider, store: TraceStore, tags: dict | None = None):
        self.inner = inner
        self.store = store
        self.tags = tags or {}
        self.name = inner.name

    def _complete(self, req: LLMRequest, model: str) -> LLMResponse:
        resp = self.inner.complete(req, model)
        self.store.add(
            Trace(request=req, response=resp, provider=self.inner.name, model=model, tags=self.tags)
        )
        return resp
