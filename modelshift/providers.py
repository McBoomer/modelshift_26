"""Provider adapters. Everything speaks LLMRequest -> LLMResponse."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod

from .core import LLMRequest, LLMResponse


class Provider(ABC):
    name = "base"

    @abstractmethod
    def _complete(self, req: LLMRequest, model: str) -> LLMResponse: ...

    def complete(self, req: LLMRequest, model: str) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self._complete(req, model)
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        return resp


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, client=None):
        if client is None:
            from openai import OpenAI  # optional dependency

            client = OpenAI()
        self.client = client

    def _complete(self, req: LLMRequest, model: str) -> LLMResponse:
        messages = ([{"role": "system", "content": req.system}] if req.system else []) + req.messages
        kwargs: dict = {"model": model, "messages": messages}
        for k in ("temperature", "top_p", "seed", "stop"):
            if k in req.params:
                kwargs[k] = req.params[k]
        if "max_tokens" in req.params:
            kwargs["max_completion_tokens"] = req.params["max_tokens"]
        if req.tools:
            kwargs["tools"] = req.tools
        if req.response_format:
            kwargs["response_format"] = req.response_format
        r = self.client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append({"name": tc.function.name, "arguments": args})
        return LLMResponse(text=msg.content or "", tool_calls=calls)


def openai_tool_to_anthropic(tool: dict) -> dict:
    fn = tool.get("function", tool)
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, client=None):
        if client is None:
            from anthropic import Anthropic  # optional dependency

            client = Anthropic()
        self.client = client

    def _complete(self, req: LLMRequest, model: str) -> LLMResponse:
        kwargs: dict = {
            "model": model,
            "messages": req.messages,
            "max_tokens": req.params.get("max_tokens", 1024),
        }
        for k in ("temperature", "top_p"):
            if k in req.params:
                kwargs[k] = req.params[k]
        system = req.system or ""
        rf = req.response_format or {}
        if rf.get("type") == "json_schema":
            schema = rf.get("json_schema", {}).get("schema", {})
            system += "\n\nRespond with a JSON object matching this schema:\n" + json.dumps(schema)
        if system.strip():
            kwargs["system"] = system
        if req.tools:
            kwargs["tools"] = [openai_tool_to_anthropic(t) for t in req.tools]
        r = self.client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in r.content if b.type == "text")
        calls = [{"name": b.name, "arguments": dict(b.input)} for b in r.content if b.type == "tool_use"]
        return LLMResponse(text=text, tool_calls=calls)


def get_provider(name: str) -> Provider:
    name = name.lower()
    if name == "openai":
        return OpenAIProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name in ("mock-old", "mock-new"):
        from .mock import MockModelProvider

        return MockModelProvider(generation=name.split("-")[1])
    raise ValueError(f"Unknown provider '{name}' (use openai, anthropic, mock-old, mock-new)")
