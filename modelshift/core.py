"""Provider-neutral request/response types used everywhere in modelshift."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    """A normalized chat request (OpenAI-style messages, system prompt pulled out)."""

    messages: list[dict]
    system: str | None = None
    tools: list[dict] | None = None  # OpenAI function-tool format
    response_format: dict | None = None  # OpenAI response_format (json_schema supported)
    params: dict = field(default_factory=dict)  # temperature, max_tokens, ...

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LLMRequest:
        return cls(**d)

    def with_system(self, system: str | None) -> LLMRequest:
        clone = copy.deepcopy(self)
        clone.system = system
        return clone

    def fingerprint(self) -> str:
        """Stable hash of the request, used for de-duplication."""
        blob = json.dumps(
            {"m": self.messages, "s": self.system, "t": self.tools, "r": self.response_format},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": str, "arguments": dict}]
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LLMResponse:
        return cls(**d)


def short_hash(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()[:8]


def last_user_text(req: LLMRequest) -> str:
    for m in reversed(req.messages):
        if m.get("role") == "user":
            return content_to_text(m.get("content"))
    return ""


def content_to_text(content: Any) -> str:
    """Flatten OpenAI/Anthropic content (str or list of parts) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(part.get("text") or "")
            else:
                out.append(getattr(part, "text", "") or "")
        return "".join(out)
    return str(content)
