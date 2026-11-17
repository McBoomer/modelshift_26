"""Turn raw production traces into a balanced regression suite."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass

from .core import LLMRequest, LLMResponse, short_hash
from .diff import strict_json
from .trace import Trace


@dataclass
class Case:
    id: str
    request: LLMRequest
    expected: LLMResponse
    signature: str
    source_model: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "request": self.request.to_dict(),
            "expected": self.expected.to_dict(),
            "signature": self.signature,
            "source_model": self.source_model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Case:
        return cls(
            id=d["id"],
            request=LLMRequest.from_dict(d["request"]),
            expected=LLMResponse.from_dict(d["expected"]),
            signature=d["signature"],
            source_model=d["source_model"],
        )

    def with_system(self, system: str | None) -> Case:
        return Case(
            self.id, self.request.with_system(system), self.expected, self.signature, self.source_model
        )


def _length_bucket(n: int) -> str:
    for limit, label in ((80, "xs"), (400, "s"), (1500, "m"), (5000, "l")):
        if n < limit:
            return label
    return "xl"


def signature(trace: Trace) -> str:
    """Behavioral fingerprint: which prompt, and what *shape* of output it produced."""
    parts = ["sys:" + short_hash(trace.request.system)]
    resp = trace.response
    if resp.tool_calls:
        parts.append("tools:" + ",".join(c["name"] for c in resp.tool_calls))
    else:
        obj = strict_json(resp.text)
        if isinstance(obj, dict):
            parts.append("json:" + ",".join(sorted(obj.keys())))
        elif obj is not None:
            parts.append("json:list")
        else:
            parts.append("text:" + _length_bucket(len(resp.text)))
    return "|".join(parts)


def build_suite(traces: list[Trace], max_cases: int = 200, seed: int = 0) -> list[Case]:
    """De-duplicate, group by behavioral signature, then round-robin sample so rare
    behaviors (tool calls, edge-case outputs) are not drowned out by the happy path."""
    rng = random.Random(seed)
    seen: set[str] = set()
    groups: dict[str, list[Trace]] = defaultdict(list)
    for t in traces:
        fp = t.request.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        groups[signature(t)].append(t)

    for g in groups.values():
        rng.shuffle(g)

    cases: list[Case] = []
    keys = sorted(groups)
    while len(cases) < max_cases and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(cases) < max_cases:
                t = groups[k].pop()
                cases.append(Case(t.id, t.request, t.response, k, t.model))
    return cases


def save_suite(cases: list[Case], path: str) -> None:
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict(), default=str) + "\n")


def load_suite(path: str) -> list[Case]:
    with open(path) as f:
        return [Case.from_dict(json.loads(line)) for line in f if line.strip()]
