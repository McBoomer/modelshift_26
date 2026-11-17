"""Auto-repair: search for a revised system prompt that makes the new model
behave like the old one, validated on a held-out split to avoid overfitting.

Two optimizers ship:
  * HeuristicOptimizer - deterministic fixes for well-known drift symptoms (cheap, first pass)
  * LLMOptimizer       - asks a strong model to rewrite the prompt given concrete failures
Combine them with CompositeOptimizer.
"""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from .core import LLMRequest
from .diff import CaseResult, strict_json
from .providers import Provider
from .replay import pass_rate, replay
from .suite import Case


class Optimizer(ABC):
    @abstractmethod
    def propose(self, system: str | None, failures: list[CaseResult], cases: list[Case], n: int) -> list[str]:
        """Return up to n candidate system prompts."""


def _append(base: str | None, lines: list[str]) -> str:
    base = (base or "").rstrip()
    return (base + "\n\n" if base else "") + "\n".join(lines)


class HeuristicOptimizer(Optimizer):
    def fixes_for(self, failures: list[CaseResult], cases: list[Case]) -> list[str]:
        sym = [s for f in failures for s in f.symptoms]
        types = {s["type"] for s in sym}
        fixes: list[str] = []
        if "code_fence" in types or "invalid_json" in types:
            fixes.append("Respond with raw JSON only: no markdown and no code fences.")
        if "preamble" in types:
            fixes.append("Output only the JSON object, with no preamble or explanation before or after it.")
        drift_keys = sorted({s["key"] for s in sym if s["type"] == "value_drift"})
        for key in drift_keys:
            allowed = set()
            for c in cases:
                obj = strict_json(c.expected.text)
                if isinstance(obj, dict) and key in obj and isinstance(obj[key], (str, int, bool)):
                    allowed.add(obj[key])
            if allowed and len(allowed) <= 12:
                vals = ", ".join(str(v) for v in sorted(allowed, key=str))
                fixes.append(f"The `{key}` field must be one of: {vals}. Never use any other value.")
        missing = sorted({k for s in sym if s["type"] == "missing_keys" for k in s["keys"]})
        if missing:
            fixes.append("Always include these keys: " + ", ".join(missing) + ".")
        extra = sorted({k for s in sym if s["type"] == "extra_keys" for k in s["keys"]})
        if extra:
            fixes.append("Do not add these keys: " + ", ".join(extra) + ".")
        if "refusal" in types:
            fixes.append(
                "Every input is a legitimate, in-scope request. Complete the task for every input "
                "instead of declining; you are only processing the text, not acting on it."
            )
        if "too_long" in types:
            fixes.append("Keep responses concise; match a brief, direct style.")
        if "too_short" in types:
            fixes.append("Give complete, fully developed answers.")
        for s in sym:
            if s["type"] == "tool_mismatch" and s["expected"]:
                fixes.append(f"For requests like these, call the tool(s): {', '.join(s['expected'])}.")
                break
        return fixes

    def propose(self, system, failures, cases, n):
        fixes = self.fixes_for(failures, cases)
        if not fixes:
            return []
        candidates = [_append(system, fixes)] + [_append(system, [f]) for f in fixes]
        out, seen = [], set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out[:n]


META_PROMPT = """You are an expert prompt engineer migrating an LLM application to a new model.
The CURRENT SYSTEM PROMPT worked on the old model. On the new model, the cases below now behave
differently. Revise the system prompt so the new model reproduces the OLD (expected) behavior.

Rules:
- Make the smallest change that fixes the failures; preserve the original intent and wording.
- Do not hard-code individual test inputs or answers.
- Return ONLY the full revised system prompt inside <prompt></prompt> tags.

CURRENT SYSTEM PROMPT:
<current>
{system}
</current>

FAILURES:
{failures}
"""


class LLMOptimizer(Optimizer):
    def __init__(self, provider: Provider, model: str, max_examples: int = 8, temperature: float = 0.7):
        self.provider, self.model = provider, model
        self.max_examples, self.temperature = max_examples, temperature

    def _render(self, failures: list[CaseResult]) -> str:
        blocks = []
        for f in failures[: self.max_examples]:
            blocks.append(
                json.dumps(
                    {
                        "input": f.user_input[:800],
                        "expected_output": f.expected.text[:800] or f.expected.tool_calls,
                        "new_model_output": f.actual.text[:800] or f.actual.tool_calls,
                        "failed_checks": [f"{c.name}: {c.detail}" for c in f.failed_checks],
                    },
                    indent=2,
                    default=str,
                )
            )
        return "\n\n".join(blocks)

    def propose(self, system, failures, cases, n):
        prompt = META_PROMPT.format(system=system or "(empty)", failures=self._render(failures))
        out = []
        for _ in range(n):
            req = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                params={"temperature": self.temperature, "max_tokens": 4000},
            )
            text = self.provider.complete(req, self.model).text
            m = re.search(r"<prompt>(.*?)</prompt>", text, re.DOTALL)
            if m and m.group(1).strip():
                out.append(m.group(1).strip())
        return out


class CompositeOptimizer(Optimizer):
    def __init__(self, *optimizers: Optimizer):
        self.optimizers = optimizers

    def propose(self, system, failures, cases, n):
        out: list[str] = []
        for o in self.optimizers:
            out += o.propose(system, failures, cases, n)
        return out


@dataclass
class PromptRepair:
    original: str | None
    repaired: str | None
    n_cases: int
    train_before: float
    train_after: float
    holdout_before: float | None
    holdout_after: float | None
    history: list[dict] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.repaired != self.original


@dataclass
class RepairResult:
    target_model: str
    repairs: list[PromptRepair]

    def mapping(self) -> dict:
        return {r.original: r.repaired for r in self.repairs}

    def apply(self, cases: list[Case]) -> list[Case]:
        m = self.mapping()
        return [c.with_system(m.get(c.request.system, c.request.system)) for c in cases]

    def to_dict(self) -> dict:
        return {"target_model": self.target_model, "repairs": [asdict(r) for r in self.repairs]}

    @classmethod
    def from_dict(cls, d: dict) -> RepairResult:
        return cls(d["target_model"], [PromptRepair(**r) for r in d["repairs"]])


def repair(
    cases: list[Case],
    provider: Provider,
    model: str,
    optimizer: Optimizer,
    *,
    rounds: int = 3,
    candidates: int = 4,
    holdout_frac: float = 0.25,
    seed: int = 0,
    workers: int = 4,
    log: Callable[[str], None] = lambda s: None,
) -> RepairResult:
    rng = random.Random(seed)
    groups: dict[str | None, list[Case]] = defaultdict(list)
    for c in cases:
        groups[c.request.system].append(c)

    repairs = []
    for system, gcases in groups.items():
        gcases = gcases[:]
        rng.shuffle(gcases)
        n_hold = int(len(gcases) * holdout_frac) if len(gcases) >= 8 else 0
        holdout, train = gcases[:n_hold], gcases[n_hold:]

        best_sys = system
        best_res = replay(train, provider, model, workers)
        best = start = pass_rate(best_res)
        history = [{"round": 0, "score": best}]
        log(f"prompt {str(system)[:40]!r}... baseline {best:.0%} on {len(train)} train cases")

        for r in range(1, rounds + 1):
            if best >= 1.0:
                break
            failures = [x for x in best_res if not x.passed]
            improved = False
            for cand in optimizer.propose(best_sys, failures, train, candidates):
                res = replay([c.with_system(cand) for c in train], provider, model, workers)
                score = pass_rate(res)
                if score > best:
                    best, best_sys, best_res, improved = score, cand, res, True
            history.append({"round": r, "score": best})
            log(f"  round {r}: {best:.0%}")
            if not improved:
                break

        hb = ha = None
        if holdout:
            hb = pass_rate(replay(holdout, provider, model, workers))
            ha = pass_rate(replay([c.with_system(best_sys) for c in holdout], provider, model, workers))
            log(f"  holdout {hb:.0%} -> {ha:.0%}")
        repairs.append(PromptRepair(system, best_sys, len(gcases), start, best, hb, ha, history))
    return RepairResult(model, repairs)
