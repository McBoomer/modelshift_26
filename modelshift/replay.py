"""Replay a suite against a target model and score it."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .core import LLMResponse, last_user_text
from .diff import CaseResult, Check, compare
from .providers import Provider
from .suite import Case


def _run_one(case: Case, provider: Provider, model: str) -> CaseResult:
    try:
        actual = provider.complete(case.request, model)
    except Exception as e:  # surface provider errors as failures, don't crash the run
        r = CaseResult(
            case.id, case.request.system, last_user_text(case.request), case.expected, LLMResponse(text="")
        )
        r.checks.append(Check("error", False, f"{type(e).__name__}: {e}"))
        r.symptoms.append({"type": "error"})
        return r
    return compare(case, actual)


def replay(cases: list[Case], provider: Provider, model: str, workers: int = 4) -> list[CaseResult]:
    if workers <= 1:
        return [_run_one(c, provider, model) for c in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: _run_one(c, provider, model), cases))


def pass_rate(results: list[CaseResult]) -> float:
    return sum(r.passed for r in results) / len(results) if results else 1.0


def summarize(results: list[CaseResult]) -> dict:
    return {
        "total": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(pass_rate(results), 4),
        "failed_checks": dict(Counter(c.name.split(":")[0] for r in results for c in r.failed_checks)),
        "symptoms": dict(Counter(s["type"] for r in results for s in r.symptoms)),
    }
