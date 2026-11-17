"""Token/cost accounting with a hard budget guard."""

from __future__ import annotations

import threading

from .core import LLMRequest, LLMResponse
from .providers import Provider

# USD per 1M tokens: model -> (input, output). Edit to match current provider pricing
# for every model in experiments/config.yaml. Unknown models are refused, never guessed.
PRICES: dict[str, tuple[float, float]] = {
    # "model-id": (input_usd_per_1m, output_usd_per_1m),
}


class BudgetExceeded(RuntimeError):
    pass


class CostTracker(Provider):
    """Wrap a live Provider; records tokens/cost per call and refuses calls once over budget.

    The call that crosses the budget still returns (its money is already spent, so the
    response should reach the cache); every call after that raises BudgetExceeded.
    """

    # shortcut: spend is per-process, not persisted across scripts; add a ledger file if the
    # weekly cap must hold across separate runs.

    def __init__(self, inner: Provider, budget_usd: float, prices: dict | None = None):
        self.inner = inner
        self.name = inner.name
        self.budget_usd = budget_usd
        self.prices = PRICES if prices is None else prices
        self.calls: list[dict] = []
        self.total_usd = 0.0
        self._lock = threading.Lock()

    def complete(self, req: LLMRequest, model: str) -> LLMResponse:
        if model not in self.prices:
            raise KeyError(f"No price for model '{model}'; add it to modelshift.cost.PRICES")
        with self._lock:
            if self.total_usd >= self.budget_usd:
                raise BudgetExceeded(f"spent ${self.total_usd:.4f} of ${self.budget_usd:.2f} budget")
        resp = self.inner.complete(req, model)
        p_in, p_out = self.prices[model]
        tin, tout = resp.input_tokens or 0, resp.output_tokens or 0
        usd = (tin * p_in + tout * p_out) / 1_000_000
        with self._lock:
            self.total_usd += usd
            self.calls.append({"model": model, "input_tokens": tin, "output_tokens": tout, "usd": usd})
        return resp

    _complete = complete
