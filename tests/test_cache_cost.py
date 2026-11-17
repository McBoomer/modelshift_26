from types import SimpleNamespace

import pytest

from modelshift.cache import CachedProvider
from modelshift.core import LLMRequest, LLMResponse
from modelshift.cost import BudgetExceeded, CostTracker
from modelshift.providers import AnthropicProvider, OpenAIProvider, Provider


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, tin=1000, tout=500):
        self.calls = 0
        self.tin, self.tout = tin, tout

    def _complete(self, req, model):
        self.calls += 1
        return LLMResponse(text=f"answer {self.calls}", input_tokens=self.tin, output_tokens=self.tout)


def _req(**params):
    return LLMRequest(messages=[{"role": "user", "content": "hi"}], system="sys", params=params)


def test_cache_hit_skips_inner_and_persists(tmp_path):
    inner = FakeProvider()
    c = CachedProvider(inner, tmp_path / "c.db")
    first = c.complete(_req(), "m")
    assert c.complete(_req(), "m").text == first.text
    assert inner.calls == 1 and c.hits == 1
    c.close()
    # A fresh instance on the same file still hits.
    c2 = CachedProvider(inner, tmp_path / "c.db")
    assert c2.complete(_req(), "m").text == first.text
    assert inner.calls == 1


def test_cache_key_covers_model_and_params(tmp_path):
    inner = FakeProvider()
    c = CachedProvider(inner, tmp_path / "c.db")
    c.complete(_req(), "m")
    c.complete(_req(), "other-model")
    c.complete(_req(temperature=0), "m")
    assert inner.calls == 3


def test_cache_does_not_store_errors(tmp_path):
    class Boom(Provider):
        name = "boom"

        def _complete(self, req, model):
            raise RuntimeError("api down")

    c = CachedProvider(Boom(), tmp_path / "c.db")
    with pytest.raises(RuntimeError):
        c.complete(_req(), "m")
    assert c.misses == 0


def test_cost_tracker_prices_and_budget():
    t = CostTracker(FakeProvider(tin=1_000_000, tout=1_000_000), budget_usd=5, prices={"m": (1.0, 2.0)})
    t.complete(_req(), "m")
    assert t.total_usd == pytest.approx(3.0)
    t.complete(_req(), "m")  # crosses the budget but still returns
    assert t.total_usd == pytest.approx(6.0)
    with pytest.raises(BudgetExceeded):
        t.complete(_req(), "m")
    assert len(t.calls) == 2


def test_cost_tracker_refuses_unpriced_model():
    inner = FakeProvider()
    with pytest.raises(KeyError):
        CostTracker(inner, budget_usd=1, prices={}).complete(_req(), "unknown")
    assert inner.calls == 0


def test_cache_hits_are_not_charged(tmp_path):
    t = CostTracker(FakeProvider(tin=1_000_000, tout=0), budget_usd=100, prices={"m": (1.0, 0.0)})
    c = CachedProvider(t, tmp_path / "c.db")
    c.complete(_req(), "m")
    c.complete(_req(), "m")
    assert t.total_usd == pytest.approx(1.0)


def test_openai_usage_populates_tokens():
    r = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: r)))
    resp = OpenAIProvider(client).complete(_req(), "m")
    assert (resp.input_tokens, resp.output_tokens) == (12, 3)


def test_anthropic_usage_populates_tokens():
    r = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=20, output_tokens=7),
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: r))
    resp = AnthropicProvider(client).complete(_req(), "m")
    assert (resp.input_tokens, resp.output_tokens) == (20, 7)
