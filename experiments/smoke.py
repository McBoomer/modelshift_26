"""Live smoke test: one JSON-extraction and one tool-calling request per configured model.

Costs real money (well under $0.10). Re-runs are free thanks to the cache.
Usage: python experiments/smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from modelshift.cache import CachedProvider
from modelshift.core import LLMRequest
from modelshift.cost import CostTracker
from modelshift.providers import get_provider

HERE = Path(__file__).parent
SMOKE_BUDGET_USD = 0.10

JSON_REQ = LLMRequest(
    system="Extract the invoice fields.",
    messages=[{"role": "user", "content": "Invoice from ACME Corp, dated 3 March 2024. Total: $41.50"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "invoice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}},
                "required": ["vendor", "total"],
                "additionalProperties": False,
            },
        },
    },
    params={"max_tokens": 200, "temperature": 0},
)

TOOL_REQ = LLMRequest(
    system="You are a support agent. Use tools when needed.",
    messages=[{"role": "user", "content": "Where is my order 12345?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "search_orders",
                "description": "Look up an order by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ],
    params={"max_tokens": 200, "temperature": 0},
)


def configured_models(cfg: dict) -> list[str]:
    names = [*cfg.get("source_models", []), *cfg.get("target_models", [])]
    names += [cfg.get("optimizer_model"), cfg.get("judge_model")]
    return list(dict.fromkeys(n for n in names if n and n != "TODO"))


def main() -> int:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    models = configured_models(cfg)
    if not models:
        print("No models configured: fill in experiments/config.yaml (format provider:model).")
        return 1
    failed = 0
    trackers = []
    for spec in models:
        provider_name, _, model = spec.partition(":")
        print(f"\n=== {spec}")
        try:
            tracker = CostTracker(get_provider(provider_name), budget_usd=SMOKE_BUDGET_USD)
            trackers.append(tracker)
            p = CachedProvider(tracker, HERE / ".cache" / "smoke.db")
            for label, req in (("json", JSON_REQ), ("tool", TOOL_REQ)):
                r = p.complete(req, model)
                print(f"[{label}] text={r.text!r}")
                print(f"[{label}] tool_calls={r.tool_calls} tokens={r.input_tokens}/{r.output_tokens}")
            print(f"cost so far for {spec}: ${tracker.total_usd:.5f}")
        except Exception as e:  # report every model, don't stop at the first failure
            failed += 1
            print(f"FAILED: {type(e).__name__}: {e}")
    print(f"\nTotal live spend: ${sum(t.total_usd for t in trackers):.5f}  failures: {failed}/{len(models)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
