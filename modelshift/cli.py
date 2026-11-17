"""modelshift CLI.

  modelshift demo                                   # full offline walkthrough
  modelshift suite  --db traces.db --out suite.jsonl
  modelshift replay --suite suite.jsonl --provider anthropic --model <new-model>
  modelshift repair --suite suite.jsonl --provider anthropic --model <new-model> \
                    --optimizer-provider anthropic --optimizer-model <strong-model>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .providers import get_provider
from .repair import CompositeOptimizer, HeuristicOptimizer, LLMOptimizer, repair
from .replay import replay, summarize
from .report import render_report
from .suite import build_suite, load_suite, save_suite
from .trace import TraceStore


def cmd_suite(a):
    store = TraceStore(a.db)
    traces = store.all(model=a.source_model)
    cases = build_suite(traces, max_cases=a.max, seed=a.seed)
    save_suite(cases, a.out)
    print(
        f"{len(traces)} traces -> {len(cases)} cases "
        f"({len({c.signature for c in cases})} behavior groups) -> {a.out}"
    )


def cmd_replay(a):
    cases = load_suite(a.suite)
    results = replay(cases, get_provider(a.provider), a.model, a.workers)
    s = summarize(results)
    Path(a.out).write_text(json.dumps({"summary": s, "results": [r.to_dict() for r in results]}, indent=2))
    print(json.dumps(s, indent=2))
    if a.report:
        src = cases[0].source_model if cases else "?"
        Path(a.report).write_text(render_report(src, a.model, results))
        print(f"report -> {a.report}")
    if a.min_pass is not None and s["pass_rate"] < a.min_pass:
        sys.exit(1)  # CI gate


def cmd_repair(a):
    cases = load_suite(a.suite)
    provider = get_provider(a.provider)
    opt = HeuristicOptimizer()
    if a.optimizer_provider:
        opt = CompositeOptimizer(opt, LLMOptimizer(get_provider(a.optimizer_provider), a.optimizer_model))
    before = replay(cases, provider, a.model, a.workers)
    rep = repair(cases, provider, a.model, opt, rounds=a.rounds, workers=a.workers, log=print)
    after = replay(rep.apply(cases), provider, a.model, a.workers)
    Path(a.out).write_text(json.dumps(rep.to_dict(), indent=2))
    src = cases[0].source_model if cases else "?"
    Path(a.report).write_text(render_report(src, a.model, before, after, rep))
    print(f"before {summarize(before)['pass_rate']:.0%} -> after {summarize(after)['pass_rate']:.0%}")
    print(f"repairs -> {a.out}   report -> {a.report}")


def cmd_demo(a):
    from .demo import run_demo

    run_demo(Path(a.dir))


def main(argv=None):
    p = argparse.ArgumentParser(prog="modelshift", description="Migrate LLM apps between models safely.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suite", help="build a regression suite from captured traces")
    s.add_argument("--db", default="modelshift_traces.db")
    s.add_argument("--out", default="suite.jsonl")
    s.add_argument("--max", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--source-model", default=None, help="only use traces from this model")
    s.set_defaults(fn=cmd_suite)

    r = sub.add_parser("replay", help="replay suite on a target model and diff behavior")
    r.add_argument("--suite", default="suite.jsonl")
    r.add_argument("--provider", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", default="results.json")
    r.add_argument("--report", default=None)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--min-pass", type=float, default=None, help="exit 1 if pass rate is below this (CI)")
    r.set_defaults(fn=cmd_replay)

    f = sub.add_parser("repair", help="auto-repair prompts for the target model")
    f.add_argument("--suite", default="suite.jsonl")
    f.add_argument("--provider", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--optimizer-provider", default=None)
    f.add_argument("--optimizer-model", default=None)
    f.add_argument("--rounds", type=int, default=3)
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--out", default="repairs.json")
    f.add_argument("--report", default="migration_report.md")
    f.set_defaults(fn=cmd_repair)

    d = sub.add_parser("demo", help="run the full pipeline offline with mock models")
    d.add_argument("--dir", default="modelshift_demo")
    d.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
