"""End-to-end offline demo: capture -> suite -> replay -> repair -> report."""

from __future__ import annotations

import json
from pathlib import Path

from .capture import RecordingProvider
from .core import LLMRequest
from .mock import TRIAGE_SYSTEM, MockModelProvider, sample_tickets
from .repair import HeuristicOptimizer, repair
from .replay import replay, summarize
from .report import render_report
from .suite import build_suite, save_suite
from .trace import TraceStore


def run_demo(outdir: Path, quiet: bool = False) -> dict:
    say = (lambda *a: None) if quiet else print
    outdir.mkdir(parents=True, exist_ok=True)
    db = outdir / "traces.db"
    if db.exists():
        db.unlink()

    say("1) Capturing 'production' traffic on the old model...")
    store = TraceStore(str(db))
    prod = RecordingProvider(MockModelProvider("old"), store)
    for ticket in sample_tickets(40):
        prod.complete(
            LLMRequest(
                system=TRIAGE_SYSTEM,
                messages=[{"role": "user", "content": ticket}],
                params={"temperature": 0},
            ),
            "mock-old",
        )
    say(f"   {store.count()} traces recorded")

    say("2) Building regression suite...")
    cases = build_suite(store.all(), max_cases=200)
    save_suite(cases, str(outdir / "suite.jsonl"))
    say(f"   {len(cases)} cases in {len({c.signature for c in cases})} behavior groups")

    new = MockModelProvider("new")
    say("3) Replaying on the new model...")
    before = replay(cases, new, "mock-new")
    sb = summarize(before)
    say(f"   pass rate {sb['pass_rate']:.0%}  symptoms {sb['symptoms']}")

    say("4) Auto-repairing prompt...")
    rep = repair(cases, new, "mock-new", HeuristicOptimizer(), log=say)
    after = replay(rep.apply(cases), new, "mock-new")
    sa = summarize(after)
    say(f"   pass rate {sb['pass_rate']:.0%} -> {sa['pass_rate']:.0%}")

    (outdir / "repairs.json").write_text(json.dumps(rep.to_dict(), indent=2))
    (outdir / "migration_report.md").write_text(render_report("mock-old", "mock-new", before, after, rep))
    say(f"5) Report written to {outdir / 'migration_report.md'}")
    store.close()
    return {"before": sb, "after": sa, "repair": rep}
