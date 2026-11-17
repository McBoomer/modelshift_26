"""Human-readable migration report (Markdown)."""

from __future__ import annotations

import difflib

from .diff import CaseResult
from .repair import RepairResult
from .replay import summarize


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def render_report(
    source_model: str,
    target_model: str,
    before: list[CaseResult],
    after: list[CaseResult] | None = None,
    repair: RepairResult | None = None,
    max_examples: int = 5,
) -> str:
    sb = summarize(before)
    lines = [
        f"# Migration report: `{source_model}` -> `{target_model}`",
        "",
        "| | Cases | Passing | Pass rate |",
        "|---|---|---|---|",
        f"| Before repair | {sb['total']} | {sb['passed']} | **{sb['pass_rate']:.0%}** |",
    ]
    if after is not None:
        sa = summarize(after)
        lines.append(f"| After repair | {sa['total']} | {sa['passed']} | **{sa['pass_rate']:.0%}** |")
    lines += ["", "## Behavior drift detected (before repair)", ""]
    if sb["symptoms"]:
        lines += ["| Symptom | Cases |", "|---|---|"]
        lines += [f"| `{k}` | {v} |" for k, v in sorted(sb["symptoms"].items(), key=lambda kv: -kv[1])]
    else:
        lines.append("No drift detected. Safe to switch.")

    fails = [r for r in before if not r.passed][:max_examples]
    if fails:
        lines += ["", "## Example regressions", ""]
        for f in fails:
            lines += [
                f"**Input:** {f.user_input[:200]}",
                "",
                "Expected:",
                "```",
                (f.expected.text or str(f.expected.tool_calls))[:500],
                "```",
                "Got:",
                "```",
                (f.actual.text or str(f.actual.tool_calls))[:500],
                "```",
                "Failed: " + ", ".join(f"`{c.name}` ({c.detail[:80]})" for c in f.failed_checks),
                "",
            ]

    if repair:
        lines += ["## Prompt repairs", ""]
        for i, r in enumerate(repair.repairs, 1):
            lines += [
                f"### Prompt {i} ({r.n_cases} cases)",
                f"Train: {_pct(r.train_before)} -> {_pct(r.train_after)} | "
                f"Holdout: {_pct(r.holdout_before)} -> {_pct(r.holdout_after)}",
                "",
            ]
            if r.changed:
                diff = difflib.unified_diff(
                    (r.original or "").splitlines(),
                    (r.repaired or "").splitlines(),
                    "original",
                    "repaired",
                    lineterm="",
                )
                lines += ["```diff", *diff, "```", ""]
            else:
                lines += ["No change needed.", ""]
    return "\n".join(lines) + "\n"
