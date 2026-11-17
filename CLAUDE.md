# CLAUDE.md: modelshift

## What this project is
modelshift migrates production LLM apps between models without silent regressions:
capture real traces -> build a regression suite -> replay on the target model -> diff behavior
-> auto-repair prompts -> report. The core bet is **automatic repair**. Measuring drift alone is
a commodity (Braintrust, Promptfoo), so repair quality is what we optimize.

## Current phase
Week 1: prove that auto-repair works on REAL models (see docs/WEEK1.md).
Do not build dashboards, auth, billing, or a web UI yet.

## Layout
- `modelshift/core.py`: LLMRequest / LLMResponse (provider-neutral; OpenAI-style messages, system separate)
- `modelshift/capture.py`: instrument_openai / instrument_anthropic / RecordingProvider
- `modelshift/trace.py`: SQLite TraceStore
- `modelshift/suite.py`: dedupe + behavior-signature sampling -> Case
- `modelshift/diff.py`: checks + machine-readable symptoms (code_fence, value_drift, refusal, ...)
- `modelshift/replay.py`: parallel replay + summarize
- `modelshift/repair.py`: Heuristic / LLM / Composite optimizers; train/holdout search
- `modelshift/report.py`: Markdown report
- `modelshift/providers.py`: OpenAI + Anthropic adapters, get_provider()
- `modelshift/mock.py`: offline drifting mock models (tests + demo only)
- `experiments/`: real-model experiments (apps, inputs, results). Never imported by the package.

## Commands
- Tests: `pytest -q` (must pass offline; never hit real APIs in unit tests)
- Lint: `ruff check . && ruff format --check .`
- Demo: `modelshift demo`
- Live smoke test (costs money): `python experiments/smoke.py`

## Rules
- The core package has zero required dependencies. openai/anthropic stay optional extras.
- Capture code must NEVER raise into the host app. Wrap everything and log instead.
- Every real-API path goes through the response cache (`modelshift/cache.py`) and the
  cost tracker. No uncached live calls in experiments.
- Never hard-code test inputs/answers into repaired prompts. The holdout split must never be
  shown to an optimizer.
- API keys come from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Never write keys
  to files, logs, or fixtures.
- Add or update tests for every behavior change. Prefer small, focused commits.
- When unsure about a design decision, write it in `docs/DECISIONS.md` with the tradeoff.
