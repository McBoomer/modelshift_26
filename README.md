# modelshift

**Migrate production LLM apps between models without silent regressions.**

When a provider retires a model, your prompts, tool definitions and output schemas were tuned
for the old one. Swap the model string and things break quietly: JSON gets wrapped in markdown,
enum values drift, refusals spike, and tool calls change. modelshift captures real traffic, turns it
into a regression suite, measures exactly how the new model behaves differently, and **repairs
the prompts automatically**. Every repair is checked on held-out cases.

```
capture  ->  suite  ->  replay + diff  ->  repair  ->  report / CI gate
```

## Try it in 10 seconds (no API keys)

```bash
pip install -e ".[dev]"
modelshift demo
```

The demo moves a ticket-triage app from a mock "old" model to a mock "new" model that has
real-world quirks (code fences, chatty preamble, `high -> critical` enum drift, over-refusal):

```
3) Replaying on the new model...
   pass rate 0%  symptoms {'code_fence': 35, 'preamble': 35, 'refusal': 5, 'value_drift': 9}
4) Auto-repairing prompt...
  round 1: 100%
  holdout 0% -> 100%
```

The report in `modelshift_demo/migration_report.md` shows the drift found and the prompt diff that fixed it.

## Use it on a real app

**1. Capture traces.** Add one line in production. It never raises into your app, skips streaming calls, and supports sampling.

```python
from openai import OpenAI
from modelshift import TraceStore, instrument_openai  # or instrument_anthropic

client = instrument_openai(OpenAI(), TraceStore("traces.db"), sample_rate=0.2)
```

**2. Build the suite.** De-duplicates requests and samples evenly across behavior groups (tool calls, JSON
shapes, text lengths), so rare paths aren't drowned out.

```bash
modelshift suite --db traces.db --out suite.jsonl --max 300
```

**3. Measure drift on the target model.**

```bash
modelshift replay --suite suite.jsonl --provider anthropic --model <new-model> --report drift.md
# in CI: add --min-pass 0.95 to fail the build on regressions
```

**4. Auto-repair.** Rule-based fixes run first. Add an LLM optimizer for everything else.

```bash
modelshift repair --suite suite.jsonl --provider anthropic --model <new-model> \
  --optimizer-provider anthropic --optimizer-model <strong-model>
```

The output is `repairs.json` (old prompt -> new prompt) and `migration_report.md`.

## What it checks

| Check | Catches |
|---|---|
| `refusal` | New model declines things the old one handled |
| `tool_names` / `tool_args` | Different tools called, or different argument keys |
| `json_valid` | Code fences, preambles, broken JSON |
| `json_keys` / `json_values` | Missing or extra keys, enum and label drift |
| `schema` | Violations of the request's `response_format` JSON schema |
| `length` / `similarity` | Free-text verbosity and content drift |

## Layout

```
modelshift/
  core.py       request/response types
  capture.py    instrument_openai / instrument_anthropic / RecordingProvider
  trace.py      SQLite trace store
  suite.py      dedupe + behavior-signature sampling
  diff.py       checks + machine-readable symptoms
  replay.py     parallel replay + scoring
  repair.py     Heuristic / LLM / Composite optimizers, train/holdout search
  report.py     Markdown migration report
  providers.py  OpenAI + Anthropic adapters
  mock.py       offline drifting models for demo and tests
```

## Roadmap

- [ ] LLM-as-judge semantic check for free text (alongside Jaccard)
- [ ] Repair tool descriptions and JSON schemas, not just system prompts
- [ ] Multi-turn and agent traces (tool-result messages, full trajectories)
- [ ] Shadow traffic + canary rollout with automatic rollback
- [ ] Deprecation-calendar watcher that opens migration PRs before sunset dates
- [ ] Hosted dashboard + GitHub app
- [ ] Cross-customer "what breaks from model X to model Y" knowledge base (the moat)
