# Week 1: Prove auto-repair works on real models

**One question this week answers:** when a real app moves from one real model to another,
can modelshift automatically recover most of the lost behavior without a human editing the prompt?

**Kill/continue threshold (decide on Day 6):**
- Continue if the LLM optimizer raises the *holdout* pass rate by 15+ points on at least 2 of 3 apps
  and lands at 90%+ on at least one.
- Pivot if gains are under 10 points, or come only from the trivial heuristic fixes.

**Budget:** cap live API spend at about $30 for the week (the cost tracker enforces it).
**Model pairs:** pick (A) an older -> newer model from the *same* provider, and (B) a *cross-provider*
pair. Write them into `experiments/config.yaml` on Day 1 so every prompt below can refer to them.

---

## How to work with Claude Code each day
1. Start the session with: `Read CLAUDE.md and docs/WEEK1.md. Today is Day N.`
2. Paste the day's prompt. For anything over ~1 hour of work, ask for a plan first
   (`Plan this before writing code`), review it, then approve.
3. End each task with: `Run the tests and lint, fix failures, then commit with a clear message.`
4. Before closing: `Update docs/PROGRESS.md with what was done, what's blocked, and tomorrow's first step.`

---

## Day 1: Repo hygiene + live adapters actually work  (~3 to 4 h)

**Prompt 1: setup**
```
Set up this repo for real development:
- git init (if needed), add a GitHub Actions workflow that runs `pytest -q` and `ruff check` on
  Python 3.10 and 3.12.
- Add ruff to the dev extra and a [tool.ruff] config (line length 110). Fix all lint issues.
- Create experiments/config.yaml with fields: budget_usd, source_models, target_models,
  optimizer_model, judge_model. Leave model names as TODO placeholders for me to fill in.
- Create docs/PROGRESS.md and docs/DECISIONS.md (empty templates).
Commit.
```

**Prompt 2: cost tracking + response cache**
```
Add modelshift/cache.py and modelshift/cost.py.
- CachedProvider(inner, path): wraps any Provider; caches LLMResponse on disk (SQLite) keyed by
  sha256 of (provider name, model, full request incl. params). Cache hits return instantly.
- CostTracker: records input/output tokens per call. Extend LLMResponse with optional
  input_tokens/output_tokens, and populate them in OpenAIProvider and AnthropicProvider from the
  API usage fields. Keep a price table in a config dict (USD per 1M tokens) that I can edit.
  Raise BudgetExceeded when the running total passes budget_usd.
- Unit tests with fake providers only (no network).
```

**Prompt 3: live smoke test**
```
Write experiments/smoke.py: for each model in experiments/config.yaml, send one tiny JSON
extraction request and one tool-calling request through CachedProvider + CostTracker using the
real OpenAI/Anthropic adapters. Print text, parsed tool calls, tokens, and cost. Fix any adapter
bugs you find (message format, tool format conversion, max_tokens naming, response_format
handling for Anthropic). Add regression unit tests for each bug using fake clients.
```

**Done when:** CI is green, and `python experiments/smoke.py` works for every configured model for under $0.10.

---

## Day 2: Three realistic apps + real traces  (~4 h)

**Prompt**
```
Create three small but realistic benchmark apps under experiments/apps/, each a module exposing
SYSTEM_PROMPT, an optional TOOLS / RESPONSE_FORMAT, and build_request(input) -> LLMRequest:

1. invoice_extract: extract vendor, date (ISO), total (number), currency, line_items[] from
   messy invoice/receipt text. Uses a json_schema response_format.
2. support_agent: single-turn agent with tools search_orders(order_id), issue_refund(order_id,
   amount, reason), escalate(reason). Must choose the right tool or answer directly.
3. ticket_reply: free-text customer-support reply in a specific brand voice (concise, no emojis,
   always ends with a next step), given a ticket + short policy snippet.

Then write experiments/gen_inputs.py that uses the optimizer model to generate 120 diverse
inputs per app into experiments/data/<app>/inputs.jsonl. Include roughly 25% hard cases: ambiguous,
adversarial, very long, multilingual, missing fields, prompt-injection attempts, and requests
that should NOT call a tool. Deduplicate near-identical inputs. Commit the inputs (they're data).

Then write experiments/capture.py: run every input through each SOURCE model via
RecordingProvider + CachedProvider, storing traces in experiments/data/<app>/<model>.db.
```

**Done when:** 3 apps × 120 inputs exist, and traces are captured on each source model. Spot-check 10 outputs per app by hand. If the source model itself is bad at the task, fix the app prompt *now*, because the old model's output is the gold standard.

---

## Day 3: Trustworthy diffing for free text + tools  (~4 h)

**Prompt**
```
Free-text diffing (Jaccard) is too crude. Add an LLM judge to modelshift/diff.py:
- Judge(provider, model): given the input, expected (old-model) output, and actual output, returns
  {"equivalent": bool, "reason": str, "dimensions": {"content","tone","format","policy"}}
  using a rubric prompt with strict JSON output. Cache all judge calls.
- compare() gets an optional judge=; when present, free-text cases use the judge instead of
  similarity/length, and failures emit symptoms by dimension (tone_drift, format_drift, ...).
- Tool calls: besides names/arg keys, compare argument VALUES for categorical args, and treat
  "called a tool vs answered directly" as its own symptom (tool_overuse / tool_underuse).
- Write experiments/judge_calibration.py: sample 40 (expected, actual) pairs from
  support_agent and ticket_reply on a target model, print them for me to label by hand in a
  CSV, then report the judge's agreement with my labels.
Unit tests use a fake judge.
```

**Done when:** you've hand-labeled 40 pairs and the judge agrees with you on at least 85% of them. If it doesn't, tune the rubric before moving on. A judge you can't trust makes every later number meaningless.

---

## Day 4: Make the LLM optimizer actually good  (~5 h)

**Prompt (start in plan mode)**
```
Plan, then implement, a stronger repair search in modelshift/repair.py:
1. Failure clustering: group failures by symptom type + failed check, and show the optimizer
   2 to 3 representative examples per cluster instead of the first N failures.
2. The optimizer prompt also receives: tool definitions and response schema (read-only for now),
   a summary of which clusters are failing and how often, and the previous attempts with their scores
   (so it doesn't repeat failed ideas).
3. Search: per round, generate K candidates (default 4), score each on a random 50% minibatch of
   train first and fully evaluate only the top 2 (saves cost). Keep the best. Stop early when
   train pass rate stops improving for 2 rounds or the budget is hit.
4. Regularization: reject candidates that grow the prompt by more than 3x, or that contain substrings
   copied from any train input longer than 30 chars (anti-memorization).
5. Record a full trace of the search (candidates, scores, cost) into the RepairResult.
Tests: with mock models, the LLM optimizer path (using a scripted fake optimizer) must
improve the score and must never see holdout cases (assert this).
```

**Done when:** tests pass and one manual run on `invoice_extract` for pair A completes under $3.

---

## Day 5: Run the full experiment  (~3 h, mostly waiting)

**Prompt**
```
Write experiments/run.py: for each app × model pair in config.yaml, run these arms with a fixed
70/30 train/holdout split (seed 0):
  baseline (no repair) | heuristic only | LLM optimizer only | heuristic + LLM
Record per arm: train and holdout pass rate, symptoms before and after, prompt length growth, $ spent,
wall time. Save to experiments/results/<date>/results.json and print a summary table.
Support --dry-run (cost estimate only) and resume-from-cache so a crash doesn't waste money.
```

Run `--dry-run` first, check the estimate fits the budget, then run it for real.

**Done when:** `results.json` covers all apps × pairs × arms.

---

## Day 6: Analyze + decide  (~3 h)

**Prompt**
```
Write experiments/analyze.py that turns results.json into experiments/results/<date>/RESULTS.md:
- Table: app × pair × arm with holdout pass rates (the headline metric), and the delta vs baseline.
- Chart (matplotlib PNG): baseline vs best arm per app/pair.
- Top 5 drift symptoms overall, with one real before/after example each.
- The repaired prompt diffs for the best arm per app.
- An honest "where repair failed" section listing the remaining failure clusters.
Then read the results and give me your assessment against the kill/continue threshold
in docs/WEEK1.md. Be blunt.
```

**You do:** read 20 repaired holdout cases yourself. Check that the repairs are real, not artifacts of the judge. Then make the call and write it in `docs/DECISIONS.md`.

---

## Day 7: Turn results into distribution  (~3 h)

**Prompt 1: blog post draft**
```
Using RESULTS.md, draft docs/blog/what-breaks-when-you-switch-models.md (800 to 1200 words):
the problem, the 3 apps, what drifted (with real examples), how much auto-repair recovered,
the limitations, and how to try modelshift. No hype; numbers first.
```

**Prompt 2: find design partners**
```
Write experiments/find_teams.py using the GitHub search API (token from GITHUB_TOKEN env) to
find public repos updated in the last 90 days whose code references the SOURCE model strings
from config.yaml. Output a CSV: repo, stars, org/owner, last push, file path of the match.
Filter out tutorials/forks/archived repos; rank by stars and recency. Top 100 only.
```

**You do:** publish the README and post, then send 20 personal DMs this week:
> "Saw <repo> calls <model>, which is being retired on <date>. I built an open-source tool that
> replays your real traffic on the new model and auto-fixes the prompts. We recovered X% of
> regressions in our tests. Want me to run it on your app for free?"

---

## End-of-week checklist
- [ ] CI green, live adapters verified, cache + budget guard in place
- [ ] 3 apps × 120 inputs, traces on each source model
- [ ] Judge calibrated (85%+ agreement with your labels)
- [ ] Optimizer with clustering, minibatch search, anti-memorization
- [ ] Full experiment run within budget
- [ ] RESULTS.md + kill/continue decision written down
- [ ] Blog post drafted, 100 candidate teams found, 20 DMs sent
