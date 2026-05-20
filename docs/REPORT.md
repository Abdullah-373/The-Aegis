# The Aegis: A Multi-Provider Contract Tribunal, Three Versions In

**Abdullah Hasan** · Student ID 807271
Source code: [github.com/Abdullah-373/The-Aegis](https://github.com/Abdullah-373/The-Aegis)

---

## Abstract

The Aegis is a small web app for reading contracts. You drop a PDF in, three AI agents argue about it, and you get a verdict — GO, NO-GO, or CONDITIONAL-GO — with a 0-to-100 risk score and a list of conditions to fix before signing.

The project went through three versions in the same repository. **V1** was three prompt calls in a row: a Strategist (Alex) makes the bullish case, a Red Team (Sam) tears it apart, a Judge (Maya) writes the ruling. That worked but it was barely a "multi-agent system" — it was a chain. **V2** rebuilt the middle of the pipeline on LangGraph: a Planner picks two-to-five domain specialists (Financial, Legal, Data, Compliance, Operations) from a fixed roster, each specialist gets access to a `search_precedent` tool over a hand-curated knowledge base of 34 contract risk patterns across 15 categories, and a critique step lets Alex and Sam push back on Maya's ruling — if either dissents, Maya runs once more. **V3** (the build this report is about) added OpenAI as a second model provider next to Gemini, swapped a CDN-loaded Tailwind for a real build, replaced `len(text) // 4` cost estimation with `tiktoken`, and added an explicit scoring rubric to Maya's prompt so the same document does not move two whole verdict bands when you change the model.

Both pipelines (Fast = three direct calls, Full = the LangGraph machine) ship in the same app behind a mode picker. Fast keeps the original V1 design for cheap or free-tier-friendly runs. Full does eight to fifteen calls per analysis and is the default once a paid OpenAI key is in play. Either mode writes to the same SQLite cache, keyed on `SHA-256(extracted_text + model_name)`, so a second click on the same PDF returns the verdict with zero API calls.

Measured against the live OpenAI API on the two sample contracts that ship with the repository (both over the 100,000-character extractor limit, so both ran through the map-reduce condensation path):

- **`gpt-5-mini` on `contract_balanced.pdf` (Full mode):** CONDITIONAL-GO / risk **78** / **5,864 tokens** / **\$0.0034** / 268.25 s on the original run, replayed from the cache in under 100 ms.
- **`gpt-5-mini` on `contract_mixed.pdf` (Full mode):** CONDITIONAL-GO / risk **85** / **5,227 tokens** / **\$0.0038** / same 268.25 s original timing.
- **`gpt-4o-mini` on `contract_balanced.pdf`:** CONDITIONAL-GO / risk **70** / 4,241 tokens / **\$0.001422** / 100.54 s — same verdict band as gpt-5 at ~67× lower dollar cost.
- **Cache replay on any of the above:** **\$0.00** of API spend, zero outbound calls, sub-hundred-millisecond turnaround.

Every dollar figure above is the actual cost stored in the cached verdict export, not an estimate. The two figures with four decimal places come from the JSON exports committed to `docs/sample_verdicts/` so a reviewer can verify them without re-running the pipeline.

---

## 1. The Problem

Contracts are long. Nobody reads them. You either skim and miss the bad parts or you pay a lawyer hundreds of dollars per page. Neither option is good for the kind of mid-size deal where the legal spend would be a meaningful fraction of the contract value itself.

The obvious move — throw an LLM at the PDF and ask for a summary — fails for two reasons. First, one model writing one summary always lands in the middle. It averages the upside and the downside into bland prose that does not actually help anyone decide whether to sign. The thing a reviewer needs is the *spread*: the strongest possible reading of the deal *and* the worst possible reading, both at full strength, with a third opinion on top. Second, free-form model output cannot be consumed by downstream code without parsing prose with regular expressions — and that parsing breaks the first time the model phrases its answer slightly differently.

The Aegis tries to fix both of those at once. It never produces a neutral summary; it shows you Alex's bullish case and Sam's attack as two separate transcripts you can read in parallel, then Maya emits a verdict as a JSON object with a fixed Pydantic schema. Downstream code consumes the JSON, the human reads the transcripts. A third problem only became visible halfway through the build: a user who already pays for an OpenAI key should not have to acquire a Gemini key just to use this app. V3 solved that — paste any supported key, the right provider is selected for you, and the model picker auto-switches to the matching provider group.

---

## 2. Design Choices

This section is the reasoning behind the building blocks, not a survey of every library used. Where a decision came back to bite me later, I cross-reference the relevant entry in §4.

**Three agents, not one summary.** A single-summary approach hides the gap between the optimistic and pessimistic reading. The whole point of a contract review *is* that gap. I picked a fixed three-role pipeline — Alex (Strategist), Sam (Red Team), Maya (Judge) — rather than a free-form multi-agent chat. The roles are hard-coded in `HELPERS` (`main.py:240–244`). The transcripts stay auditable, which matters for a domain where a reviewer may need to justify their reading to a manager.

**LangGraph for the V2 pipeline, not a hand-rolled state machine.** Once V2 grew a Planner, five conditional specialists, two parallel critique nodes, and an optional revision pass, the control flow in `main.py` got ugly. I moved the topology into a `StateGraph` with a typed `TribunalState` (`agents.py:114–127`) and explicit nodes. Conditional edges handle the "if either critic dissents, send the ruling back to Maya for revision" branch. The graph is now one diagram instead of three pages of `if/elif` blocks.

**One tool, not a tool-use framework.** Specialists and the Red Team have access to exactly one tool: `search_precedent(query)` (`tools.py:13–42`). It runs TF-IDF over 34 hand-written `Precedent` entries in `knowledge_base.py` across 15 categories (liability, indemnification, termination, pricing, data, IP, disputes, SLA, assignment, confidentiality, governing law, warranty, exit, compliance, audit) and returns the top four matches. No external embedding service, no vector database, no extra dependencies. The corpus is small enough that this is *faster* than calling anything external.

**Pydantic with `Literal` enums for the verdict, plus a four-stage recovery chain.** Maya is told to end her response with a fenced JSON block. The schema (`FinalAnswer` in `main.py:285–290`) uses `Literal["GO", "NO-GO", "CONDITIONAL-GO"]` for the verdict and `Literal["Low", "Medium", "High"]` for severity. If the JSON is wrong, validation fails loudly instead of silently accepting `"high"` as a valid severity. The recovery chain behind that — regex re-extraction, temperature-zero re-ask, per-provider strong-model escalation, heuristic floor — is documented end-to-end in §3.

**Provider routing by key prefix.** V3 inspects the first few characters of the API key. `AIza...` is Gemini. `sk-...` is OpenAI. `sk-ant-...` is Anthropic, which I detect *on purpose* so the WebSocket can reject it with a clear error instead of letting the user wait ten seconds for an authentication failure. The model picker in the UI is grouped by provider and auto-switches when the detected provider changes. The user never picks "Provider" from a dropdown. The factory function (`_make_llm` in `main.py:180–219`) is the only place that knows which client class to instantiate; the rest of the pipeline is provider-agnostic.

**SQLite with a content-hash key, not Postgres or Redis.** The cache key is `SHA-256(extracted_text + model_name)`. Three things follow. (a) Renaming the PDF still hits the cache — the filename is not in the hash. (b) Switching models forces a fresh run, because the model name *is* in the hash. (c) A one-character edit gives a completely different hash, so the cache cannot get poisoned by an almost-identical file. The schema has idempotent migrations in `database.py:63–75` so a fresh install and an upgraded install both end up with the right columns. SQLite needs zero infrastructure and is fast enough for single-user scenarios; the bottleneck on a concurrent multi-user deployment is documented in §6.

**Lifespan handler instead of `@app.on_event("startup")`.** Starlette has been warning about `@app.on_event` for a while. On Python 3.14 the `DeprecationWarning` was the first thing printed on boot, which made the app look unfinished even though nothing was broken. V3 migrated the OCR-availability log line into a real `lifespan` async context manager (`main.py:230–263`). Clean startup log. While I was in there I added a one-second `webbrowser.open(...)` so the dashboard pops in your default browser automatically — the original quick-start was "run `python main.py` then open `http://localhost:8000`", and the first few times I ran V3 I forgot the second half and sat looking at the uvicorn log waiting for nothing.

**`tiktoken` instead of `len(text) // 4` for cost estimation.** V1 and V2 reported dollar costs computed from `len(text) // 4`. Wrong on non-English text, wrong on code-heavy text, defensible only as a rough estimate. V3 added a `tiktoken`-backed token counter (`main.py:475–510`) that uses the right encoder for OpenAI models (`o200k_base` for the gpt-5 family that tiktoken does not register yet) and falls back to `len // 4` only for Gemini, since tiktoken cannot tokenise Gemini honestly. The cost figures in §5 are therefore actually accurate for the OpenAI rows.

**An explicit scoring rubric in Maya's prompt.** V1 and V2 asked Maya for an integer 0-100 with no definition of what the numbers mean. The result was that the same document landed at risk 61 on gpt-5, risk 70 on gpt-4o-mini, and risk 78 on gpt-5-mini — the agents agreed on the verdict band but disagreed on the score by tens of points. V3 added an additive rubric to both copies of `MAYA_SYSTEM` (`main.py:315` and `agents.py:194`): start at 10, add bands per risk-row likelihood/impact, fixed bumps for known-bad clauses (six-months-or-less liability cap → +15, anonymised data licence surviving termination → +15, no clean data export → +10), and a fixed verdict-band map (0–39 GO, 40–74 CONDITIONAL-GO, 75–100 NO-GO). It is a prompt-level nudge, not a hard schema constraint, and §5 reports how well the model actually obeys it.
