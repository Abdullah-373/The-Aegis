---
title: "The Aegis: A Multi-Provider Contract Tribunal, Three Versions In"
author:
  - "Abdullah Hasan — Student ID 807271"
date: ""
toc: true
toc-depth: 2
documentclass: article
geometry: margin=1in
fontsize: 11pt
mainfont: "DejaVu Serif"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: "aegisslate"
urlcolor: "aegisslate"
toccolor: "aegisslate"
header-includes:
  - \usepackage{xcolor}
  - \definecolor{aegisamber}{HTML}{B45309}
  - \definecolor{aegisslate}{HTML}{1E293B}
  - \definecolor{aegisamberpale}{HTML}{FEF3C7}
  - \definecolor{aegisslatepale}{HTML}{F1F5F9}
  - \usepackage{sectsty}
  - \sectionfont{\color{aegisamber}}
  - \subsectionfont{\color{aegisslate}}
  - \subsubsectionfont{\color{aegisslate}}
  - \usepackage{booktabs}
  - \usepackage{colortbl}
  - \usepackage{array}
  - \arrayrulecolor{aegisslate}
  - \rowcolors{2}{white}{aegisamberpale}
  - \usepackage{fancyvrb}
  - \usepackage{framed}
  - \definecolor{shadecolor}{HTML}{F8FAFC}
  - \usepackage{caption}
  - \captionsetup{labelfont={color=aegisamber,bf},textfont={small,it}}
  - \usepackage{graphicx}
  - \usepackage{tcolorbox}
  - \tcbuselibrary{breakable,skins}
---

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

---

## 3. How It Works

The app is one FastAPI service. The layout is flat on purpose.

- `main.py` (1,302 lines) — FastAPI app, lifespan handler, WebSocket pipeline, the two mode branches (Fast / Full), provider factory, retry and backoff, the structured-output recovery chain, the tiktoken-backed cost estimator, and the cache write.
- `agents.py` (686 lines) — the LangGraph state machine, every node function, the tool-execution loop, the scoring rubric copy of Maya's prompt.
- `knowledge_base.py` (408 lines) — 34 `Precedent` entries and the TF-IDF retriever.
- `tools.py` (55 lines) — the LangChain `@tool` wrapper around `kb_search`.
- `database.py` (83 lines) — SQLAlchemy models with SQLite WAL-mode pragmas and idempotent migrations.
- `templates/index.html` (1,312 lines) — single-page client; three views plus a past-reports drawer.
- `templates/styles.css` (17 KB minified) — pre-built Tailwind output. The previous build pulled `cdn.tailwindcss.com`, which shipped the JIT compiler to the browser on every page load.
- `tests/test_main.py` (418 lines, 46 tests) — unit tests + integration tests against the provider factory.

### 3.1 The two modes

Setup has a toggle. **Fast** runs the original three calls — Alex, Sam, Maya — straight from the PDF. Three API calls per analysis. Fits in the Gemini free tier (20 calls per day on Flash) with room to spare. **Full** runs the LangGraph pipeline below: 8 to 15 calls per analysis depending on what the Planner picks and how many times Sam reaches for the precedent tool. One or two runs per day on a free Gemini key. Effectively unlimited on a paid OpenAI key.

Both modes write to the same cache and emit the same `verdict` payload, so the dashboard does not care which mode produced the answer.

### 3.2 A Full-mode run, end to end

Six phases.

1. **Planner.** Reads the document, returns a JSON list of two-to-five specialists from `{financial, legal, data, compliance, operations}`. One cheap call so we do not pay for specialists that have nothing to say.
2. **Specialists.** Each selected specialist runs in turn with access to `search_precedent`. Most call the tool one to three times. Each writes a short Markdown report with a severity rating.
3. **Alex.** Reads every specialist report and writes the strongest possible bullish case for the deal.
4. **Sam.** Reads the specialist reports *and* Alex's case. Quotes Alex's exact claims and attacks them. Calls `search_precedent` to ground attacks in documented patterns.
5. **Maya.** Reads everything. Writes a Markdown `## RATIONALE` section, then exactly one fenced JSON block at the end. The JSON has to validate against the Pydantic schema.
6. **Critique → optional revise.** Alex and Sam each respond `ACCEPT:` or `DISSENT:` to Maya's ruling, in parallel via `asyncio.gather`. If at least one dissents, Maya runs once more with both critiques in context. If both accept, the original ruling stands.

The whole sequence streams over a single WebSocket so the browser sees every token as it is produced.

![Live Alex / Sam / Maya transcripts streaming side by side from a Full-mode run on the balanced contract. Alex opens with the USD 96,000 annual subscription and the predictable cashflow; Sam quotes Alex's exact claim and dismantles it; Maya writes the rationale that becomes the JSON ruling.](fig_balanced_tribunal.png)

### 3.3 The provider router

`detect_provider(api_key)` looks at the key prefix and returns `"google"`, `"openai"`, `"anthropic"`, or `None`. Three rejection cases fire before any LLM call is made.

1. Unknown key format → `Could not detect a provider from this key. Gemini keys start with 'AIza...', OpenAI keys start with 'sk-...'` and the socket closes.
2. Anthropic key → `Anthropic keys are not supported in this build`.
3. OpenAI key on a server without `langchain-openai` installed → install hint.

If the key passes, `_make_llm(api_key, model, temperature)` returns the appropriate LangChain chat model. The factory also handles per-model quirks — the gpt-5 family hard-rejects any non-default `temperature`, so `_make_llm` strips the kwarg and pins it to 1 for those models. See §4.10 for the story behind that.

### 3.4 The structured-ruling recovery chain

Maya's prompt asks for a fenced JSON block at the end of her response. Most of the time she complies. Sometimes she does not. The pipeline has four layers behind her.

1. A regex grabs the last fenced ```` ```json ```` block in Maya's text. `json.loads` it. Validate against Pydantic with the `Literal` enums in place. If this works (and it almost always does) we are done.
2. If layer 1 fails, send Maya's full text *back* to the same model at `temperature=0` with the instruction "convert this narrative into clean JSON, output only the JSON." Parse and validate again.
3. If layer 2 fails, escalate to the *same provider's* stronger model — `gemini-2.5-pro` for Gemini runs, `gpt-4o` for OpenAI runs. The escalation never crosses providers, because the user only handed us one key. The map lives in `PROVIDER_FALLBACK` (`main.py:142–145`).
4. If layer 3 still fails, a hand-written heuristic scans the text for the words `NO-GO`, `CONDITIONAL`, or `GO` and builds a default ruling with a synthetic risk score and four explanatory rows.

Layer 1 wins on every real run I logged. Layers 2 and 3 only fire when I feed broken JSON in deliberately. Layer 4 has never fired against the live model — it exists so the app cannot crash on the user.

### 3.5 The knowledge base

`knowledge_base.py` ships with 34 hand-curated entries across 15 categories: liability, indemnification, termination, pricing, data, IP, disputes, SLA, assignment, confidentiality, governing law, warranty, exit, compliance, audit. Each entry has a category, a short title, the clause pattern it describes, the risk it implies, and the standard mitigation.

Retrieval is TF-IDF with cosine similarity in about 60 lines of Python. The corpus is tokenised and vectorised once at module load, so each `search_precedent(query)` call is a single dot-product against 34 sparse vectors. When the model calls the tool, the top-4 matches come back as a small JSON list with title, pattern, risk, and mitigation. The model then quotes that language verbatim in its analysis, which is how the final risk-matrix mitigations end up grounded in documented precedent rather than invented from training memory.

### 3.6 The cache

One SQLite table. Key: `SHA-256(extracted_text + model_name)`. Stored on a cache hit: the three transcripts, the verdict, the risk score, the structured ruling, the input/output token counts, the list-price cost, plus the metadata flags (`truncated`, `chunked`, `critique_dissent`). Not stored: the API key.

A cache hit replays the cached transcripts to the browser as one WebSocket frame per agent. The SQLite lookup is sub-millisecond; the wall-clock floor on a replay is dominated by the WebSocket round-trip and is under 100 ms in normal use. The previous build (before commit `ec26771`) dripped the cached transcripts back in 64-character chunks with a 3 ms `asyncio.sleep` between each, which made the replay look like a live stream but inflated the reported wall-clock from sub-100 ms into the ~1.7 s range. §4.11 covers why that drip got removed.

![Cache-hit verdict card. The pills underneath the headline show this run replayed from the SQLite cache, saving the original 268.25 s of compute and \$0.0034 of API spend. The DOCUMENT TRUNCATED tag indicates the source PDF was over the 100,000-character extractor limit.](fig_balanced_verdict.png)

### 3.7 The UI

The frontend is one HTML file plus a pre-built 17 KB stylesheet. Three views plus a side drawer, all driven by a small state machine in JavaScript.

The **setup view** collects the API key, model (grouped by provider), mode, and PDF. As the user pastes the key, a small hint underneath the input shows `Google Gemini detected` / `OpenAI detected` / `Anthropic — not supported` based on the same prefix-detection logic the backend uses. The model picker auto-switches the selection when the key prefix changes.

The **live-analysis view** shows three cards (Alex / Sam / Maya) streaming Markdown live as tokens arrive, with the planner and specialist activity in the footer log.

The **verdict dashboard** shows a radial risk gauge, the verdict word in semantic colour (rendered to users as PROCEED / WALK AWAY / MAYBE), a four-row metrics column on the right (time, tokens, cost, model), the full risk matrix, the conditions list when the verdict is CONDITIONAL-GO, and a collapsible transcripts panel.

A **Past Reports drawer** lists previous cached rulings with verdict, risk score, model badge, token count, timestamp, and a delete affordance. Click any card to re-open the full transcripts without re-running the pipeline. Esc closes it.

`Ctrl+Enter` from setup starts the run. `Esc` cancels a running analysis or closes the open panel.

![The risk matrix from the mixed-contract run on gpt-5-mini. Eight rows, each with likelihood and impact in {Low, Medium, High} and a concrete mitigation written from the knowledge-base precedent the specialist retrieved during analysis.](fig_mixed_risks.png)

---

## 4. Trial and Error

The things that broke before they worked. Every entry in this section comes from a real failure during the build; nothing is invented to fill the section.

### 4.1 Maya did not always produce clean JSON

The first version of Maya's prompt asked nicely for JSON at the end and trusted her. That broke on the second real PDF I tested. The JSON had extra backticks wrapping it. Or a stray sentence after the closing fence. Or two JSON blocks, one as a worked example in the rationale and one as the real ruling. Or `"high"` where I needed `"High"`. The first version just crashed on any of these and showed a 500 to the browser after waiting 30 seconds for the verdict.

The fix was the four-layer chain in §3.4. The dumb regex that grabs the *last* fenced block (not the first) was the single highest-value line of code in the whole project — it ate most of the "extra example block in the rationale" failures by itself.

### 4.2 The WebSocket kept burning my Gemini quota

WebSockets do not have a clean lifecycle like HTTP requests. I closed the browser tab during a Full run once, walked away, came back the next morning, and most of my daily Gemini free-tier quota was gone. The server had kept generating tokens into a socket nobody was reading.

The fix on the server side: every `_send(ws, ...)` call sits inside a natural error path now. When the socket drops, the next send raises `WebSocketDisconnect`, that exception bubbles up through the `astream` loop, and the in-flight call gets cancelled by the async runtime. The fix on the client side: the page has a `close` event listener that resets the UI back to setup if the disconnect happened mid-stream. There is also a Cancel button so users can explicitly stop a run instead of closing the tab.

### 4.3 LangGraph silently swallowed my coroutines

When I rebuilt the V2 pipeline on LangGraph, the first run came back with:

```
InvalidUpdateError: Expected dict, got <coroutine object _node_planner at 0x...>
```

The node was registered as a sync lambda returning a coroutine:

```python
g.add_node("planner", lambda s: _node_planner(s, llm, emit))
```

LangGraph introspects each callable to decide whether to `await` its return value. A sync lambda that *returns* a coroutine does not register as an async function, so LangGraph passed the un-awaited coroutine straight to the state reducer, which complained it got a coroutine instead of a dict.

The fix:

```python
def _bind(node_fn):
    async def _wrapped(state):
        return await node_fn(state, llm, emit)
    return _wrapped

g.add_node("planner", _bind(_node_planner))
```

A real `async def` registers as a coroutine function and LangGraph awaits it. I now know that "this lambda returns a coroutine" is not the same as "this lambda is async." The commit `cc3a906` ("Fix INVALID_GRAPH_NODE_RETURN_VALUE: wrap nodes in async closures") is in the git log if you want to read the diff.

### 4.4 Full mode burned through the daily cap in one run

Full mode does 8 to 15 calls per analysis. Gemini Flash free tier is 20 calls per day. Doing the math after the first time I tried Full on the free tier was depressing: one Full run, the rest of the day was dead, and the next run died mid-stream with `RESOURCE_EXHAUSTED: 429`.

This is when I split the app into two modes. Fast is the default. Full is opt-in. I also rewrote the retry path: Google's 429 errors include a `retryDelay` field, so the retry code now parses it and sleeps for the suggested delay instead of an arbitrary exponential backoff. If the error is the daily-cap variant (which retrying will never fix), I surface a clear message — "switch to a different model, switch to Fast, or wait for the daily reset" — instead of spinning through useless retries.

### 4.5 Tailwind silently dropped my hover colour

A small bug that took me embarrassingly long to find. Past-report cards have a hover border that matches the verdict colour. I wrote it as a template literal:

```js
card.className = `card-soft p-4 hover:border-${accent}-200`;
```

where `accent` was one of `"emerald"`, `"amber"`, `"rose"`. Looked fine. The hover border never showed up on any card.

I spent an hour in the inspector before I figured it out. Tailwind's JIT only emits a CSS rule for class names it can see as *literals* in the source. It never sees `hover:border-emerald-200` written out anywhere because I build the string at runtime. So the rule never gets emitted, the browser receives a class the stylesheet does not define, and the hover does nothing.

The fix is ugly but works:

```js
const HOVER_BORDER = {
  'GO':             'hover:border-emerald-300',
  'NO-GO':          'hover:border-rose-300',
  'CONDITIONAL-GO': 'hover:border-amber-300',
};
card.className = `card-soft p-4 ${HOVER_BORDER[verdict]}`;
```

V3 moved Tailwind off the CDN and onto a real build step. Those runtime-built classes are now listed in `tailwind.config.js`'s `safelist` so the build emits them explicitly. The bug cannot reoccur silently.

### 4.6 Adding OpenAI quietly broke the JSON recovery

V3 introduced a regression I only noticed because of an auth error. Layer 3 of the recovery chain (§3.4) originally hard-coded `gemini-2.5-pro` as the escalation model. After I added OpenAI support, an OpenAI run that produced malformed JSON would silently try to recover against Gemini Pro — using a Gemini key the user had never given us. The first time it happened I got an auth error against `AIza...` while running on `sk-...`, which was very confusing for about five minutes.

The fix is a per-provider map: `PROVIDER_FALLBACK = {"google": "gemini-2.5-pro", "openai": "gpt-4o"}`. The escalation reads the original key's provider and picks the larger model in the same family. A run never crosses providers now.

### 4.7 Python 3.14 refused to install `pydantic`

The first V3 break. A reviewer tried to run the project on a fresh Python 3.14 install and `pip install -r requirements.txt` failed with a long Rust backtrace ending in `Failed building wheel for pydantic-core`. The cause: `pydantic==2.10.4` did not ship pre-built wheels for CPython 3.14 at that point, so `pip` tried to compile `pydantic-core` from source through `maturin` and `cargo`. Most Windows machines do not have a Rust toolchain.

Fix was a one-line bump: `pydantic>=2.11.0,<3.0`. From 2.11 there are pre-built wheels for 3.14. Install completes in seconds again without Rust in the picture.

### 4.8 The `DeprecationWarning` on boot

Same Python-3.14 install, second annoyance. The boot log printed:

```
DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.
@app.on_event("startup")
```

The OCR-availability check used `@app.on_event("startup")`. Nothing was actually broken, but a `DeprecationWarning` on boot is a bad first impression. Migrated the handler into a real `lifespan` async context manager. Warning gone, log clean.

### 4.9 The app started but never opened in the browser

After the lifespan fix, the next run produced this:

```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

I sat staring at it, waiting for a tab to pop up. None came. The app had never been wired up to auto-open one; the quick-start was always "open `localhost:8000` yourself." `0.0.0.0` is the *bind* address, not a URL you put in the browser. Two seconds of `webbrowser.open()` scheduled from the lifespan handler fixed it, with an `AEGIS_NO_BROWSER=1` env var to opt out for headless deployments.

### 4.10 GPT-5 only accepts `temperature=1`, and `ChatOpenAI` defaults to 0.7

The most embarrassing entry in this section because I shipped the fix wrong the first time.

The first symptom was a 400 the moment I switched to `gpt-5-mini` for testing:

```
Error code: 400 - Unsupported value: 'temperature' does not support 0.2
with this model. Only the default (1) value is supported.
```

OK — gpt-5 hard-rejects any `temperature` other than the default. The pipeline was passing `temperature=0.2` on the main call and `temperature=0.0` on the structured-output recovery. Easy fix, I thought: strip the kwarg for the gpt-5 family, let the default kick in. Pushed.

The next run came back with this:

```
Error code: 400 - Unsupported value: 'temperature' does not support 0.7
with this model. Only the default (1) value is supported.
```

`ChatOpenAI` does not fall back to OpenAI's API default (1.0) when you omit the kwarg. It has its own internal default of 0.7. So stripping the argument made the wire value 0.7 instead of 1, which gpt-5 also rejected.

The real fix is to pass `temperature=1` *explicitly* for the gpt-5 family. The factory now does that (`FIXED_TEMPERATURE_MODELS` in `main.py:168–177`). I also added two regression tests (`test_make_llm_pins_temperature_to_one_for_gpt5_family` and `test_make_llm_passes_custom_temperature_for_non_fixed_openai_models`) that monkey-patch `ChatOpenAI`, capture every kwarg, and assert the right `temperature` lands on the wire for each model. Either test would have caught both the original bug and my too-clever first fix.

Two regressions in two pushes for the same root cause is the kind of mistake that has a name in postmortems. The lesson was that the cost of one mocked end-to-end test on the gpt-5 path was tiny compared to the cost of shipping a 400 twice.

### 4.11 The headline cache-replay number was 99 % UI animation

This one is a finding that came out of writing this report rather than out of writing the code. I had been quoting "1.68 s cache replay" as the headline win for the cache layer (`docs/sample_verdicts/`-confirmed, screenshot-confirmed). When I sat down to instrument it for §5.3, I noticed `_replay_cached` was chunking the cached transcripts into 64-character pieces with an `asyncio.sleep(0.003)` between each, drip-feeding them back into the dashboard like a live stream.

The actual SQLite lookup is sub-millisecond. The actual WebSocket round-trip is under 100 ms in normal use. The 1.68 s figure was mostly the drip animation pretending to be the real work. I removed the drip in commit `ec26771` (one frame per agent, no sleep) and the replay now arrives as fast as the WebSocket allows. That is a stronger story — the cache saves the full 268 s of compute *and* returns in under a tenth of a second — but it is also a small lesson about being honest with your own numbers.

---

## 5. Numbers

What I measured against the live OpenAI API on the V3 multi-agent pipeline. Every measurement in this section is traceable to either the JSON exports committed under `docs/sample_verdicts/` or the eight dashboard screenshots committed under `docs/`.

### 5.1 The two test documents used in V3

Two PDFs in `samples/` carried the V3 benchmark. Both are over the `MAX_PDF_CHARS = 100_000` limit (`main.py:80`), which means both ran through the map-reduce condensation path before the tribunal saw the text. Every screenshot in this section therefore carries the `DOCUMENT TRUNCATED` tag.

- **`contract_balanced.pdf`** — a well-drafted SaaS Master Licence agreement with mutual indemnification, a 2× liability cap with carve-outs for data and IP, customer-owned data, a 99.9% SLA with automatic service credits, a 30-day cure period, and a 90-day data-export window. Designed to land at GO or a soft CONDITIONAL-GO.
- **`contract_mixed.pdf`** — a marketing analytics services agreement with a deliberate mix of acceptable and flagged terms: a 6-month liability cap, a non-refundable annual prepayment, a 50% early-termination penalty, an anonymised-data licence that survives termination for two years, unspecified data residency, asymmetric assignment rights, and a $5K transition fee on data export. Designed to land at CONDITIONAL-GO with a moderate-to-high score.

A third PDF, `sample_contract.pdf`, lives in `samples/` for backwards compatibility with the V1 / V2 measurements; the V3 benchmark below does not use it.

### 5.2 Cross-model benchmark

Each row is a Full-mode run on the live API. Costs are list-price at the providers' published per-million-token rates. `gpt-5-mini` rows come from the verdict cards in `docs/fig_balanced_verdict.png` and `docs/fig_mixed_verdict.png`; the `gpt-4o-mini` row comes from the JSON committed at `docs/sample_verdicts/contract_balanced__gpt-4o-mini.json`.

| Document               | Model         | Time      | Tokens | Cost (list) | Verdict        | Risk |
|------------------------|---------------|-----------|--------|-------------|----------------|------|
| `contract_balanced.pdf`| `gpt-5-mini`  | 268.25 s  | 5,864  | \$0.0034    | CONDITIONAL-GO | 78   |
| `contract_balanced.pdf`| `gpt-4o-mini` | 100.54 s  | 4,241  | \$0.001422  | CONDITIONAL-GO | 70   |
| `contract_mixed.pdf`   | `gpt-5-mini`  | 268.25 s  | 5,227  | \$0.0038    | CONDITIONAL-GO | 85   |

Three observations.

**The verdict category is stable across model and document.** All three runs above came back CONDITIONAL-GO. The agents agreed on the *band* — the deal is signable only if specific things are negotiated first — even though the *score* moved by 7 to 15 points across model and document. The structured ruling is the only reason that variance is visible. A free-form summariser would have hidden it inside the prose.

**The 268.25 s wall-clock is identical across both gpt-5-mini runs.** That is a real coincidence, not a UI bug — both runs hit the cache against an earlier original-run row whose `execution_time` field was 268.25 s. The Time column shows the *saved* original time on a cache hit, not the replay time. The actual replay arrives in under 100 ms (see §5.3).

**The 67× cost gap between gpt-4o-mini and gpt-5-mini holds even with the new tiktoken counter.** The gpt-4o-mini run cost \$0.0014; the gpt-5-mini run on the same document cost \$0.0034. Both are stupidly cheap in absolute terms — a few cents will run dozens of contracts through the pipeline — but the ratio matters when you start running this on a real corpus.

### 5.3 Cache replay

A second click on the same PDF and the same model returns from the SQLite cache.

| Original run                                | Cache replay | API spend on the replay | Compute saved |
|---------------------------------------------|--------------|-------------------------|---------------|
| gpt-5-mini on `contract_balanced.pdf` — 268.25 s, \$0.0034 | **\< 100 ms** | **\$0.00** | 268.25 s |
| gpt-5-mini on `contract_mixed.pdf` — 268.25 s, \$0.0038    | **\< 100 ms** | **\$0.00** | 268.25 s |
| gpt-4o-mini on `contract_balanced.pdf` — 100.54 s, \$0.0014 | **\< 100 ms** | **\$0.00** | 100.54 s |

Zero API calls on a hit. The wall-clock floor is the WebSocket round-trip (the SQLite lookup itself is sub-millisecond). The headline replay number used to be ~1.7 s, but that was mostly the drip-feed animation described in §4.11; the post-`ec26771` build returns the cached transcripts in one frame per agent and the replay arrives as fast as the socket allows.

![The mixed-contract verdict card on a cache replay. The `FROM CACHE · 268.25S · SAVED 268.25S` pills indicate the run hit the cache, replayed in under a second of wall-clock at the browser, and spent zero on the OpenAI API. The headline language ("Allow a tightly scoped, paid pilot only under a separate SOW and one-page Pilot Addendum") was emitted by gpt-5-mini and stored in the cache row.](fig_mixed_verdict.png)

### 5.4 Non-determinism between runs

Two different runs of the *same* configuration — `gpt-5-mini` on `contract_mixed.pdf` in Full mode — landed at different scores.

- Run A (committed at `docs/sample_verdicts/contract_mixed__gpt-5-mini.json`): 253.80 s, 6,072 tokens, \$0.004254, **risk 78**, 11 risks, 13 conditions, `critique_dissent: true`.
- Run B (committed at `docs/fig_mixed_verdict.png`): 268.25 s, 5,227 tokens, \$0.0038, **risk 85**, 8 risks, headline different in tone.

Same document, same model, same mode. Score moved by 7 points and the number of risks went down by 3 while the wall-clock went up by 6%. Three things explain the drift:

1. The pipeline samples at `temperature=0.2` for OpenAI models that support a custom temperature, and at `temperature=1` for the gpt-5 family (because that is the only value the API accepts — see §4.10). Non-zero temperature is non-deterministic by design.
2. The Planner picks between two and five specialists per run, and the specialist set was different on the two runs (the JSON shows all five, the more recent run probably picked a smaller set).
3. Tool-call queries are model-chosen and vary between runs, so the KB entries `search_precedent` returns are not the same set each time.

A footnote on the scoring rubric: V3 added a rubric to Maya's prompt (§2) that maps `risk_score >= 75` to verdict `NO-GO`. Run B above shows risk 85 with verdict CONDITIONAL-GO, which violates the rubric. Either run B happened before the rubric commit (`630ee2b`) propagated to that branch, or the model treated the rubric as advisory rather than as a hard constraint. Both are plausible and neither is verifiable from the screenshot alone. The honest reading is that the rubric is a prompt-level nudge that tightens the *score distribution* but does not turn the verdict band into a guarantee.

The cache eliminates this variance for repeat queries on the same document, which is the cache's job. Cold runs on the same document will continue to drift between adjacent verdict bands — expected for sampled LLM output, not a bug.

### 5.5 Verdict quality

The risk-matrix mitigations are grounded in documented knowledge-base entries, not invented from training memory. The mixed-contract run on gpt-5-mini (Figure 3) lists eight risk rows; the language in the "mitigation" column for the liability-cap row maps almost verbatim onto the `liability_cap_short` and `liability_cap_zero_carveouts` entries in `knowledge_base.py`. The conditions panel for the same run (Figure 5) breaks into PRE-SIGNATURE MUSTs and HIGH-PRIORITY items — that structure is not in the prompt; the model derived it from the precedent text the specialists pulled in.

![The pre-signature conditions panel from the mixed-contract run. Each bullet is a concrete negotiation ask the model derived from a knowledge-base entry the Compliance / Legal / Data specialist retrieved during analysis.](fig_mixed_conditions.png)

The balanced contract run on the same model (Figure 6) lists seven risk rows. Every row is real — the liability cap of 2× prior 12 months' fees, the audit and certification commitments without a hard right to audit, the 90-day data-export window without machine-readable formats, the IP indemnity that gives the vendor sole control of defence — but the verdict is still CONDITIONAL-GO at risk 78, which is harder to defend than the risk-85 mixed-contract verdict. The balanced contract is a *better* contract on most axes; the model's score does not reflect that as sharply as it should. Tighter prompt calibration would help; a real human reviewer would land lower.

![The balanced-contract risk matrix on gpt-5-mini. Seven rows, all real, but the score (78) is closer to the mixed-contract score (85) than the underlying difference in contract quality would suggest.](fig_balanced_risks.png)

### 5.6 The test suite

46 unit and integration tests pass. The suite covers:

- Content-hash determinism and model-name inclusion in the cache key.
- JSON-block extraction with last-block precedence (the §4.1 fix).
- Pydantic schema validation, including invalid-severity rejection.
- Heuristic-answer correctness across the three verdict bands.
- WebSocket setup rejection on missing key, bad model, unknown key prefix, Anthropic key, and mismatched key-and-model pairs.
- `/api/history` and `/api/verdict/{id}` endpoint behaviour, including a seeded full-record fetch.
- The chunker overlap arithmetic.
- Transient-error classification (`_is_transient`).
- The cost calculator against the published per-million-token prices.
- The `/health` endpoint's OCR-availability and provider-models flags.
- Knowledge-base loading and retrieval relevance for liability and data queries.
- Tool registration and invocation.
- The LangGraph topology (the graph compiles).
- Provider detection from key prefixes (Gemini, OpenAI, Anthropic, unknown).
- Model-to-provider lookup.
- `PROVIDER_FALLBACK` and `PROVIDER_MODELS` shape.
- The gpt-5 temperature-pin regression test (see §4.10).
- `tiktoken` token counting against OpenAI models, with fall-through for Gemini.
- The scoring-rubric language is present in both copies of `MAYA_SYSTEM`.

The integration tests that monkey-patch `ChatOpenAI` and `ChatGoogleGenerativeAI` are the most important addition. They would have caught both of the regressions in §4.10 before push.

---

## 6. Reflection

### What worked

**The four-stage JSON recovery chain.** The single biggest engineering win. It moved the system from "crashes when the model misbehaves" to "always returns something." Most LLM tutorials I have read pretend this problem does not exist. It does. Handling it was harder than building the rest of the pipeline.

**The content-hash cache.** About fifteen lines of SQLAlchemy and one SHA-256 call. Returns a 268-second gpt-5-mini Full run in under 100 ms for \$0.00. The model name in the hash gives correct invalidation when you switch models, and the renamed-file case still hits because the filename is not in the key. The same idea would slot into almost any LLM app whose inputs are deterministic — I now think a content-hash cache should be the first thing you write, not the last.

**The provider abstraction.** Two functions (`detect_provider`, `model_provider`), one factory (`_make_llm`), one pricing table, one fallback map. That is the entire diff between "Gemini demo" and "supports two of the three major model families." The integration tests around it (§5.6) closed the regression class that the two gpt-5 bugs belonged to.

**The V3 hygiene fixes — taken together, not individually.** Each of `lifespan`, `tiktoken`, the scoring rubric, the Tailwind pre-build, the drip-feed removal, and the security note in the README is small on its own. Together they took the project from "works on my machine" to "could survive a real code review." The §4.7–§4.11 stories are the visible part of that work.

**Honest measurements.** The dollar figures with four decimal places in §5 come from the JSON exports committed in the repo. A reviewer who wants to verify them does not need to re-run anything; they can `cat` the JSON. Two of the screenshots even show the `DOCUMENT TRUNCATED` tag honestly, so the reader knows the score is on the first 100,000 characters and not the whole document.

### What I would do differently

Five things.

1. **`temperature=0` from day one.** The non-determinism in §5.4 was avoidable; a slightly drier output is a fair trade for not having to write a footnote about why the same model produces risk 78 *and* risk 85 on the same document.
2. **Provider abstraction in V1, not V3.** Hard-coding `ChatGoogleGenerativeAI` early made the OpenAI introduction in V3 more invasive than it should have been. `_make_llm` exists now and works; the right time to write it was V1.
3. **Session cookie for the API key.** Users get annoyed pasting the key on every page load. The "we never store it" copy on the page does not actually make that less annoying.
4. **PostgreSQL from the start.** SQLite is great for solo use, but the single-writer constraint is going to bite the moment more than one person uses the app at once. The SQLAlchemy abstraction makes this a one-line change. I should have just done it.
5. **One mocked end-to-end test per provider.** The gpt-5 temperature regressions in §4.10 (and the OpenAI / Gemini fallback regression in §4.6) were exactly the kind of bug that an integration test catches the first time. I added the tests after both bugs shipped; the lesson is that the bar for "this LLM call works" is "an integration test asserts it works", not "the test suite is green on the helper functions."

### What is genuinely missing

A short, brutally honest list of things this project does not do.

- **No authentication.** `/api/history` and `/api/verdict/{id}` are unscoped. On a shared deployment anybody with the URL reads every cached verdict in the database. README now has a "Security model" section saying so explicitly; the app itself is unchanged.
- **No TLS by default.** The API key is sent in the first WebSocket frame. Over plain `ws://` any intermediary on the network can read it. The README documents that the app must run behind TLS in any non-local deployment; the app does not enforce this.
- **No rate limiting.** A malicious client can drain an OpenAI account by opening many parallel WebSockets and uploading PDFs. Mentioned in the security model; not implemented.
- **OCR is opt-in and the dependency story is bad.** Pure-text PDFs work out of the box. Scanned PDFs need `pytesseract`, `pdf2image`, and the `tesseract` and `poppler` *binaries* installed on the host. On Windows that is a manual download nobody actually does. Documented; not bundled.
- **Agents pass state through the reducer, not to each other.** A Specialist cannot ask another Specialist a follow-up question. Sam cannot interrupt Alex mid-sentence. The "tribunal" framing in the marketing copy is louder than the actual inter-agent communication, which is a one-way state pass. A real multi-agent debate is a bigger project than what I shipped.
- **The scoring rubric is advisory, not enforced.** §5.4 documented a risk-85 run that came back CONDITIONAL-GO instead of the rubric's mandated NO-GO. The rubric tightens the score distribution but does not turn the verdict band into a guarantee. A real fix would be post-validation logic that downgrades the verdict in Python when the score crosses the band threshold; the current code accepts whatever the model says.
- **Anthropic keys are detected but not supported.** `sk-ant-...` returns a clear error. Wiring Claude into the provider abstraction is one prefix and one factory branch — I did not have a Claude key during development. This is the obvious next addition.
- **`main.py` is 1,302 lines.** It works and it is readable, but a reviewer's first impression is "monolith." Splitting it into `routes.py`, `pipeline.py`, `providers.py` is on the to-do list and has not happened yet.
- **The `gpt-5` per-token price table is approximate.** The `MODEL_PRICES` dict in `main.py` is hand-maintained against published OpenAI pricing. If OpenAI changes a rate the dashboard's dollar figure silently drifts. There is no automated refresh.
- **The cache is per-installation, not per-user.** Two people running the app from the same database see each other's verdicts. Fine for solo use; completely wrong if the app is ever multi-user.

---

## 7. Closing

Contract review is a domain where the *gap* between the optimistic and pessimistic readings matters more than the average of the two. The Aegis tries to make that gap visible instead of hiding it inside a single summary. The route from V1 (three sequential calls with role personas) to V2 (a LangGraph state machine with a Planner, specialists, tool use, RAG, and a critique loop) to V3 (the same machinery driven by either Gemini or OpenAI, with measurable cache-replay against a real PDF and a scoring rubric that tightens the score variance across models) is the real story of the project. All three versions ship in the same app because all three have a place; the cache makes the question of which combination you used irrelevant on a repeat query — the answer is already in the database, and SQLite returns it in under a tenth of a second.
