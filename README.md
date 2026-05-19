# The Aegis

**A multi-agent AI system for contract risk analysis.**

Upload a PDF. A planner agent picks which specialist analysts to run, the specialists do their analysis with access to a precedent search tool, a Strategist agent argues the bullish case, a Red Team agent attacks it, and a Judge agent renders a structured verdict with a 0–100 risk score and a list of conditions to fix. The whole deliberation streams live to the browser. Same PDF the second time? It comes back from the SQLite cache in under two seconds at zero API cost.

![Verdict dashboard](docs/screenshot.png)

---

## Two pipeline modes

The Gemini free tier is capped at 20 requests per day on `gemini-2.5-flash`, so the app ships with two pipelines and a mode picker:

| Mode | API calls per run | Free-tier runs per day | Pipeline |
|---|---|---|---|
| **Fast** (default) | 3 | ~6 | Alex → Sam → Maya |
| **Full multi-agent** | 8 – 15 | 1 – 2 | Planner → Specialists with tool use → Alex → Sam → Maya → Critique → optional revise |

Both modes write to the same cache and produce the same verdict shape, so the dashboard works the same either way.

## The agents

| Agent | Role | What it does |
|---|---|---|
| **Planner** (Full only) | Picks specialists | Reads the document and chooses 2–5 specialists from `{financial, legal, data, compliance, operations}` |
| **Specialists** (Full only) | Domain analysis | Each one runs with access to the `search_precedent` tool against the built-in knowledge base. Each writes a markdown report. |
| **Alex** | Strategist | Synthesises the strongest bullish case from the specialist reports (or directly from the PDF in Fast mode) |
| **Sam** | Red Team | Reads Alex's case and tears it apart point by point. Also has access to `search_precedent`. |
| **Maya** | Judge | Reads everything and emits a fenced JSON ruling that matches a Pydantic schema |
| **Critique** (Full only) | Alex & Sam respond | If either dissents from Maya's ruling, Maya runs once more to revise |

## The ruling

Maya's output ends with a fenced JSON block that always validates against this schema:

- `verdict` — `GO`, `NO-GO`, or `CONDITIONAL-GO`
- `risk_score` — integer 0 to 100
- `headline` — one-line summary
- `risks` — at least four rows, each with `Low`/`Medium`/`High` likelihood and impact and a mitigation
- `conditions` — list of fixes to apply (populated only for `CONDITIONAL-GO`)

There's a three-stage fallback chain: primary parse → temperature-0 re-extraction with the same model → escalation to `gemini-2.5-pro` → heuristic floor that scans the text for verdict keywords. The app never crashes on the user, even when the model misbehaves.

## The knowledge base

`knowledge_base.py` ships with 34 hand-curated contract risk patterns across 15 categories: liability, indemnification, termination, pricing, data, IP, disputes, SLA, assignment, confidentiality, governing law, warranty, exit, compliance, audit.

Retrieval is pure-Python TF-IDF with cosine similarity. The corpus is tokenised and vectorised once at module load, so a `search_precedent("liability cap 3 months")` query is one dot-product against 34 sparse vectors. No vector store, no embedding service, no extra deps.

When a specialist calls the tool, the top-4 matches come back as a JSON list with title, pattern, risk, and mitigation. The model then quotes that language in its analysis, which is how the final risk-matrix mitigations end up grounded in documented precedent rather than invented from training memory.

## Measured performance

Tested end-to-end against the live Gemini API with the bundled `samples/sample_contract.pdf`.

**Cache experiment (Fast mode, clean measurement):**

| Metric              | First run         | Cached replay     | Change            |
|---------------------|-------------------|-------------------|-------------------|
| Wall-clock time     | 30.65 s           | 1.68 s            | **94.5% faster**  |
| Tokens used         | 3,768             | 0                 | 100% saved        |
| API cost (list)     | $0.0051           | $0.0000           | 100% saved        |
| Verdict             | CONDITIONAL-GO    | CONDITIONAL-GO    | Bit-identical     |
| Risk score          | 85                | 85                | Bit-identical     |

**Full multi-agent mode (first run on same PDF):**

| Metric              | Value                                       |
|---------------------|---------------------------------------------|
| Wall-clock time     | **132.38 s**                                |
| Tokens used         | **4,337**                                   |
| API cost (list)     | **$0.0057**                                 |
| Verdict             | **NO-GO**                                   |
| Risk score          | **95** (high risk band)                     |
| Risks flagged       | 5 (data licence, prepayment+termination, liability cap, weak SLA, no data export) |

The Full pipeline produces a stronger verdict (NO-GO 95 vs CONDITIONAL-GO 85) because the specialists surface specific clauses that Alex and Sam in Fast mode can only see through their general lens. I did not get a clean cache-hit measurement for Full mode because of free-tier rate limits — the cache mechanism is the same in both modes (no API call on a hit), so the cost saving is identical; only the wall-clock replay number is unverified for Full.

The cost figures are list-price equivalents at Gemini's published per-token rates. Every test ran on the free tier so the actual out-of-pocket cost was $0.00.

## Quick start

```bash
git clone https://github.com/Abdullah-373/The-Aegis.git
cd The-Aegis
pip install -r requirements.txt
python main.py
```

Open **http://localhost:8000**. Paste your Gemini API key, pick a model, pick a mode (Fast for free-tier-friendly, Full for the multi-agent pipeline), drop `samples/sample_contract.pdf` on the upload zone, and hit **Start analysis**.

### Docker

```bash
docker build -t aegis .
docker run -p 8000:8000 aegis
```

## Tech stack

- **Backend** — FastAPI, WebSockets, LangChain, LangGraph, Gemini 2.5 Flash
- **Frontend** — vanilla JS, Tailwind CSS, Marked; three-view state machine
- **Storage** — SQLite (WAL mode), SQLAlchemy
- **Validation** — Pydantic v2 with a four-stage JSON recovery chain
- **PDF** — pypdf with optional OCR fallback via Tesseract
- **Knowledge base** — pure-Python TF-IDF + cosine similarity over 34 in-code precedents
- **Tests** — pytest, 33 unit tests covering parsing, schema validation, retry classification, cost calculation, the API endpoints, knowledge-base retrieval, tool registration, and graph topology

## Architecture at a glance

```
Browser ───── WebSocket ─────→  FastAPI
                                  │
                            ┌─────┼─────┐
                          PDF     Cache   Gemini
                           │        │        │
                       pypdf/OCR  SQLite  ChatGoogleGenerativeAI
                                              │
                                  ┌───────────┴───────────┐
                                  │  Fast (3 calls)       │  Full (8-15 calls)
                                  │  Alex → Sam → Maya    │  Planner → Specialists
                                  └────────────────────┐  │     (with search_precedent
                                                       │  │      tool, RAG over 34-entry KB)
                                                       │  │     → Alex → Sam → Maya
                                                       │  │     → Critique → optional Revise
                                                       ↓  ↓
                                            Pydantic-validated structured ruling
                                            (4-stage recovery: primary parse →
                                             temp=0 retry → Pro escalation →
                                             heuristic floor)
```

## BYOK and privacy

The Gemini API key is supplied in the first WebSocket frame. It lives only in the LLM client's memory for the duration of the request. It is never written to disk, never logged, never stored in the cache. Cached verdicts contain transcripts and the structured ruling — never the key that produced them.

## Project layout

```
├── main.py                    FastAPI app, WebSocket pipeline, mode switch, retry
├── agents.py                  LangGraph state machine, nodes, tool-execution loop
├── knowledge_base.py          34 contract risk patterns + TF-IDF retrieval
├── tools.py                   @tool wrappers (search_precedent)
├── database.py                SQLAlchemy models, WAL-mode pragmas
├── templates/
│   └── index.html             Single-page client (setup / live / verdict)
├── tests/
│   └── test_main.py           33 unit tests
├── samples/
│   └── sample_contract.pdf    Adversarial test contract used in the report
├── docs/
│   └── screenshot.png         Verdict-dashboard screenshot
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md
```

## Limitations

- `/api/history` and `/api/verdict/{id}` are unscoped — fine for solo use, needs auth before public deployment
- SQLite supports many readers but only one writer at a time (WAL mode helps but does not eliminate the bottleneck)
- OCR requires the `tesseract` and `poppler` binaries to be installed externally
- Per-token cost is computed from a hard-coded price table; will drift if Google changes prices
- Agents pass state through the LangGraph reducer rather than calling each other directly — no live inter-agent dialogue

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Abdullah Hasan** · Student ID 807271

University final project. The full technical report (architecture, development challenges, empirical assessment) is in the submission package.
