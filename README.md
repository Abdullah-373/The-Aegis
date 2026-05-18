# The Aegis

**A three-agent AI system for contract risk analysis.**

Upload a PDF. Three Gemini-backed agents argue about it live in your browser: one sells the deal, one attacks it, and the third one delivers a structured verdict with a 0–100 risk score and a list of conditions to fix. Same PDF the second time? It comes back from the SQLite cache in under two seconds at zero API cost.

![Verdict dashboard](docs/screenshot.png)

---

## What it does

The Aegis runs an adversarial three-agent pipeline over a single WebSocket connection:

| Agent     | Role         | Job |
|-----------|--------------|-----|
| **Alex**  | Strategist   | Reads the contract and lists every reason it is a good deal. |
| **Sam**   | Red Team     | Reads the contract *and* Alex's transcript, then attacks Alex's claims one by one. |
| **Maya**  | Judge        | Reads both transcripts and renders a final ruling. Output ends with a fenced JSON block that matches a Pydantic schema, so the verdict can be consumed by code, not just by humans. |

The Judge's ruling contains:

- A verdict — `GO`, `NO-GO`, or `CONDITIONAL-GO` (shown as `MAYBE` in the UI)
- A risk score from 0 to 100
- A one-line headline
- A risk matrix of at least four rows (`Low`/`Medium`/`High` likelihood × impact, plus mitigation)
- A list of conditions to satisfy (for `CONDITIONAL-GO` verdicts)

Verdicts are cached in SQLite keyed on `SHA-256(extracted_text + model_name)`, so a renamed copy of the same document hits the cache instantly and a model upgrade correctly invalidates the prior ruling.

## Measured performance

Tested end-to-end against the live Gemini API with the bundled `samples/sample_contract.pdf`:

| Metric              | First run         | Cached run        | Change             |
|---------------------|-------------------|-------------------|--------------------|
| Wall-clock time     | 30.65 s           | 1.68 s            | **94.5% faster**   |
| Tokens used         | ~3,768            | 0                 | 100% saved         |
| API cost (list)     | $0.0051           | $0.0000           | 100% saved         |
| Verdict             | CONDITIONAL-GO    | CONDITIONAL-GO    | Bit-identical      |

## Quick start

```bash
git clone https://github.com/Abdullah-373/The-Aegis.git
cd The-Aegis
pip install -r requirements.txt
python main.py
```

Open **http://localhost:8000** in your browser. Paste your Gemini API key, drop a PDF on the upload zone (or use `samples/sample_contract.pdf`), and hit **Start analysis**. The three agents will stream their output live; the final verdict appears as a structured dashboard.

### Docker

```bash
docker build -t aegis .
docker run -p 8000:8000 aegis
```

## Tech stack

- **Backend** — FastAPI · WebSockets · LangChain · Gemini 2.5 Flash
- **Frontend** — Vanilla JS · Tailwind CSS · Marked · single-page state machine over three views
- **Storage** — SQLite with WAL mode · SQLAlchemy
- **Validation** — Pydantic v2 with three-stage JSON recovery
- **PDF** — `pypdf` with optional OCR fallback via Tesseract
- **Tests** — pytest, 22 tests covering parsing, schema validation, retry classification, cost calculation, and the API endpoints

## Architecture at a glance

```
Browser <──── WebSocket ────> FastAPI
                                │
                          ┌─────┼─────┐
                    PDF text   Cache   Gemini
                       │         │       │
                   pypdf/OCR  SQLite     ChatGoogleGenerativeAI
                                          │
                                  ┌───────┼───────┐
                                Alex     Sam     Maya
                            (Strategist) (Red)  (Judge)
                                          │
                                  Pydantic-validated
                                  structured ruling
```

## BYOK and privacy

The Gemini API key is supplied in the first WebSocket frame. It lives only in the LLM client's memory for the duration of the request. It is never written to disk, never logged, and never stored in the cache. Cached verdicts contain the transcripts and the structured ruling — never the key that produced them.

## Project layout

```
├── main.py               FastAPI app, WebSocket pipeline, agent prompts, parsing
├── database.py           SQLAlchemy models, SQLite WAL-mode pragmas
├── templates/
│   └── index.html        Single-page client (setup / live / verdict views)
├── tests/
│   └── test_main.py      22 unit tests
├── samples/
│   └── sample_contract.pdf   Adversarial test contract used in the report
├── docs/
│   └── screenshot.png    Verdict dashboard screenshot
├── requirements.txt
├── Dockerfile
└── README.md
```

## Limitations

- `/api/history` and `/api/verdict/{id}` are unscoped — fine for solo use, needs auth before public deployment
- SQLite supports many readers but only one writer at a time (WAL mode helps but does not eliminate the bottleneck)
- OCR requires the `tesseract` and `poppler` binaries to be installed externally
- Per-token cost is computed from a hard-coded price table; will drift if Google changes prices

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Abdullah Hasan** · Student ID 807271

University final project. The full technical report (architecture, development challenges, empirical assessment) is included as a PDF in the project submission.
