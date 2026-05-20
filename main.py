"""The Aegis - Multi-Agent AI Tribunal for Document Risk Analysis.

A FastAPI BYOK service that streams three Gemini agents (Alex the Strategist,
Sam the Red Team, Maya the Judge) over a single WebSocket and returns a
structured ruling with a risk score and a mitigation matrix.

This module wires together:
  - PDF text extraction with optional OCR fallback for scanned documents
  - Map-reduce condensation for documents longer than the context budget
  - Per-call retry with exponential backoff on transient Gemini errors
  - Three-stage structured-output recovery (primary parse, escalated fallback,
    heuristic last resort) so the Judge's verdict is always machine-readable
  - SQLite WAL-mode cache keyed by content hash + model name
  - Structured logging on the server side
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import time
import warnings
import webbrowser
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

# Silence the LangChain → Pydantic V1 shim warning on Python 3.14. The shim
# functionality we don't use is broken under 3.14; the bits we do use work
# fine. Suppressing the warning keeps the startup log readable.
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic V1.*",
)

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

from database import SessionLocal, VerdictCache, init_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [aegis] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aegis")

# ---------------------------------------------------------------------------
# Optional OCR
# ---------------------------------------------------------------------------

try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_bytes  # type: ignore
    OCR_AVAILABLE = True
except Exception:  # noqa: BLE001
    OCR_AVAILABLE = False
# The OCR availability log line is emitted once from the FastAPI startup
# event (see _log_startup below) rather than at module-import time, so it
# does not print twice when uvicorn re-imports the module.

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Raised from 18k. Gemini 2.x models support large contexts; only documents
# longer than this trigger the map-reduce condensation path.
MAX_PDF_CHARS = 100_000
CHUNK_SIZE = 70_000
CHUNK_OVERLAP = 1_500

# ---------------------------------------------------------------------------
# Providers (Gemini + OpenAI)
# ---------------------------------------------------------------------------

# OpenAI support is optional. If langchain-openai is installed the user can
# paste a key starting with "sk-" and the system routes to ChatOpenAI; if it
# is missing, the system still works for Gemini and rejects OpenAI keys with
# a clear error.
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:  # noqa: BLE001
    OPENAI_AVAILABLE = False

GEMINI_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
}
OPENAI_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-mini",
}
ALLOWED_MODELS = GEMINI_MODELS | OPENAI_MODELS
DEFAULT_MODEL = "gemini-2.5-flash"

# Per-provider fallback model used to recover a structured ruling when the
# primary parse fails. Stronger / more reliable than the default for each.
PROVIDER_FALLBACK = {
    "google": "gemini-2.5-pro",
    "openai": "gpt-4o",
}

# Per-provider grouping for the model picker — frontend renders these as
# <optgroup> sections so the user can see at a glance which provider each
# model belongs to.
PROVIDER_MODELS = {
    "google": sorted(GEMINI_MODELS),
    "openai": sorted(OPENAI_MODELS),
}

# Real per-million-token prices in USD (input, output). OpenAI figures are
# the published rates as of 2026; update the table if Google or OpenAI
# changes them.
MODEL_PRICES = {
    # Gemini
    "gemini-2.0-flash":      (0.10,  0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-flash":      (0.30,  2.50),
    "gemini-2.5-flash-lite": (0.10,  0.40),
    "gemini-2.5-pro":        (1.25, 10.00),
    # OpenAI
    "gpt-4o":                (2.50, 10.00),
    "gpt-4o-mini":           (0.15,  0.60),
    "gpt-5":                 (5.00, 20.00),
    "gpt-5-mini":            (0.25,  1.00),
}


def detect_provider(api_key: str) -> str | None:
    """Pick a provider from the key prefix. Returns None for unknown keys."""
    key = (api_key or "").strip()
    if key.startswith("AIza"):
        return "google"
    if key.startswith("sk-ant-"):
        return "anthropic"  # detected but not supported
    if key.startswith("sk-"):
        return "openai"
    return None


def model_provider(model: str) -> str | None:
    """Return the provider a model belongs to, or None if unknown."""
    if model in GEMINI_MODELS:
        return "google"
    if model in OPENAI_MODELS:
        return "openai"
    return None


# Models that hard-reject any non-default `temperature` value. The gpt-5
# family raises HTTP 400 "Unsupported value: 'temperature' does not support
# X with this model. Only the default (1) value is supported." if you pass
# anything other than 1 (or omit the arg). The structured-output recovery
# and the main pipeline both try to set custom temperatures, so we have to
# strip the argument for these models rather than pass it through.
FIXED_TEMPERATURE_MODELS = {
    "gpt-5",
    "gpt-5-mini",
}


def _make_llm(api_key: str, model: str, temperature: float = 0.2):
    """Construct the right LangChain chat model for the given API key.

    Falls back to a Pydantic / LangChain default temperature semantics —
    same parameter name, different normalisation across providers. For
    models in FIXED_TEMPERATURE_MODELS we omit the kwarg entirely so the
    provider uses its hard-coded default (which is the only value it
    accepts).
    """
    provider = detect_provider(api_key)
    if provider == "google":
        return ChatGoogleGenerativeAI(
            google_api_key=api_key, model=model, temperature=temperature,
        )
    if provider == "openai":
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI support requires the langchain-openai package. "
                "Install it with: pip install langchain-openai"
            )
        # ChatOpenAI uses `api_key` rather than the provider-named arg.
        # The gpt-5 family ONLY accepts temperature=1 (OpenAI's API default).
        # ChatOpenAI's own default is 0.7, not 1.0, so omitting the kwarg
        # does not work — we have to pass temperature=1 explicitly.
        if model in FIXED_TEMPERATURE_MODELS:
            return ChatOpenAI(api_key=api_key, model=model, temperature=1)
        return ChatOpenAI(
            api_key=api_key, model=model, temperature=temperature,
        )
    raise RuntimeError(
        f"Could not detect provider from API key (key starts with "
        f"'{api_key[:4]}...'). Supported prefixes: AIza... (Gemini), "
        f"sk-... (OpenAI)."
    )

HELPERS = {
    "alex": {"name": "Alex", "role": "Strategist", "tag": "finds the upside"},
    "sam":  {"name": "Sam",  "role": "Red Team",   "tag": "finds the problems"},
    "maya": {"name": "Maya", "role": "Judge",      "tag": "renders the ruling"},
}

async def _open_browser_when_ready(url: str, delay: float = 1.0) -> None:
    """Open the dashboard in the user's default browser after a short delay.

    The delay gives uvicorn a moment to finish binding the socket so the
    new tab does not race the server and show "site can't be reached".
    """
    await asyncio.sleep(delay)
    try:
        webbrowser.open(url, new=2)
        log.info("opened dashboard in default browser: %s", url)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not auto-open browser: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Single OCR-status log line at server boot, plus auto-open browser.

    Logging from a lifespan handler rather than at module import time
    avoids the duplicate-message issue when uvicorn imports `main` twice
    (once as the entry-point script, once again to resolve `main:app`).
    """
    if OCR_AVAILABLE:
        log.info("OCR available (pytesseract + pdf2image installed)")
    else:
        log.info(
            "OCR not available — install pytesseract, pdf2image and the "
            "tesseract binary to enable scanned-PDF support"
        )
    # Auto-open the dashboard in the default browser on first launch. The
    # uvicorn reloader imports the app twice; the env-var guard ensures we
    # only open the tab once. Users running headless (CI, Docker, SSH) can
    # opt out by setting AEGIS_NO_BROWSER=1.
    if not os.environ.get("AEGIS_NO_BROWSER") and not os.environ.get("AEGIS_BROWSER_OPENED"):
        os.environ["AEGIS_BROWSER_OPENED"] = "1"
        asyncio.create_task(_open_browser_when_ready("http://localhost:8000"))
    yield


app = FastAPI(
    title="The Aegis",
    description="Three AI agents review your PDF",
    lifespan=_lifespan,
)
templates = Jinja2Templates(directory="templates")

init_db()


# ---------------------------------------------------------------------------
# Structured ruling schema
# ---------------------------------------------------------------------------

Severity = Literal["Low", "Medium", "High"]
Verdict = Literal["GO", "NO-GO", "CONDITIONAL-GO"]


class RiskRow(BaseModel):
    risk: str
    likelihood: Severity
    impact: Severity
    mitigation: str


class FinalAnswer(BaseModel):
    verdict: Verdict
    risk_score: int = Field(..., ge=0, le=100)
    headline: str
    risks: list[RiskRow] = Field(..., min_length=3)
    conditions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

ALEX_SYSTEM = (
    "You are ALEX, the Strategist. You read business documents and produce "
    "a sharp, numbered breakdown of the strategic BUSINESS BENEFITS: upside "
    "scenarios, revenue levers, market positioning gains, and competitive "
    "advantages. Be confident, concrete, and precise. Cite clauses where "
    "useful. Match your register to the document. Keep the answer under "
    "320 words. Use markdown."
)

SAM_SYSTEM = (
    "You are SAM, the Red Team. You are a ruthless adversarial general "
    "counsel. The Strategist just made the bullish case. ATTACK IT. Find "
    "every legal loophole, liability trap, indemnification landmine, "
    "ambiguous clause, jurisdictional risk, exit penalty, IP exposure, and "
    "worst-case scenario. Quote the Strategist's claims and dismantle them "
    "one by one. Be precise. Keep the answer under 320 words. Use markdown."
)

MAYA_SYSTEM = """You are MAYA, the Judge. You have heard the Strategist and the Red Team. Render the final ruling.

OUTPUT FORMAT (STRICT):

First, output a markdown section titled `## RATIONALE` with 2-4 sentences
weighing both sides.

Then, output EXACTLY ONE fenced JSON code block at the very end of your
response. The JSON MUST validate against this schema:

```json
{
  "verdict": "GO" | "NO-GO" | "CONDITIONAL-GO",
  "risk_score": <integer 0-100>,
  "headline": "<one-line ruling summary, max 90 chars>",
  "risks": [
    {
      "risk": "<concise risk description>",
      "likelihood": "Low" | "Medium" | "High",
      "impact": "Low" | "Medium" | "High",
      "mitigation": "<actionable mitigation>"
    }
  ],
  "conditions": ["<condition>", "..."]
}
```

SCORING RUBRIC for `risk_score` (apply additively, then cap at 100):
- Start at 10 (baseline operational risk on any contract).
- +5 to +15 per HIGH-impact risk row, scaled by likelihood
  (Low likelihood adds the floor of the band, High adds the ceiling).
- +3 to +8 per MEDIUM-impact risk row, same likelihood scaling.
- +1 to +3 per LOW-impact risk row.
- +15 if the contract has an uncapped or six-months-or-less liability cap
  with no carve-outs for IP, data breach, or gross negligence.
- +15 if customer data may be used to train commercial models without an
  explicit opt-in, or if anonymised-data licence survives termination.
- +10 if there is no clean data-export path on termination.
- +10 if there is a punitive (>25%) early-termination fee with no
  mutual-fault carve-outs.

VERDICT BAND mapping (apply after scoring):
- 0-39   -> `GO`               (proceed; ordinary contract hygiene)
- 40-74  -> `CONDITIONAL-GO`   (proceed only after listed conditions met)
- 75-100 -> `NO-GO`            (do not sign in current form)

Requirements:
- The JSON block MUST be the LAST thing in your response.
- `risks` MUST have at least 4 entries.
- `conditions` is required only for CONDITIONAL-GO (otherwise empty list).
- Do not include any other JSON blocks anywhere in the response.
"""

CONDENSE_SYSTEM = (
    "You are a document summarizer. Read the section of a contract or "
    "business document below and produce a dense factual summary preserving "
    "every clause, financial figure, obligation, deadline, party name, "
    "termination condition, indemnification provision, and risk-bearing "
    "phrase. Do not editorialise. Output markdown, under 600 words."
)


# ---------------------------------------------------------------------------
# PDF extraction (with optional OCR fallback)
# ---------------------------------------------------------------------------

def _extract_pdf_text_native(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            continue
    return "\n".join(parts).strip()


def _ocr_pdf(raw: bytes) -> str:
    """OCR fallback for image-only PDFs. Returns '' if OCR not installed."""
    if not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_bytes(raw, dpi=200)
        pages = []
        for img in images:
            pages.append(pytesseract.image_to_string(img) or "")
        return "\n".join(pages).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("OCR failed: %s", exc)
        return ""


def _extract_pdf(raw: bytes) -> tuple[str, int, bool, bool]:
    """Return (text, original_chars, truncated, used_ocr)."""
    text = _extract_pdf_text_native(raw)
    used_ocr = False
    if not text:
        log.info("native PDF extraction returned no text — trying OCR")
        text = _ocr_pdf(raw)
        used_ocr = bool(text)
    original = len(text)
    truncated = original > MAX_PDF_CHARS
    return text[:MAX_PDF_CHARS], original, truncated, used_ocr


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _hash_content(text: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_final_json(text: str) -> dict[str, Any] | None:
    matches = _JSON_BLOCK_RE.findall(text)
    if not matches:
        return None
    for raw in reversed(matches):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None


def _is_transient(exc_str: str) -> bool:
    msg = exc_str.lower()
    return any(
        s in msg
        for s in (
            "503", "429", "500", "502", "504",
            "unavailable", "timeout", "temporarily",
            "connection reset", "deadline exceeded",
            "internal error", "overloaded",
            "resource_exhausted",
        )
    )


_RETRY_DELAY_RE = re.compile(
    r"retry[Dd]elay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s?", re.IGNORECASE,
)


def _suggested_retry_delay(exc_str: str) -> float | None:
    """Parse Google's `retryDelay` hint out of a 429 error body."""
    m = _RETRY_DELAY_RE.search(exc_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _is_quota_exhausted(exc_str: str) -> bool:
    """Detect free-tier daily-cap exhaustion (a non-retryable 429 variant)."""
    msg = exc_str.lower()
    return (
        "free_tier" in msg or "free-tier" in msg
        or "per day" in msg or "perdaypper" in msg
        or "generaterequestsperday" in msg
    )


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = MODEL_PRICES.get(model, MODEL_PRICES[DEFAULT_MODEL])
    return round(
        (input_tokens * in_price + output_tokens * out_price) / 1_000_000, 6
    )


# Token counting. We prefer tiktoken when the dependency is installed and the
# model has a registered encoder (OpenAI models do; the gpt-5 family falls
# through to o200k_base which is the closest available encoder). Gemini and
# anything unknown fall through to a four-chars-per-token approximation. The
# approximation is honest within ~15% on English prose and bad on code-heavy
# or non-ASCII text, which is why the dashboard footnotes the figures as
# list-price estimates.
try:
    import tiktoken  # type: ignore
    TIKTOKEN_AVAILABLE = True
except Exception:  # noqa: BLE001
    TIKTOKEN_AVAILABLE = False

_TIKTOKEN_ENCODERS: dict[str, Any] = {}


def _tiktoken_encoder(model: str):
    """Return a cached tiktoken encoder for `model`, or None if unavailable."""
    if not TIKTOKEN_AVAILABLE:
        return None
    if model in _TIKTOKEN_ENCODERS:
        return _TIKTOKEN_ENCODERS[model]
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001
        try:
            # gpt-5 / gpt-5-mini / new models tiktoken does not know yet —
            # the o200k_base encoding is the right default for them.
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001
            enc = None
    _TIKTOKEN_ENCODERS[model] = enc
    return enc


def _count_tokens(model: str, text: str) -> int:
    """Count tokens for `text` against `model`. Falls back to len/4 if tiktoken
    is missing or the model is not OpenAI-shaped (e.g. Gemini)."""
    if not text:
        return 1
    if model in OPENAI_MODELS:
        enc = _tiktoken_encoder(model)
        if enc is not None:
            try:
                return max(1, len(enc.encode(text)))
            except Exception:  # noqa: BLE001
                pass
    return max(1, len(text) // 4)


def _approx_input_tokens(*texts: str, model: str = DEFAULT_MODEL) -> int:
    return sum(_count_tokens(model, t) for t in texts) or 1


def _approx_output_tokens(text: str, *, model: str = DEFAULT_MODEL) -> int:
    return _count_tokens(model, text)


async def _send(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Retrying streamer
# ---------------------------------------------------------------------------

async def _stream_helper(
    ws: WebSocket,
    llm: ChatGoogleGenerativeAI,
    helper_id: str,
    helper_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_attempts: int = 3,
) -> str:
    """Stream one helper's output. Retries only on transient errors BEFORE
    the first token is delivered, so the client never sees duplicated tokens.
    """
    await _send(ws, {
        "type": "helper_start", "helper": helper_id, "name": helper_name,
    })

    full = ""
    last_exc_msg = ""

    for attempt in range(1, max_attempts + 1):
        full = ""
        first_token = True
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            async for chunk in llm.astream(messages):
                token = getattr(chunk, "content", "") or ""
                if not token:
                    continue
                first_token = False
                full += token
                await _send(ws, {
                    "type": "helper_token", "helper": helper_id, "token": token,
                })
            break  # success
        except (WebSocketDisconnect, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc_msg = str(exc)
            transient = _is_transient(last_exc_msg)
            # If the free-tier daily cap is the cause there is no point
            # retrying — the next call will fail with the same error until
            # the cap resets. Surface a clear message instead.
            if _is_quota_exhausted(last_exc_msg):
                log.warning("%s hit the free-tier daily cap", helper_id)
                msg = (
                    "Gemini free-tier daily quota reached. The current "
                    "tribunal run was incomplete. Switch the model picker "
                    "to a different model, wait for the daily reset, or "
                    "use Fast mode (3 API calls per run) instead of Full "
                    "multi-agent mode (8-15 calls)."
                )
                try:
                    await _send(ws, {"type": "helper_token", "helper": helper_id,
                                     "token": f"\n\n[QUOTA EXHAUSTED: {msg}]"})
                except Exception:  # noqa: BLE001
                    pass
                full += f"\n\n[QUOTA EXHAUSTED]"
                break
            if first_token and transient and attempt < max_attempts:
                suggested = _suggested_retry_delay(last_exc_msg)
                # Cap the suggested delay to a sensible upper bound so a
                # buggy server hint cannot wedge the pipeline indefinitely.
                if suggested is not None:
                    delay = min(suggested + 0.5, 30.0)
                else:
                    delay = 1.0 * (2 ** (attempt - 1))
                log.warning(
                    "%s attempt %d/%d transient (delay=%.1fs%s): %s",
                    helper_id, attempt, max_attempts, delay,
                    " from server" if suggested is not None else "",
                    last_exc_msg[:160],
                )
                await _send(ws, {
                    "type": "status",
                    "message": (f"Transient error in {helper_id} — "
                                f"retrying in {delay:.1f}s..."),
                })
                await asyncio.sleep(delay)
                continue
            # either non-transient, or we already streamed tokens, or out of attempts
            log.error("%s failed: %s", helper_id, last_exc_msg)
            err = f"\n\n[ERROR: {last_exc_msg}]"
            full += err
            try:
                await _send(ws, {
                    "type": "helper_token", "helper": helper_id, "token": err,
                })
            except Exception:  # noqa: BLE001
                pass
            break

    await _send(ws, {
        "type": "helper_end", "helper": helper_id,
        "chars": len(full),
        "est_tokens": _approx_output_tokens(full),
    })
    return full


# ---------------------------------------------------------------------------
# Map-reduce condensation for long documents
# ---------------------------------------------------------------------------

def _chunk_text(text: str, *, size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


async def _condense_long_document(
    ws: WebSocket,
    llm: ChatGoogleGenerativeAI,
    text: str,
) -> str:
    """Map-reduce: summarize each chunk, return concatenated summaries."""
    chunks = _chunk_text(text)
    log.info("condensing long document: %d chars -> %d chunks", len(text), len(chunks))
    await _send(ws, {
        "type": "status",
        "message": (f"Document is long ({len(text):,} chars). Condensing "
                    f"into {len(chunks)} chunks before the tribunal..."),
    })

    summaries: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        await _send(ws, {
            "type": "status",
            "message": f"Condensing chunk {idx}/{len(chunks)}...",
        })

        async def _do_call() -> str:
            resp = await llm.ainvoke([
                SystemMessage(content=CONDENSE_SYSTEM),
                HumanMessage(content=f"SECTION {idx}/{len(chunks)}:\n\n{chunk}"),
            ])
            return resp.content or ""

        for attempt in range(1, 4):
            try:
                summary = await _do_call()
                break
            except (WebSocketDisconnect, asyncio.CancelledError):
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt < 3 and _is_transient(str(exc)):
                    await asyncio.sleep(1.0 * (2 ** (attempt - 1)))
                    continue
                log.warning("condense chunk %d failed: %s", idx, exc)
                summary = f"[chunk {idx} could not be summarised: {exc}]"
                break
        summaries.append(f"### Section {idx} of {len(chunks)}\n\n{summary}")

    return "\n\n".join(summaries)


# ---------------------------------------------------------------------------
# Structured-output recovery
# ---------------------------------------------------------------------------

async def _try_extract_json(
    llm: ChatGoogleGenerativeAI, maya_text: str
) -> FinalAnswer | None:
    prompt = (
        "Convert the following tribunal narrative into the structured JSON "
        "ruling. Output ONLY the JSON object, no fences, no extra text.\n\n"
        "Schema: {verdict: 'GO'|'NO-GO'|'CONDITIONAL-GO', "
        "risk_score: int 0-100, headline: str, "
        "risks: [{risk, likelihood, impact, mitigation}] (min 4, "
        "likelihood/impact in {Low,Medium,High}), "
        "conditions: [str] (populated only for CONDITIONAL-GO)}.\n\n"
        f"NARRATIVE:\n{maya_text}"
    )
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = (resp.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
        data = json.loads(raw)
        return FinalAnswer.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        log.warning("structured fallback parse failed: %s", exc)
        return None


async def _structured_fallback(
    api_key: str, original_model: str, maya_text: str
) -> FinalAnswer | None:
    """Two-stage recovery: same model at temp=0, then escalate to the
    stronger model from the same provider (Gemini → 2.5-pro, OpenAI →
    gpt-4o)."""
    provider = model_provider(original_model)
    fallback_model = PROVIDER_FALLBACK.get(provider or "google", "gemini-2.5-pro")

    try:
        llm_strict = _make_llm(api_key, original_model, temperature=0.0)
        result = await _try_extract_json(llm_strict, maya_text)
        if result:
            log.info("structured fallback recovered (same model, temp=0)")
            return result
    except Exception as exc:  # noqa: BLE001
        log.warning("structured fallback level 1 init failed: %s", exc)

    if original_model != fallback_model:
        try:
            log.info("escalating structured fallback to %s", fallback_model)
            llm_pro = _make_llm(api_key, fallback_model, temperature=0.0)
            result = await _try_extract_json(llm_pro, maya_text)
            if result:
                log.info("structured fallback recovered via %s", fallback_model)
                return result
        except Exception as exc:  # noqa: BLE001
            log.warning("structured fallback level 2 failed: %s", exc)

    return None


def _heuristic_answer(text: str) -> FinalAnswer:
    upper = text.upper()
    if "NO-GO" in upper or "NO GO" in upper:
        verdict, score = "NO-GO", 80
    elif "CONDITIONAL" in upper:
        verdict, score = "CONDITIONAL-GO", 55
    else:
        verdict, score = "GO", 30
    return FinalAnswer(
        verdict=verdict,  # type: ignore[arg-type]
        risk_score=score,
        headline="Ruling reconstructed heuristically — review the Judge transcript.",
        risks=[
            RiskRow(risk="Structured ruling parse failed twice",
                    likelihood="Medium", impact="Medium",
                    mitigation="Read Maya's transcript above directly."),
            RiskRow(risk="Verdict inferred from prose keywords",
                    likelihood="Medium", impact="Medium",
                    mitigation="Re-run with the Force re-analyze toggle."),
            RiskRow(risk="Risk matrix unavailable from this run",
                    likelihood="Low", impact="Low",
                    mitigation="Treat this ruling as advisory only."),
            RiskRow(risk="Risk score is a default estimate",
                    likelihood="Low", impact="Low",
                    mitigation="Use the numeric score as a hint only."),
        ],
    )


# ---------------------------------------------------------------------------
# Cache replay
# ---------------------------------------------------------------------------

async def _replay_cached(ws: WebSocket, cached: VerdictCache) -> None:
    try:
        structured = json.loads(cached.structured_json or "{}")
    except json.JSONDecodeError:
        structured = {}

    await _send(ws, {
        "type": "cache_hit",
        "filename": cached.pdf_filename,
        "verdict": cached.verdict,
        "risk_score": cached.risk_score,
        "headline": cached.headline,
        "original_time": cached.execution_time,
        "model": cached.model_used,
        "truncated": cached.truncated,
    })

    for helper_id, text in (
        ("alex", cached.alex_output),
        ("sam",  cached.sam_output),
        ("maya", cached.maya_output),
    ):
        name = HELPERS[helper_id]["name"]
        await _send(ws, {
            "type": "helper_start", "helper": helper_id,
            "name": f"{name} ({HELPERS[helper_id]['role']})",
        })
        step = 64
        for i in range(0, len(text), step):
            await _send(ws, {
                "type": "helper_token", "helper": helper_id,
                "token": text[i:i + step],
            })
            await asyncio.sleep(0.003)
        await _send(ws, {
            "type": "helper_end", "helper": helper_id,
            "chars": len(text),
            "est_tokens": _approx_output_tokens(text),
        })

    await _send(ws, {
        "type": "verdict",
        "verdict": cached.verdict,
        "risk_score": cached.risk_score,
        "headline": cached.headline,
        "structured": structured,
        "cached": True,
        "execution_time": 0.0,
        "original_time": cached.execution_time,
        "total_tokens": cached.total_tokens,
        "input_tokens": cached.input_tokens,
        "output_tokens": cached.output_tokens,
        "model": cached.model_used,
        "estimated_cost_usd": 0.0,
        "original_cost_usd": cached.cost_usd,
    })


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": sorted(ALLOWED_MODELS),
            "default_model": DEFAULT_MODEL,
            "helpers": HELPERS,
            "ocr_available": OCR_AVAILABLE,
            "provider_models": PROVIDER_MODELS,
            "openai_available": OPENAI_AVAILABLE,
        },
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "The Aegis",
        "ocr": OCR_AVAILABLE,
        "openai_provider": OPENAI_AVAILABLE,
        "models": sorted(ALLOWED_MODELS),
        "provider_models": PROVIDER_MODELS,
    }


def _row_to_summary(r: VerdictCache) -> dict[str, Any]:
    return {
        "id": r.id,
        "filename": r.pdf_filename,
        "verdict": r.verdict,
        "risk_score": r.risk_score,
        "headline": r.headline,
        "model": r.model_used,
        "execution_time": r.execution_time,
        "total_tokens": r.total_tokens,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "cost_usd": r.cost_usd,
        "truncated": r.truncated,
        "chunked": r.chunked,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@app.get("/api/history")
async def history(limit: int = 20) -> JSONResponse:
    limit = max(1, min(100, limit))
    db = SessionLocal()
    try:
        rows = (
            db.query(VerdictCache)
            .order_by(VerdictCache.created_at.desc())
            .limit(limit)
            .all()
        )
        return JSONResponse([_row_to_summary(r) for r in rows])
    finally:
        db.close()


@app.get("/api/verdict/{cache_id}")
async def get_verdict(cache_id: int) -> JSONResponse:
    db = SessionLocal()
    try:
        row = db.query(VerdictCache).filter(VerdictCache.id == cache_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        try:
            structured = json.loads(row.structured_json or "{}")
        except json.JSONDecodeError:
            structured = {}
        payload = {
            **_row_to_summary(row),
            "structured": structured,
            "transcripts": {
                "alex": row.alex_output,
                "sam":  row.sam_output,
                "maya": row.maya_output,
            },
        }
        return JSONResponse(payload)
    finally:
        db.close()


@app.delete("/api/cache/{cache_id}")
async def delete_cache(cache_id: int) -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = db.query(VerdictCache).filter(VerdictCache.id == cache_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(row)
        db.commit()
        log.info("cache id=%d deleted", cache_id)
        return {"deleted": cache_id}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# WebSocket pipeline
# ---------------------------------------------------------------------------

@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket) -> None:
    await ws.accept()
    try:
        await _run_pipeline(ws)
    except WebSocketDisconnect:
        log.info("client disconnected")
    except asyncio.CancelledError:
        log.info("pipeline cancelled")
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline fatal: %s", exc)
        try:
            await _send(ws, {"type": "fatal", "message": str(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


async def _run_pipeline(ws: WebSocket) -> None:
    # ---- 1) Read setup frame -------------------------------------------
    meta_raw = await ws.receive_text()
    try:
        meta = json.loads(meta_raw)
    except json.JSONDecodeError:
        await _send(ws, {"type": "fatal", "message": "Bad setup data."})
        return

    api_key = (meta.get("api_key") or "").strip()
    filename = (meta.get("filename") or "document.pdf").strip()
    model = (meta.get("model") or DEFAULT_MODEL).strip()
    force = bool(meta.get("force"))
    mode = (meta.get("mode") or "fast").strip().lower()
    if mode not in ("fast", "full"):
        mode = "fast"

    if not api_key:
        await _send(ws, {"type": "fatal",
                         "message": "Please paste a Gemini or OpenAI API key first."})
        return
    if model not in ALLOWED_MODELS:
        await _send(ws, {"type": "fatal",
                         "message": f"This model is not allowed: {model}"})
        return

    # Validate that the key matches the selected model's provider. Anthropic
    # keys are detected but explicitly rejected since the agent layer is not
    # wired up for Anthropic in this build.
    key_provider = detect_provider(api_key)
    model_prov = model_provider(model)
    if key_provider is None:
        await _send(ws, {"type": "fatal",
                         "message": ("Could not detect a provider from this "
                                     "key. Gemini keys start with 'AIza...', "
                                     "OpenAI keys start with 'sk-...'.")})
        return
    if key_provider == "anthropic":
        await _send(ws, {"type": "fatal",
                         "message": ("Anthropic keys are not supported in "
                                     "this build. Please use a Gemini or "
                                     "OpenAI key.")})
        return
    if key_provider == "openai" and not OPENAI_AVAILABLE:
        await _send(ws, {"type": "fatal",
                         "message": ("OpenAI support is not installed on the "
                                     "server. Run `pip install "
                                     "langchain-openai` and restart.")})
        return
    if key_provider != model_prov:
        await _send(ws, {"type": "fatal",
                         "message": (f"The '{model}' model belongs to the "
                                     f"{model_prov} provider, but the key "
                                     f"you pasted is a {key_provider} key. "
                                     f"Pick a model from the matching "
                                     f"provider group.")})
        return

    log.info("new pipeline: provider=%s model=%s filename=%s force=%s",
             key_provider, model, filename, force)

    # ---- 2) Read PDF ---------------------------------------------------
    await _send(ws, {"type": "status", "message": "Waiting for your file..."})
    pdf_bytes = await ws.receive_bytes()
    await _send(ws, {
        "type": "status",
        "message": f"Reading the PDF ({len(pdf_bytes):,} bytes)...",
    })

    try:
        text, original_chars, truncated, used_ocr = _extract_pdf(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        log.exception("PDF extraction failed")
        await _send(ws, {"type": "fatal",
                         "message": f"Could not read the PDF: {exc}"})
        return

    if not text:
        msg = ("No text found in the PDF. It appears to be scanned. "
               "Install pytesseract and pdf2image (and the tesseract binary) "
               "to enable OCR, or convert the PDF to a text-based PDF first.")
        log.warning("no text extracted from %s", filename)
        await _send(ws, {"type": "fatal", "message": msg})
        return

    if used_ocr:
        await _send(ws, {"type": "status",
                         "message": "Used OCR (scanned document detected)."})

    await _send(ws, {
        "type": "doc_meta",
        "chars": original_chars,
        "analyzed_chars": len(text),
        "truncated": truncated,
        "used_ocr": used_ocr,
        "model": model,
    })

    content_hash = _hash_content(text, model)

    # ---- 3) Cache check ------------------------------------------------
    if not force:
        db = SessionLocal()
        try:
            cached = (
                db.query(VerdictCache)
                .filter(VerdictCache.content_hash == content_hash)
                .first()
            )
            if cached:
                log.info("cache HIT id=%d filename=%s", cached.id, cached.pdf_filename)
                await _replay_cached(ws, cached)
                return
        finally:
            db.close()

    # ---- 4) Initialise LLM ---------------------------------------------
    try:
        llm = _make_llm(api_key, model, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        log.exception("LLM init failed")
        await _send(ws, {"type": "fatal",
                         "message": f"Could not start the model: {exc}"})
        return

    start = time.perf_counter()

    # ---- 5) Optional condensation for very long docs -------------------
    chunked = False
    analysis_text = text
    if truncated:
        await _send(ws, {
            "type": "warning",
            "message": (f"File is very long ({original_chars:,} chars). "
                        f"Condensing the first {MAX_PDF_CHARS:,} via map-reduce "
                        f"before the tribunal."),
        })
        chunked = True
        analysis_text = await _condense_long_document(ws, llm, text)
        await _send(ws, {
            "type": "status",
            "message": f"Condensed to {len(analysis_text):,} chars. Convening tribunal...",
        })
    else:
        await _send(ws, {"type": "status",
                         "message": "Convening the tribunal. 3 agents deploying..."})

    # ---- 6) Run the chosen pipeline ------------------------------------
    # Two modes:
    #   "fast" — three sequential LLM calls (Alex → Sam → Maya). 3 API
    #             requests per analysis, free-tier friendly (the Gemini
    #             free tier is 20 requests/day). Default.
    #   "full" — LangGraph multi-agent system: Planner → Specialists with
    #             tool use over the knowledge base → Alex → Sam (also
    #             tool-using) → Maya → Critique (Alex+Sam parallel) →
    #             optional Maya revision. 8-15+ API requests per run.
    await _send(ws, {"type": "mode", "mode": mode})

    specialists_used: list[str] = []
    specialist_reports: dict[str, str] = {}
    critique_dissent = False
    revision_output = ""
    pre_validated_ruling: dict[str, Any] | None = None

    if mode == "full":
        from agents import build_graph

        async def _emit(payload: dict[str, Any]) -> None:
            await _send(ws, payload)

        graph = build_graph(llm, _emit)
        try:
            final_state = await graph.ainvoke({"document": analysis_text})
        except (WebSocketDisconnect, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("graph execution failed: %s", exc)
            await _send(ws, {
                "type": "fatal",
                "message": f"Tribunal pipeline failed: {exc}",
            })
            return

        alex_out = final_state.get("alex_output", "")
        sam_out = final_state.get("sam_output", "")
        maya_out = final_state.get("final_maya") or final_state.get("maya_output", "")
        specialists_used = final_state.get("selected_specialists", [])
        specialist_reports = final_state.get("specialist_reports", {})
        critique_dissent = bool(final_state.get("needs_revision", False))
        revision_output = final_state.get("revision_output", "")
        pre_validated_ruling = final_state.get("final_ruling")
    else:
        # Fast mode: classic 3-call pipeline. Identical to the original
        # design before the multi-agent refactor.
        alex_prompt = (
            "DOCUMENT UNDER REVIEW:\n\n"
            f"```\n{analysis_text}\n```\n\n"
            "Write your strategic business benefits breakdown now."
        )
        alex_out = await _stream_helper(
            ws, llm, "alex", "Alex (Strategist)",
            ALEX_SYSTEM, alex_prompt,
        )
        sam_prompt = (
            "DOCUMENT UNDER REVIEW:\n\n"
            f"```\n{analysis_text}\n```\n\n"
            "STRATEGIST'S BULLISH ANALYSIS (attack these points):\n\n"
            f"```\n{alex_out}\n```\n\n"
            "Execute the adversarial Red Team attack now."
        )
        sam_out = await _stream_helper(
            ws, llm, "sam", "Sam (Red Team)",
            SAM_SYSTEM, sam_prompt,
        )
        maya_prompt = (
            "DOCUMENT EXCERPT:\n\n"
            f"```\n{analysis_text[:6000]}\n```\n\n"
            "STRATEGIST POSITION:\n\n"
            f"```\n{alex_out}\n```\n\n"
            "RED TEAM REBUTTAL:\n\n"
            f"```\n{sam_out}\n```\n\n"
            "Render your tribunal ruling now. Rationale first, then exactly one "
            "fenced JSON block at the very end."
        )
        maya_out = await _stream_helper(
            ws, llm, "maya", "Maya (Judge)",
            MAYA_SYSTEM, maya_prompt,
        )

    elapsed = round(time.perf_counter() - start, 2)
    log.info(
        "pipeline complete in %.2fs (mode=%s, chunked=%s, specialists=%s, dissent=%s)",
        elapsed, mode, chunked, specialists_used, critique_dissent,
    )

    # ---- 7) Parse structured ruling ------------------------------------
    answer: FinalAnswer | None = None
    if pre_validated_ruling:
        try:
            answer = FinalAnswer.model_validate(pre_validated_ruling)
            log.info("graph-supplied JSON parse succeeded")
        except ValidationError as exc:
            log.warning("graph-supplied JSON failed schema: %s", exc)
            answer = None
    if answer is None:
        raw = _extract_final_json(maya_out)
        if raw:
            try:
                answer = FinalAnswer.model_validate(raw)
                log.info("re-extracted JSON parse succeeded")
            except ValidationError as exc:
                log.warning("re-extracted JSON failed schema: %s", exc)
                answer = None
        else:
            log.info("no JSON block found in Maya's output")

    if answer is None:
        await _send(ws, {
            "type": "status",
            "message": "Structured ruling needs recovery — escalating...",
        })
        answer = await _structured_fallback(api_key, model, maya_out)

    if answer is None:
        log.warning("falling back to heuristic answer")
        answer = _heuristic_answer(maya_out)

    # ---- 10) Cost --------------------------------------------------------
    # tiktoken-counted for OpenAI models, len/4-approximated for Gemini.
    input_tokens = _approx_input_tokens(analysis_text, alex_out, sam_out, model=model)
    output_tokens = (
        _approx_output_tokens(alex_out, model=model)
        + _approx_output_tokens(sam_out, model=model)
        + _approx_output_tokens(maya_out, model=model)
    )
    total_tokens = input_tokens + output_tokens
    cost_usd = _estimate_cost(model, input_tokens, output_tokens)

    # ---- 11) Persist -----------------------------------------------------
    structured_dict = answer.model_dump()
    db = SessionLocal()
    try:
        existing = (
            db.query(VerdictCache)
            .filter(VerdictCache.content_hash == content_hash)
            .first()
        )
        if existing:
            existing.pdf_filename = filename
            existing.model_used = model
            existing.alex_output = alex_out
            existing.sam_output = sam_out
            existing.maya_output = maya_out
            existing.verdict = answer.verdict
            existing.risk_score = answer.risk_score
            existing.headline = answer.headline
            existing.structured_json = json.dumps(structured_dict)
            existing.execution_time = elapsed
            existing.total_tokens = total_tokens
            existing.input_tokens = input_tokens
            existing.output_tokens = output_tokens
            existing.cost_usd = cost_usd
            existing.truncated = truncated
            existing.chunked = chunked
            existing.pdf_chars = original_chars
            existing.specialists_json = json.dumps(specialists_used)
            existing.specialist_reports_json = json.dumps(specialist_reports)
            existing.critique_dissent = critique_dissent
            existing.revision_output = revision_output
        else:
            db.add(VerdictCache(
                pdf_filename=filename,
                content_hash=content_hash,
                model_used=model,
                alex_output=alex_out,
                sam_output=sam_out,
                maya_output=maya_out,
                verdict=answer.verdict,
                risk_score=answer.risk_score,
                headline=answer.headline,
                structured_json=json.dumps(structured_dict),
                execution_time=elapsed,
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                truncated=truncated,
                chunked=chunked,
                pdf_chars=original_chars,
                specialists_json=json.dumps(specialists_used),
                specialist_reports_json=json.dumps(specialist_reports),
                critique_dissent=critique_dissent,
                revision_output=revision_output,
            ))
        db.commit()
    except Exception as exc:  # noqa: BLE001
        log.exception("cache write failed: %s", exc)
        db.rollback()
    finally:
        db.close()

    await _send(ws, {
        "type": "verdict",
        "verdict": answer.verdict,
        "risk_score": answer.risk_score,
        "headline": answer.headline,
        "structured": structured_dict,
        "cached": False,
        "execution_time": elapsed,
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "specialists": specialists_used,
        "critique_dissent": critique_dissent,
        "model": model,
        "estimated_cost_usd": cost_usd,
        "chunked": chunked,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
