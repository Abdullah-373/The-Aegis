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
import re
import time
from typing import Any, Literal

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
    log.info("OCR available (pytesseract + pdf2image installed)")
except Exception:  # noqa: BLE001
    OCR_AVAILABLE = False
    log.info(
        "OCR not available — install pytesseract, pdf2image and the tesseract "
        "binary to enable scanned-PDF support"
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Raised from 18k. Gemini 2.x models support large contexts; only documents
# longer than this trigger the map-reduce condensation path.
MAX_PDF_CHARS = 100_000
CHUNK_SIZE = 70_000
CHUNK_OVERLAP = 1_500

ALLOWED_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
}
DEFAULT_MODEL = "gemini-2.5-flash"

# Model used to recover a structured ruling when the primary parse fails.
FALLBACK_MODEL = "gemini-2.5-pro"

# Real Gemini prices in USD per 1M tokens (input, output).
MODEL_PRICES = {
    "gemini-2.0-flash":      (0.10,  0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-flash":      (0.30,  2.50),
    "gemini-2.5-flash-lite": (0.10,  0.40),
    "gemini-2.5-pro":        (1.25, 10.00),
}

HELPERS = {
    "alex": {"name": "Alex", "role": "Strategist", "tag": "finds the upside"},
    "sam":  {"name": "Sam",  "role": "Red Team",   "tag": "finds the problems"},
    "maya": {"name": "Maya", "role": "Judge",      "tag": "renders the ruling"},
}

app = FastAPI(title="The Aegis", description="Three AI agents review your PDF")
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
  "risk_score": <integer 0-100, where 0=safe and 100=catastrophic>,
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
        )
    )


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = MODEL_PRICES.get(model, MODEL_PRICES[DEFAULT_MODEL])
    return round(
        (input_tokens * in_price + output_tokens * out_price) / 1_000_000, 6
    )


def _approx_input_tokens(*texts: str) -> int:
    return max(1, sum(len(t) for t in texts) // 4)


def _approx_output_tokens(text: str) -> int:
    return max(1, len(text) // 4)


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
            if first_token and transient and attempt < max_attempts:
                delay = 1.0 * (2 ** (attempt - 1))
                log.warning(
                    "%s attempt %d/%d transient: %s — retrying in %.1fs",
                    helper_id, attempt, max_attempts, last_exc_msg, delay,
                )
                await _send(ws, {
                    "type": "status",
                    "message": f"Transient error in {helper_id} — retrying...",
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
    """Two-stage recovery: same model at temp=0, then escalate to pro."""
    try:
        llm_strict = ChatGoogleGenerativeAI(
            google_api_key=api_key, model=original_model, temperature=0.0,
        )
        result = await _try_extract_json(llm_strict, maya_text)
        if result:
            log.info("structured fallback recovered (same model, temp=0)")
            return result
    except Exception as exc:  # noqa: BLE001
        log.warning("structured fallback level 1 init failed: %s", exc)

    if original_model != FALLBACK_MODEL:
        try:
            log.info("escalating structured fallback to %s", FALLBACK_MODEL)
            llm_pro = ChatGoogleGenerativeAI(
                google_api_key=api_key, model=FALLBACK_MODEL, temperature=0.0,
            )
            result = await _try_extract_json(llm_pro, maya_text)
            if result:
                log.info("structured fallback recovered via %s", FALLBACK_MODEL)
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
        },
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "The Aegis",
        "ocr": OCR_AVAILABLE,
        "models": sorted(ALLOWED_MODELS),
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

    if not api_key:
        await _send(ws, {"type": "fatal",
                         "message": "Please add your Gemini API key first."})
        return
    if model not in ALLOWED_MODELS:
        await _send(ws, {"type": "fatal",
                         "message": f"This model is not allowed: {model}"})
        return

    log.info("new pipeline: model=%s filename=%s force=%s", model, filename, force)

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
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key, model=model, temperature=0.4,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Gemini init failed")
        await _send(ws, {"type": "fatal",
                         "message": f"Could not start Gemini: {exc}"})
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

    # ---- 6) Alex --------------------------------------------------------
    alex_prompt = (
        "DOCUMENT UNDER REVIEW:\n\n"
        f"```\n{analysis_text}\n```\n\n"
        "Write your strategic business benefits breakdown now."
    )
    alex_out = await _stream_helper(
        ws, llm, "alex", "Alex (Strategist)",
        ALEX_SYSTEM, alex_prompt,
    )

    # ---- 7) Sam ---------------------------------------------------------
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

    # ---- 8) Maya --------------------------------------------------------
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
    log.info("tribunal complete in %.2fs (chunked=%s)", elapsed, chunked)

    # ---- 9) Parse structured ruling ------------------------------------
    raw = _extract_final_json(maya_out)
    answer: FinalAnswer | None = None
    if raw:
        try:
            answer = FinalAnswer.model_validate(raw)
            log.info("primary JSON parse succeeded")
        except ValidationError as exc:
            log.warning("primary JSON failed schema: %s", exc)
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
    input_tokens = _approx_input_tokens(analysis_text, alex_out, sam_out)
    output_tokens = (
        _approx_output_tokens(alex_out)
        + _approx_output_tokens(sam_out)
        + _approx_output_tokens(maya_out)
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
        "model": model,
        "estimated_cost_usd": cost_usd,
        "chunked": chunked,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
