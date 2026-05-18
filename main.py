"""The Aegis - three AI helpers (Alex, Sam, Maya) that check a PDF together.

A FastAPI app. The user brings their own Gemini API key (BYOK).
Alex finds the good points. Sam finds the bad points. Maya makes the
final choice and gives a clear JSON answer. Answers are saved in
SQLite, so the same PDF returns the same answer very fast next time.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
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
# Config
# ---------------------------------------------------------------------------

MAX_PDF_CHARS = 18_000
ALLOWED_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
}
DEFAULT_MODEL = "gemini-2.5-flash"

# Rough USD per 1K tokens. Used only for the UI cost hint.
MODEL_COST_PER_1K = {
    "gemini-2.0-flash": 0.0002,
    "gemini-2.0-flash-lite": 0.0001,
    "gemini-2.5-flash": 0.0003,
    "gemini-2.5-flash-lite": 0.0001,
    "gemini-2.5-pro": 0.005,
}

# Human-friendly names for the three helpers.
HELPERS = {
    "alex": {"name": "Alex", "role": "The Helper", "tag": "finds the good points"},
    "sam":  {"name": "Sam",  "role": "The Checker", "tag": "finds the problems"},
    "maya": {"name": "Maya", "role": "The Judge",  "tag": "makes the choice"},
}

app = FastAPI(title="The Aegis", description="Three AI helpers check your PDF")
templates = Jinja2Templates(directory="templates")

init_db()


# ---------------------------------------------------------------------------
# Structured answer schema
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
# Helper prompts (B1 English)
# ---------------------------------------------------------------------------

ALEX_SYSTEM = (
    "You are ALEX. You look for the good things in business papers. "
    "Read the document and write a short, clear list of why this is a "
    "good deal. Talk about how it can help make money, help the company "
    "grow, and beat other companies. Be confident. Use simple words "
    "(B1 English). Keep your answer under 300 words. Use markdown."
)

SAM_SYSTEM = (
    "You are SAM. You look for problems. Alex just said why this deal "
    "is good. Now you must find what is BAD. Look at the document and "
    "find: hidden traps, unclear rules, things that can go wrong, big "
    "costs, and legal problems. Be tough. Take Alex's main points and "
    "show why they may be wrong or risky. Use simple words (B1 English). "
    "Keep your answer under 300 words. Use markdown."
)

MAYA_SYSTEM = """You are MAYA. You heard Alex and Sam. Now you must decide.

WRITE YOUR ANSWER LIKE THIS:

First, write a short section called `## REASON` with 2 to 4 short sentences.
Say why you chose your answer. Compare what Alex said and what Sam said.
Use simple words (B1 English).

Then, at the very end, write ONE block of JSON inside code fences.
The JSON must match this shape exactly:

```json
{
  "verdict": "GO" or "NO-GO" or "CONDITIONAL-GO",
  "risk_score": a whole number from 0 to 100 (0 means safe, 100 means very bad),
  "headline": "one short line, 90 letters max",
  "risks": [
    {
      "risk": "short name of the risk",
      "likelihood": "Low" or "Medium" or "High",
      "impact": "Low" or "Medium" or "High",
      "mitigation": "what to do about it"
    }
  ],
  "conditions": ["thing to fix", "..."]
}
```

Rules:
- The JSON block MUST be the LAST thing in your answer.
- `risks` must have at least 4 items.
- `conditions` should be a list of things to fix. Use it only if the verdict
  is CONDITIONAL-GO. If not, use an empty list [].
- Do not write any other JSON blocks anywhere in your answer.
"""


# ---------------------------------------------------------------------------
# Helpers (text + parsing)
# ---------------------------------------------------------------------------

def _extract_pdf_text(raw: bytes) -> tuple[str, int, bool]:
    reader = PdfReader(io.BytesIO(raw))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    full = "\n".join(parts).strip()
    original = len(full)
    truncated = original > MAX_PDF_CHARS
    return full[:MAX_PDF_CHARS], original, truncated


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


async def _structured_fallback(
    llm: ChatGoogleGenerativeAI, maya_text: str
) -> FinalAnswer | None:
    """If the streamed answer had no valid JSON, ask once more for clean JSON."""
    prompt = (
        "Turn the following text into clean JSON. Output ONLY the JSON, "
        "no fences, no extra words.\n\n"
        "Shape:\n"
        "{verdict: 'GO'|'NO-GO'|'CONDITIONAL-GO', "
        "risk_score: int 0-100, headline: str, "
        "risks: [{risk, likelihood, impact, mitigation}] (min 4 items, "
        "likelihood/impact in {Low,Medium,High}), "
        "conditions: [str] (only if CONDITIONAL-GO)}.\n\n"
        f"TEXT:\n{maya_text}"
    )
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = (resp.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
        data = json.loads(raw)
        return FinalAnswer.model_validate(data)
    except (json.JSONDecodeError, ValidationError, Exception):
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
        headline="Answer made from a simple guess. Please read Maya's text.",
        risks=[
            RiskRow(risk="Clean JSON answer not found",
                    likelihood="Medium", impact="Medium",
                    mitigation="Read Maya's text above by hand."),
            RiskRow(risk="Verdict was guessed from key words",
                    likelihood="Medium", impact="Medium",
                    mitigation="Try a stronger model."),
            RiskRow(risk="No risk table this time",
                    likelihood="Low", impact="Low",
                    mitigation="Turn on 'Run again' to retry."),
            RiskRow(risk="Risk score is a default number",
                    likelihood="Low", impact="Low",
                    mitigation="Use the number as a hint only."),
        ],
    )


async def _send(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload))


async def _stream_helper(
    ws: WebSocket,
    llm: ChatGoogleGenerativeAI,
    helper_id: str,
    helper_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Stream one helper's words over the WebSocket. Return all the words."""
    await _send(ws, {"type": "helper_start", "helper": helper_id, "name": helper_name})

    full = ""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    try:
        async for chunk in llm.astream(messages):
            token = getattr(chunk, "content", "") or ""
            if not token:
                continue
            full += token
            await _send(ws, {
                "type": "helper_token", "helper": helper_id, "token": token,
            })
    except (WebSocketDisconnect, asyncio.CancelledError):
        raise
    except Exception as exc:
        err = f"\n\n[ERROR: {exc!s}]"
        full += err
        try:
            await _send(ws, {"type": "helper_token", "helper": helper_id, "token": err})
        except Exception:
            pass

    await _send(ws, {
        "type": "helper_end", "helper": helper_id,
        "chars": len(full),
        "est_tokens": max(1, len(full) // 4),
    })
    return full


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
        await _send(ws, {"type": "helper_start", "helper": helper_id,
                         "name": f"Helper :: {name}"})
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
            "est_tokens": max(1, len(text) // 4),
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
        "model": cached.model_used,
    })


# ---------------------------------------------------------------------------
# Routes
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
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "The Aegis"}


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
        return JSONResponse([
            {
                "id": r.id,
                "filename": r.pdf_filename,
                "verdict": r.verdict,
                "risk_score": r.risk_score,
                "headline": r.headline,
                "model": r.model_used,
                "execution_time": r.execution_time,
                "total_tokens": r.total_tokens,
                "truncated": r.truncated,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ])
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
        return {"deleted": cache_id}
    finally:
        db.close()


@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket) -> None:
    await ws.accept()
    try:
        await _run_pipeline(ws)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        return
    except Exception as exc:
        try:
            await _send(ws, {"type": "fatal", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
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

    # ---- 2) Read PDF ----------------------------------------------------
    await _send(ws, {"type": "status", "message": "Waiting for your file..."})
    pdf_bytes = await ws.receive_bytes()

    await _send(ws, {"type": "status", "message": "Reading the PDF..."})
    try:
        text, original_chars, truncated = _extract_pdf_text(pdf_bytes)
    except Exception as exc:
        await _send(ws, {"type": "fatal",
                         "message": f"Could not read the PDF: {exc}"})
        return

    if not text:
        await _send(ws, {
            "type": "fatal",
            "message": ("No text found in the PDF. It may be a scan or image. "
                        "This app does not read images."),
        })
        return

    await _send(ws, {
        "type": "doc_meta",
        "chars": original_chars,
        "analyzed_chars": len(text),
        "truncated": truncated,
        "model": model,
    })
    if truncated:
        await _send(ws, {
            "type": "warning",
            "message": (f"The file is long ({original_chars:,} letters). "
                        f"We will only read the first {MAX_PDF_CHARS:,} letters."),
        })

    content_hash = _hash_content(text, model)

    # ---- 3) Check the cache --------------------------------------------
    if not force:
        db = SessionLocal()
        try:
            cached = (
                db.query(VerdictCache)
                .filter(VerdictCache.content_hash == content_hash)
                .first()
            )
            if cached:
                await _replay_cached(ws, cached)
                return
        finally:
            db.close()

    # ---- 4) Start the model --------------------------------------------
    try:
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=0.4,
        )
    except Exception as exc:
        await _send(ws, {"type": "fatal",
                         "message": f"Could not start Gemini: {exc}"})
        return

    start = time.perf_counter()
    await _send(ws, {"type": "status",
                     "message": "Starting the three helpers..."})

    # ---- 5) Alex --------------------------------------------------------
    alex_prompt = (
        "THE DOCUMENT:\n\n"
        f"```\n{text}\n```\n\n"
        "Now write your list of good points."
    )
    alex_out = await _stream_helper(
        ws, llm, "alex", "Helper 1 :: Alex (finds the good points)",
        ALEX_SYSTEM, alex_prompt,
    )

    # ---- 6) Sam ---------------------------------------------------------
    sam_prompt = (
        "THE DOCUMENT:\n\n"
        f"```\n{text}\n```\n\n"
        "WHAT ALEX SAID (attack these points):\n\n"
        f"```\n{alex_out}\n```\n\n"
        "Now write your list of problems."
    )
    sam_out = await _stream_helper(
        ws, llm, "sam", "Helper 2 :: Sam (finds the problems)",
        SAM_SYSTEM, sam_prompt,
    )

    # ---- 7) Maya --------------------------------------------------------
    maya_prompt = (
        "THE DOCUMENT (short part):\n\n"
        f"```\n{text[:6000]}\n```\n\n"
        "WHAT ALEX SAID:\n\n"
        f"```\n{alex_out}\n```\n\n"
        "WHAT SAM SAID:\n\n"
        f"```\n{sam_out}\n```\n\n"
        "Now write your answer. Remember: short reason first, "
        "then ONE JSON block at the very end."
    )
    maya_out = await _stream_helper(
        ws, llm, "maya", "Helper 3 :: Maya (makes the choice)",
        MAYA_SYSTEM, maya_prompt,
    )

    elapsed = round(time.perf_counter() - start, 2)

    # ---- 8) Parse the JSON answer --------------------------------------
    raw = _extract_final_json(maya_out)
    answer: FinalAnswer | None = None
    if raw:
        try:
            answer = FinalAnswer.model_validate(raw)
        except ValidationError:
            answer = None
    if answer is None:
        await _send(ws, {"type": "status",
                         "message": "JSON not clean — trying again..."})
        answer = await _structured_fallback(llm, maya_out)
    if answer is None:
        answer = _heuristic_answer(maya_out)

    total_est_tokens = (
        len(alex_out) // 4 + len(sam_out) // 4 + len(maya_out) // 4
    )

    # ---- 9) Save to cache ----------------------------------------------
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
            existing.total_tokens = total_est_tokens
            existing.truncated = truncated
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
                total_tokens=total_est_tokens,
                truncated=truncated,
                pdf_chars=original_chars,
            ))
        db.commit()
    except Exception:
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
        "total_tokens": total_est_tokens,
        "model": model,
        "estimated_cost_usd": round(
            (total_est_tokens / 1000.0) * MODEL_COST_PER_1K.get(model, 0.0002), 6,
        ),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
