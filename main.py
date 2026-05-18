"""The Aegis - Multi-Agent AI Tribunal for Contract / Document Risk Analysis.

FastAPI backend with a streaming WebSocket pipeline that orchestrates three
LangChain-powered agents (Strategist, Red Team, Judge) against a user-supplied
PDF, using a per-request BYOK Gemini API key.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

from database import SessionLocal, VerdictCache, init_db

app = FastAPI(title="The Aegis", description="Multi-Agent AI Tribunal")
templates = Jinja2Templates(directory="templates")

init_db()


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

STRATEGIST_SYSTEM = (
    "You are THE STRATEGIST. You are a McKinsey-grade business consultant "
    "embedded in a corporate war room. Read the supplied document and "
    "produce a sharp, numbered breakdown of the strategic BUSINESS BENEFITS, "
    "upside scenarios, revenue levers, market positioning gains, and "
    "competitive advantages. Be confident, optimistic, and concrete. "
    "Cite clauses where useful. Keep output under 350 words and use markdown."
)

RED_TEAM_SYSTEM = (
    "You are THE RED TEAM. You are a ruthless adversarial general counsel "
    "with a black-hat mindset. The Strategist just made the bullish case for "
    "this document. ATTACK IT. Find every legal loophole, liability trap, "
    "indemnification landmine, ambiguous clause, jurisdictional risk, "
    "exit penalty, IP exposure, and worst-case scenario. Quote the "
    "Strategist's claims and dismantle them one by one. Be merciless. "
    "Keep output under 350 words and use markdown."
)

JUDGE_SYSTEM = (
    "You are THE JUDGE. You have heard the Strategist and the Red Team. "
    "Render the final tribunal ruling. Output MUST contain exactly these "
    "sections in this order:\n\n"
    "## RISK MATRIX\n"
    "A markdown table with columns: Risk | Likelihood (Low/Med/High) | "
    "Impact (Low/Med/High) | Mitigation. Minimum 4 rows.\n\n"
    "## FINAL VERDICT\n"
    "Exactly one line, in the form: `VERDICT: GO` or `VERDICT: NO-GO` or "
    "`VERDICT: CONDITIONAL-GO`.\n\n"
    "## RATIONALE\n"
    "2-4 sentences explaining the call.\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(parts).strip()
    # Trim aggressively to keep prompts small / fast.
    return text[:18000]


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _parse_verdict(judge_text: str) -> str:
    upper = judge_text.upper()
    for token in ("NO-GO", "NO GO", "CONDITIONAL-GO", "CONDITIONAL GO", "GO"):
        marker = f"VERDICT: {token}"
        if marker in upper:
            return token.replace(" ", "-")
    # Fallback heuristic
    if "NO-GO" in upper or "NO GO" in upper:
        return "NO-GO"
    if "CONDITIONAL" in upper:
        return "CONDITIONAL-GO"
    return "GO"


async def _send(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload))


async def _stream_agent(
    ws: WebSocket,
    llm: ChatGoogleGenerativeAI,
    agent_id: str,
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Stream a single agent's tokens over the WebSocket and return full text."""
    await _send(ws, {"type": "agent_start", "agent": agent_id, "name": agent_name})

    full = ""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    try:
        async for chunk in llm.astream(messages):
            token = getattr(chunk, "content", "") or ""
            if not token:
                continue
            full += token
            await _send(ws, {"type": "agent_token", "agent": agent_id, "token": token})
    except Exception as exc:
        err = f"\n\n[STREAM ERROR: {exc!s}]"
        full += err
        await _send(ws, {"type": "agent_token", "agent": agent_id, "token": err})

    await _send(ws, {"type": "agent_end", "agent": agent_id})
    return full


async def _replay_cached(ws: WebSocket, cached: VerdictCache) -> None:
    """Replay a cached run as if it were live, but very fast."""
    await _send(ws, {"type": "cache_hit", "filename": cached.pdf_filename,
                     "verdict": cached.verdict, "original_time": cached.execution_time})

    for agent_id, agent_name, text in (
        ("strategist", "Agent 1 :: The Strategist", cached.strategist_output),
        ("red_team", "Agent 2 :: The Red Team", cached.red_team_output),
        ("judge", "Agent 3 :: The Judge", cached.judge_output),
    ):
        await _send(ws, {"type": "agent_start", "agent": agent_id, "name": agent_name})
        # Stream cached text in small chunks for visual effect, but very quickly.
        step = 48
        for i in range(0, len(text), step):
            await _send(ws, {"type": "agent_token", "agent": agent_id,
                             "token": text[i:i + step]})
            await asyncio.sleep(0.005)
        await _send(ws, {"type": "agent_end", "agent": agent_id})

    await _send(ws, {"type": "verdict", "verdict": cached.verdict,
                     "cached": True, "execution_time": 0.0,
                     "original_time": cached.execution_time})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "operational", "service": "The Aegis"}


@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket) -> None:
    await ws.accept()
    try:
        await _run_pipeline(ws)
    except WebSocketDisconnect:
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
    # ---- 1) Receive metadata (API key + filename) -----------------------
    meta_raw = await ws.receive_text()
    try:
        meta = json.loads(meta_raw)
    except json.JSONDecodeError:
        await _send(ws, {"type": "fatal", "message": "Invalid metadata JSON."})
        return

    api_key = (meta.get("api_key") or "").strip()
    filename = (meta.get("filename") or "document.pdf").strip()

    if not api_key:
        await _send(ws, {"type": "fatal",
                         "message": "Missing Gemini API key. BYOK is required."})
        return

    # ---- 2) Receive PDF binary -----------------------------------------
    await _send(ws, {"type": "status", "message": "Awaiting encrypted payload..."})
    pdf_bytes = await ws.receive_bytes()

    await _send(ws, {"type": "status", "message": "Decrypting & extracting PDF..."})
    try:
        text = _extract_pdf_text(pdf_bytes)
    except Exception as exc:
        await _send(ws, {"type": "fatal", "message": f"PDF extraction failed: {exc}"})
        return

    if not text:
        await _send(ws, {"type": "fatal",
                         "message": "No extractable text in PDF."})
        return

    content_hash = _hash_content(text)

    # ---- 3) Cache check -------------------------------------------------
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

    # ---- 4) Initialize LLM with BYOK key --------------------------------
    try:
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-1.5-flash",
            temperature=0.4,
        )
    except Exception as exc:
        await _send(ws, {"type": "fatal",
                         "message": f"Failed to initialize Gemini: {exc}"})
        return

    start = time.perf_counter()

    await _send(ws, {"type": "status",
                     "message": "Convening the tribunal. 3 agents deploying..."})

    # ---- 5) Agent 1 : Strategist ---------------------------------------
    strategist_prompt = (
        "DOCUMENT UNDER REVIEW:\n\n"
        f"```\n{text}\n```\n\n"
        "Produce the strategic business benefits analysis as instructed."
    )
    strategist_out = await _stream_agent(
        ws, llm, "strategist", "Agent 1 :: The Strategist",
        STRATEGIST_SYSTEM, strategist_prompt,
    )

    # ---- 6) Agent 2 : Red Team -----------------------------------------
    red_team_prompt = (
        "DOCUMENT UNDER REVIEW:\n\n"
        f"```\n{text}\n```\n\n"
        "THE STRATEGIST'S BULLISH ANALYSIS (attack this):\n\n"
        f"```\n{strategist_out}\n```\n\n"
        "Now execute the adversarial Red Team attack as instructed."
    )
    red_team_out = await _stream_agent(
        ws, llm, "red_team", "Agent 2 :: The Red Team",
        RED_TEAM_SYSTEM, red_team_prompt,
    )

    # ---- 7) Agent 3 : Judge --------------------------------------------
    judge_prompt = (
        "DOCUMENT UNDER REVIEW (excerpt):\n\n"
        f"```\n{text[:6000]}\n```\n\n"
        "STRATEGIST POSITION:\n\n"
        f"```\n{strategist_out}\n```\n\n"
        "RED TEAM REBUTTAL:\n\n"
        f"```\n{red_team_out}\n```\n\n"
        "Render your tribunal ruling now."
    )
    judge_out = await _stream_agent(
        ws, llm, "judge", "Agent 3 :: The Judge",
        JUDGE_SYSTEM, judge_prompt,
    )

    elapsed = round(time.perf_counter() - start, 2)
    verdict = _parse_verdict(judge_out)

    # ---- 8) Persist to cache -------------------------------------------
    db = SessionLocal()
    try:
        existing = (
            db.query(VerdictCache)
            .filter(VerdictCache.pdf_filename == filename)
            .first()
        )
        if existing:
            existing.content_hash = content_hash
            existing.strategist_output = strategist_out
            existing.red_team_output = red_team_out
            existing.judge_output = judge_out
            existing.verdict = verdict
            existing.execution_time = elapsed
        else:
            db.add(VerdictCache(
                pdf_filename=filename,
                content_hash=content_hash,
                strategist_output=strategist_out,
                red_team_output=red_team_out,
                judge_output=judge_out,
                verdict=verdict,
                execution_time=elapsed,
            ))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

    await _send(ws, {"type": "verdict", "verdict": verdict,
                     "cached": False, "execution_time": elapsed})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
