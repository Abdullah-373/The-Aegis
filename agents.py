"""LangGraph state machine for The Aegis tribunal.

Replaces the previous straight-line Alex→Sam→Maya pipeline with a real
multi-agent system:

  1. PLANNER         scans the document, picks which specialist agents to run,
                     and routes each specialist to its relevant clauses.
  2. SPECIALISTS     (Financial, Legal, Data, Compliance, Operations) — only
                     the ones the planner selected actually run. Each
                     specialist has access to the search_precedent tool and
                     iterates until it stops calling tools.
  3. STRATEGIST      reads all specialist findings and writes the bullish
                     business case (the Alex agent).
  4. RED TEAM        attacks the Strategist's case using specialist findings
                     and its own follow-up precedent searches (the Sam agent).
  5. JUDGE           weighs both sides and emits a fenced JSON ruling
                     validated against a Pydantic schema (the Maya agent).
  6. CRITIQUE        Alex and Sam each get to respond to Maya's ruling. If
                     either dissents substantively, Maya runs a single
                     revision pass; otherwise the original ruling stands.

The whole graph streams its intermediate state over a WebSocket so the
frontend can render each phase live.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable, Literal, TypedDict

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from knowledge_base import search as kb_search
from tools import ALL_TOOLS, tool_call_summary

log = logging.getLogger("aegis.agents")

# ---------------------------------------------------------------------------
# Specialist registry
# ---------------------------------------------------------------------------

SpecialistId = Literal["financial", "legal", "data", "compliance", "operations"]

SPECIALISTS: dict[str, dict[str, str]] = {
    "financial": {
        "name": "Financial Analyst",
        "tag": "fees, payment terms, penalties",
        "focus": (
            "Concentrate on fees, payment terms, late-payment interest, "
            "auto-renewal pricing uplifts, refundability of prepayments, "
            "early-termination penalties, and any clause that affects "
            "long-term cost. Quote exact dollar amounts and percentages "
            "from the document."
        ),
    },
    "legal": {
        "name": "Legal Counsel",
        "tag": "liability, indemnification, disputes",
        "focus": (
            "Concentrate on the liability cap, indemnification asymmetry, "
            "warranty disclaimers, governing law, dispute-resolution "
            "mechanisms (arbitration, jury-trial waivers, class-action "
            "waivers), and limitation-of-damages clauses. Note any clause "
            "that may be unenforceable as a penalty under typical "
            "jurisdictions."
        ),
    },
    "data": {
        "name": "Data & Privacy Specialist",
        "tag": "data ownership, IP, residency, breach",
        "focus": (
            "Concentrate on data ownership, IP assignment, data-licence "
            "scope (especially perpetual or ML-training licences), data "
            "residency, sub-processor controls, breach-notification "
            "timing, and any clause that affects the customer's GDPR or "
            "similar regulatory posture."
        ),
    },
    "compliance": {
        "name": "Compliance Officer",
        "tag": "certifications, audit, regulation",
        "focus": (
            "Concentrate on commitments to maintain security "
            "certifications (SOC 2, ISO 27001), audit rights, "
            "sub-contractor flow-down obligations, and regulatory "
            "compliance representations. Flag any commitment that is "
            "stated only as current-state and not as ongoing."
        ),
    },
    "operations": {
        "name": "Operations Lead",
        "tag": "SLA, support, transition, exit",
        "focus": (
            "Concentrate on uptime SLAs and the strength of their "
            "remedies, scheduled-maintenance carve-outs, support response "
            "times, transition assistance on termination, data-export "
            "rights and formats, and any operational cliff at the end of "
            "the term."
        ),
    },
}

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class TribunalState(TypedDict, total=False):
    document: str
    selected_specialists: list[str]
    specialist_reports: dict[str, str]   # id -> markdown report
    alex_output: str
    sam_output: str
    maya_output: str
    structured_ruling: dict[str, Any] | None
    critique_alex: str
    critique_sam: str
    needs_revision: bool
    revision_output: str
    final_maya: str
    final_ruling: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the PLANNER. Your job is to read a business
document and decide which specialist analysts should review it.

Available specialists:
- financial    — fees, payment terms, penalties
- legal        — liability, indemnification, disputes
- data         — data ownership, IP, residency, privacy
- compliance   — certifications, audit, regulation
- operations   — SLA, support, transition, exit

Output ONLY a fenced JSON block with this exact shape:

```json
{
  "selected": ["financial", "legal", "..."],
  "reasoning": "<one short sentence per specialist explaining why>"
}
```

Pick between 2 and 5 specialists. Pick a specialist only if the document
contains clauses that fall within that specialist's remit.
"""

SPECIALIST_SYSTEM = """You are the {name} for The Aegis tribunal.

{focus}

You have access to a tool called `search_precedent(query)` that searches a
curated knowledge base of contract risk patterns. Call this tool 1-3 times
for the specific clauses you find concerning, then write your analysis.

Output a markdown report under 250 words. Structure:

## Findings
A numbered list of the most important issues you found, each one citing
the relevant clause and quoting a short fragment. Where a precedent
search returned a useful match, cite it inline (e.g. "matches precedent
on perpetual data licence — see KB entry on `perpetual_data_license`").

## Severity
A single line: `Severity: Low | Medium | High`.
"""

ALEX_SYSTEM = """You are ALEX, the Strategist. You have read a stack of
specialist reports on a contract. Synthesize them into the strongest
possible BULLISH case for proceeding with the deal: revenue potential,
strategic positioning, growth optionality, and any clauses that work in
the customer's favour. Be sharp, specific, and quote clauses or numbers
where useful. Keep your answer under 300 words. Use markdown.
"""

SAM_SYSTEM = """You are SAM, the Red Team. You have read the same
specialist reports and Alex's bullish synthesis. Attack Alex's case. For
each major claim Alex made, quote it and dismantle it using the
specialist findings. If a particular angle was not covered, call the
`search_precedent` tool once or twice to ground your attack in
precedent. Be merciless and specific. Keep your answer under 300 words.
Use markdown.
"""

MAYA_SYSTEM = """You are MAYA, the Judge. You have heard the specialists,
Alex, and Sam. Render the final ruling.

OUTPUT FORMAT (STRICT):

First, write a `## RATIONALE` section with 2-4 sentences weighing both
sides.

Then, output EXACTLY ONE fenced JSON code block at the very end:

```json
{
  "verdict": "GO" | "NO-GO" | "CONDITIONAL-GO",
  "risk_score": <integer 0-100>,
  "headline": "<one-line ruling, max 90 chars>",
  "risks": [
    {"risk": "...", "likelihood": "Low"|"Medium"|"High",
     "impact": "Low"|"Medium"|"High", "mitigation": "..."}
  ],
  "conditions": ["...", "..."]
}
```

Rules:
- The JSON MUST be the LAST thing in your response.
- `risks` MUST have at least 4 entries.
- `conditions` MUST be populated only for CONDITIONAL-GO (otherwise []).
- No other JSON blocks anywhere.
"""

CRITIQUE_SYSTEM = """You have just heard MAYA render a final ruling on
a contract. Your job is to react. You are {persona}: argue from your
original position.

If you accept Maya's ruling as reasonable, say so in 1-2 sentences and
start your response with `ACCEPT:`.

If you believe Maya missed something important or got the weighting
wrong, say so in 2-3 sentences and start your response with `DISSENT:`,
then state the specific point Maya overlooked.
"""

REVISION_SYSTEM = """You are MAYA again. Alex and Sam have responded to
your earlier ruling. At least one of them dissented. Read their
responses below and either revise your ruling or restate it with a
stronger justification.

Output a fresh structured ruling in the same JSON-block format as before
(```json ... ```). Place the JSON block at the very end of your
response after a short rationale.
"""


# ---------------------------------------------------------------------------
# Streaming utilities
# ---------------------------------------------------------------------------

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
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


async def _stream_simple(
    llm: ChatGoogleGenerativeAI,
    emit: EmitFn,
    agent_id: str,
    name: str,
    system: str,
    user: str,
    *,
    max_attempts: int = 3,
) -> str:
    """Stream a non-tool-using agent's tokens."""
    await emit({"type": "helper_start", "helper": agent_id, "name": name})
    full = ""
    last_exc_msg = ""
    for attempt in range(1, max_attempts + 1):
        full = ""
        first_token = True
        try:
            messages = [SystemMessage(content=system), HumanMessage(content=user)]
            async for chunk in llm.astream(messages):
                token = getattr(chunk, "content", "") or ""
                if not token:
                    continue
                first_token = False
                full += token
                await emit({"type": "helper_token", "helper": agent_id, "token": token})
            break
        except Exception as exc:  # noqa: BLE001
            last_exc_msg = str(exc)
            if first_token and _is_transient(last_exc_msg) and attempt < max_attempts:
                delay = 1.0 * (2 ** (attempt - 1))
                log.warning("%s transient attempt %d: %s", agent_id, attempt, last_exc_msg)
                await emit({
                    "type": "status",
                    "message": f"Transient error in {agent_id} — retrying...",
                })
                await asyncio.sleep(delay)
                continue
            err = f"\n\n[ERROR: {last_exc_msg}]"
            full += err
            try:
                await emit({"type": "helper_token", "helper": agent_id, "token": err})
            except Exception:  # noqa: BLE001
                pass
            break
    await emit({
        "type": "helper_end", "helper": agent_id,
        "chars": len(full), "est_tokens": max(1, len(full) // 4),
    })
    return full


async def _run_with_tools(
    llm_with_tools: Any,
    emit: EmitFn,
    agent_id: str,
    name: str,
    system: str,
    user: str,
    *,
    max_iterations: int = 4,
) -> str:
    """Run a tool-using agent. Each iteration either streams a final answer
    or issues tool calls that we execute and feed back in.
    """
    await emit({"type": "helper_start", "helper": agent_id, "name": name})

    messages: list[Any] = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    visible_text = ""

    for _iteration in range(max_iterations):
        # Non-streaming for tool-call iterations; stream the final pass.
        ai_msg: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        text_part = ai_msg.content or ""
        if isinstance(text_part, list):
            text_part = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in text_part
            )

        if tool_calls:
            for tc in tool_calls:
                summary = tool_call_summary(tc)
                await emit({
                    "type": "helper_token", "helper": agent_id,
                    "token": f"\n{summary}\n",
                })
                visible_text += f"\n{summary}\n"
                # Execute the tool.
                tool_name = tc.get("name")
                tool_args = tc.get("args") or {}
                tool_id = tc.get("id") or tool_name
                result_text = ""
                for t in ALL_TOOLS:
                    if t.name == tool_name:
                        try:
                            result_text = await t.ainvoke(tool_args)
                        except Exception as exc:  # noqa: BLE001
                            result_text = f"[tool error: {exc}]"
                        break
                else:
                    result_text = f"[unknown tool: {tool_name}]"
                # Show a one-line confirmation in the UI.
                try:
                    parsed = json.loads(result_text)
                    n = len(parsed.get("matches", []))
                    await emit({
                        "type": "helper_token", "helper": agent_id,
                        "token": f"  ↳ {n} match(es) found\n",
                    })
                    visible_text += f"  ↳ {n} match(es) found\n"
                except Exception:  # noqa: BLE001
                    pass
                messages.append(ToolMessage(content=result_text, tool_call_id=tool_id))
            continue

        # No more tool calls — this is the final answer. Stream a synthetic
        # version of it so the UI gets the text token-by-token.
        if text_part:
            visible_text += text_part if not visible_text else "\n" + text_part
            # Stream in chunks for visual effect.
            step = 60
            for i in range(0, len(text_part), step):
                await emit({
                    "type": "helper_token", "helper": agent_id,
                    "token": text_part[i:i + step],
                })
                await asyncio.sleep(0.005)
        break

    await emit({
        "type": "helper_end", "helper": agent_id,
        "chars": len(visible_text),
        "est_tokens": max(1, len(visible_text) // 4),
    })
    return visible_text


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def _node_planner(
    state: TribunalState, llm: ChatGoogleGenerativeAI, emit: EmitFn,
) -> TribunalState:
    user = (
        "DOCUMENT EXCERPT:\n\n"
        f"```\n{state['document'][:9000]}\n```\n\n"
        "Decide which specialists should review this document and explain why."
    )
    raw = await _stream_simple(
        llm, emit, "planner", "Planner",
        PLANNER_SYSTEM, user,
    )
    parsed = _extract_json(raw) or {}
    selected = parsed.get("selected") or []
    # Sanitise and bound.
    valid = [s for s in selected if s in SPECIALISTS]
    if len(valid) < 2:
        valid = ["legal", "financial"]  # safe default
    valid = valid[:5]
    await emit({"type": "planner_decision", "selected": valid})
    return {**state, "selected_specialists": valid, "specialist_reports": {}}


def _make_specialist_node(spec_id: str):
    async def _node(
        state: TribunalState,
        llm: ChatGoogleGenerativeAI,
        emit: EmitFn,
    ) -> TribunalState:
        if spec_id not in state.get("selected_specialists", []):
            return state
        spec = SPECIALISTS[spec_id]
        system = SPECIALIST_SYSTEM.format(name=spec["name"], focus=spec["focus"])
        user = (
            "DOCUMENT UNDER REVIEW:\n\n"
            f"```\n{state['document']}\n```\n\n"
            f"Produce your specialist report as {spec['name']}."
        )
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
        text = await _run_with_tools(
            llm_with_tools, emit,
            f"specialist_{spec_id}", f"{spec['name']}",
            system, user,
        )
        reports = dict(state.get("specialist_reports", {}))
        reports[spec_id] = text
        return {**state, "specialist_reports": reports}
    return _node


def _format_specialist_pack(reports: dict[str, str]) -> str:
    if not reports:
        return "(no specialist reports were produced)"
    parts = []
    for sid, text in reports.items():
        name = SPECIALISTS.get(sid, {}).get("name", sid)
        parts.append(f"## {name}\n{text}")
    return "\n\n".join(parts)


async def _node_alex(
    state: TribunalState, llm: ChatGoogleGenerativeAI, emit: EmitFn,
) -> TribunalState:
    pack = _format_specialist_pack(state.get("specialist_reports", {}))
    user = (
        "SPECIALIST REPORTS:\n\n"
        f"{pack}\n\n"
        "Synthesize the strongest bullish case for the deal."
    )
    text = await _stream_simple(
        llm, emit, "alex", "Alex (Strategist)",
        ALEX_SYSTEM, user,
    )
    return {**state, "alex_output": text}


async def _node_sam(
    state: TribunalState, llm: ChatGoogleGenerativeAI, emit: EmitFn,
) -> TribunalState:
    pack = _format_specialist_pack(state.get("specialist_reports", {}))
    user = (
        "SPECIALIST REPORTS:\n\n"
        f"{pack}\n\n"
        "ALEX'S BULLISH CASE:\n\n"
        f"{state.get('alex_output', '')}\n\n"
        "Attack Alex's case using the specialist findings. Call "
        "search_precedent if you need to ground a specific claim."
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    text = await _run_with_tools(
        llm_with_tools, emit,
        "sam", "Sam (Red Team)",
        SAM_SYSTEM, user,
    )
    return {**state, "sam_output": text}


async def _node_maya(
    state: TribunalState, llm: ChatGoogleGenerativeAI, emit: EmitFn,
) -> TribunalState:
    pack = _format_specialist_pack(state.get("specialist_reports", {}))
    user = (
        "SPECIALIST REPORTS:\n\n"
        f"{pack}\n\n"
        "ALEX (Strategist):\n\n"
        f"{state.get('alex_output', '')}\n\n"
        "SAM (Red Team):\n\n"
        f"{state.get('sam_output', '')}\n\n"
        "Render your tribunal ruling now."
    )
    text = await _stream_simple(
        llm, emit, "maya", "Maya (Judge)",
        MAYA_SYSTEM, user,
    )
    return {**state, "maya_output": text, "structured_ruling": _extract_json(text)}


async def _node_critique(
    state: TribunalState, llm: ChatGoogleGenerativeAI, emit: EmitFn,
) -> TribunalState:
    """Alex and Sam respond to Maya's ruling, in parallel."""
    ruling = state.get("maya_output", "")

    async def _crit(agent_id: str, persona: str, name: str) -> str:
        user = f"MAYA'S RULING:\n\n{ruling}\n\nYour response:"
        return await _stream_simple(
            llm, emit, agent_id, name,
            CRITIQUE_SYSTEM.format(persona=persona), user,
        )

    alex_resp, sam_resp = await asyncio.gather(
        _crit("critique_alex", "the Strategist who argued the bullish case",
              "Alex — Critique"),
        _crit("critique_sam",  "the Red Team who attacked the bullish case",
              "Sam — Critique"),
    )
    needs = any(r.strip().upper().startswith("DISSENT") for r in (alex_resp, sam_resp))
    await emit({"type": "critique_outcome", "needs_revision": needs})
    return {
        **state,
        "critique_alex": alex_resp,
        "critique_sam": sam_resp,
        "needs_revision": needs,
    }


async def _node_revise(
    state: TribunalState, llm: ChatGoogleGenerativeAI, emit: EmitFn,
) -> TribunalState:
    user = (
        "YOUR EARLIER RULING:\n\n"
        f"{state.get('maya_output', '')}\n\n"
        "ALEX (response):\n\n"
        f"{state.get('critique_alex', '')}\n\n"
        "SAM (response):\n\n"
        f"{state.get('critique_sam', '')}\n\n"
        "Issue your revised or restated ruling now."
    )
    text = await _stream_simple(
        llm, emit, "maya_revised", "Maya (Revised Ruling)",
        REVISION_SYSTEM, user,
    )
    return {
        **state,
        "revision_output": text,
        "final_maya": text,
        "final_ruling": _extract_json(text),
    }


async def _node_finalise(state: TribunalState) -> TribunalState:
    if state.get("needs_revision"):
        return {
            **state,
            "final_maya": state.get("revision_output", ""),
            "final_ruling": state.get("final_ruling")
                            or _extract_json(state.get("revision_output", "")),
        }
    return {
        **state,
        "final_maya": state.get("maya_output", ""),
        "final_ruling": state.get("structured_ruling"),
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(llm: ChatGoogleGenerativeAI, emit: EmitFn):
    """Build a compiled LangGraph that runs the full tribunal pipeline."""

    # LangGraph inspects the node callable to decide whether to await it.
    # A sync lambda that *returns* a coroutine is NOT recognised as a
    # coroutine function — langgraph passes the coroutine downstream
    # unawaited and the StateGraph reducer then complains it got a
    # coroutine instead of a dict. Every node has to be wrapped in a
    # real `async def` closure so the runtime introspection works.
    def _bind(node_fn):
        async def _wrapped(state):
            return await node_fn(state, llm, emit)
        _wrapped.__name__ = f"bound_{getattr(node_fn, '__name__', 'node')}"
        return _wrapped

    g = StateGraph(TribunalState)

    g.add_node("planner", _bind(_node_planner))
    for sid in SPECIALISTS:
        g.add_node(f"spec_{sid}", _bind(_make_specialist_node(sid)))
    g.add_node("alex", _bind(_node_alex))
    g.add_node("sam", _bind(_node_sam))
    g.add_node("maya", _bind(_node_maya))
    g.add_node("critique", _bind(_node_critique))
    g.add_node("revise", _bind(_node_revise))
    g.add_node("finalise", _node_finalise)

    g.set_entry_point("planner")

    # planner → first selected specialist
    def _after_planner(state: TribunalState) -> str:
        for sid in state.get("selected_specialists", []):
            return f"spec_{sid}"
        return "alex"

    g.add_conditional_edges("planner", _after_planner,
                            {f"spec_{sid}": f"spec_{sid}" for sid in SPECIALISTS}
                            | {"alex": "alex"})

    # each specialist routes to the next selected specialist, or to alex
    def _next_specialist(after: str) -> Callable[[TribunalState], str]:
        def _route(state: TribunalState) -> str:
            selected = state.get("selected_specialists", [])
            try:
                idx = selected.index(after)
            except ValueError:
                return "alex"
            if idx + 1 < len(selected):
                return f"spec_{selected[idx + 1]}"
            return "alex"
        return _route

    for sid in SPECIALISTS:
        mapping = {f"spec_{other}": f"spec_{other}" for other in SPECIALISTS} | {"alex": "alex"}
        g.add_conditional_edges(f"spec_{sid}", _next_specialist(sid), mapping)

    g.add_edge("alex", "sam")
    g.add_edge("sam", "maya")
    g.add_edge("maya", "critique")
    g.add_conditional_edges(
        "critique",
        lambda s: "revise" if s.get("needs_revision") else "finalise",
        {"revise": "revise", "finalise": "finalise"},
    )
    g.add_edge("revise", "finalise")
    g.add_edge("finalise", END)

    return g.compile()
