"""LangChain tools that specialist agents can call during analysis."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from knowledge_base import search as kb_search


@tool
def search_precedent(query: str) -> str:
    """Search the contract risk precedent knowledge base.

    Returns up to four precedent entries that match the query. Each entry
    contains a pattern description, the risk it implies, and the standard
    mitigation. Use this tool whenever you want to ground a claim about a
    contract clause in a documented precedent rather than relying purely on
    the model's own training.

    Args:
        query: A short natural-language query describing the clause or risk
               you want precedents for, e.g. "liability cap 3 months fees"
               or "perpetual data licence for ML training".
    """
    hits = kb_search(query, k=4)
    if not hits:
        return json.dumps({"matches": [], "note": "no relevant precedents found"})
    return json.dumps({
        "matches": [
            {
                "id": p.id,
                "title": p.title,
                "category": p.category,
                "pattern": p.pattern,
                "risk": p.risk,
                "mitigation": p.mitigation,
            }
            for p in hits
        ]
    })


ALL_TOOLS = [search_precedent]


def tool_call_summary(tool_call: dict[str, Any]) -> str:
    """Render a tool call as a short human-readable line for the UI stream."""
    name = tool_call.get("name", "tool")
    args = tool_call.get("args", {}) or {}
    if name == "search_precedent":
        q = (args.get("query") or "").strip()
        return f"🔎 search_precedent(\"{q}\")"
    return f"🔧 {name}({json.dumps(args)[:80]})"
