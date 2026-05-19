"""Unit tests that don't hit the Gemini API."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use a throwaway DB for tests.
os.environ.setdefault("AEGIS_TEST", "1")
import database  # noqa: E402
database.DATABASE_URL = f"sqlite:///{tempfile.mkstemp(suffix='.db')[1]}"
database.engine = database.create_engine(
    database.DATABASE_URL, connect_args={"check_same_thread": False}
)
database.SessionLocal.configure(bind=database.engine)

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from main import (  # noqa: E402
    FinalAnswer, _chunk_text, _estimate_cost, _extract_final_json,
    _hash_content, _heuristic_answer, _is_transient, MODEL_PRICES,
)

client = TestClient(main.app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_renders():
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    for needle in ("Alex", "Sam", "Maya", "The Aegis", "Gemini API Key"):
        assert needle in body


def test_history_empty():
    r = client.get("/api/history")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_hash_changes_with_model():
    a = _hash_content("same body", "gemini-2.5-flash")
    b = _hash_content("same body", "gemini-2.5-pro")
    assert a != b


def test_extract_final_json_trailing_block():
    text = (
        "## REASON\n"
        "Maya says ok.\n\n"
        "```json\n"
        '{"verdict":"GO","risk_score":12,"headline":"safe",'
        '"risks":[{"risk":"x","likelihood":"Low","impact":"Low","mitigation":"y"}],'
        '"conditions":[]}\n'
        "```\n"
    )
    data = _extract_final_json(text)
    assert data is not None
    assert data["verdict"] == "GO"
    assert data["risk_score"] == 12


def test_extract_final_json_picks_last_block():
    text = (
        "```json\n{\"verdict\":\"NO-GO\"}\n```\n"
        "## REASON\nthought again\n"
        '```json\n{"verdict":"GO","risk_score":5,"headline":"ok",'
        '"risks":[{"risk":"x","likelihood":"Low","impact":"Low","mitigation":"y"}]}\n```'
    )
    data = _extract_final_json(text)
    assert data["verdict"] == "GO"


def test_extract_final_json_none_when_missing():
    assert _extract_final_json("no json here") is None


def test_heuristic_answer_nogo():
    a = _heuristic_answer("MAYA: this is NO-GO for sure")
    assert a.verdict == "NO-GO"
    assert 0 <= a.risk_score <= 100
    assert len(a.risks) >= 4


def test_heuristic_answer_conditional():
    a = _heuristic_answer("we say CONDITIONAL go with fixes")
    assert a.verdict == "CONDITIONAL-GO"


def test_heuristic_answer_default_go():
    a = _heuristic_answer("everything looks fine")
    assert a.verdict == "GO"


def test_final_answer_validates_severity():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FinalAnswer.model_validate({
            "verdict": "GO", "risk_score": 10, "headline": "x",
            "risks": [
                {"risk": "a", "likelihood": "INVALID", "impact": "Low", "mitigation": "z"},
                {"risk": "b", "likelihood": "Low", "impact": "Low", "mitigation": "z"},
                {"risk": "c", "likelihood": "Low", "impact": "Low", "mitigation": "z"},
            ],
        })


def test_ws_rejects_missing_key():
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(json.dumps({"filename": "x.pdf", "model": "gemini-2.5-flash"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "fatal"
        assert "key" in msg["message"].lower()


def test_ws_rejects_bad_model():
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(json.dumps({
            "api_key": "fake", "filename": "x.pdf", "model": "evil-model",
        }))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "fatal"
        assert "model" in msg["message"].lower()


def test_delete_nonexistent_cache():
    r = client.delete("/api/cache/999999")
    assert r.status_code == 404


def test_get_nonexistent_verdict():
    r = client.get("/api/verdict/999999")
    assert r.status_code == 404


def test_chunk_text_under_size():
    chunks = _chunk_text("hello world", size=100, overlap=10)
    assert chunks == ["hello world"]


def test_chunk_text_splits_with_overlap():
    text = "a" * 200
    chunks = _chunk_text(text, size=80, overlap=20)
    assert len(chunks) >= 3
    # overlap means consecutive chunks share tail/head
    for i in range(len(chunks) - 1):
        assert len(chunks[i]) == 80
        # next chunk starts inside the previous chunk's content
        assert chunks[i][-20:] == chunks[i + 1][:20]


def test_is_transient_recognises_common_errors():
    for s in ("503 Service Unavailable", "429 Too Many Requests",
              "Connection reset by peer", "Deadline Exceeded", "overloaded"):
        assert _is_transient(s), s
    for s in ("401 Unauthorized", "API key not valid", "permission denied"):
        assert not _is_transient(s), s


def test_estimate_cost_uses_per_million_prices():
    # 1M input + 1M output on flash should equal the listed prices summed.
    in_p, out_p = MODEL_PRICES["gemini-2.5-flash"]
    cost = _estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
    assert abs(cost - (in_p + out_p)) < 0.0001


def test_estimate_cost_falls_back_for_unknown_model():
    cost = _estimate_cost("unknown-model", 1_000, 1_000)
    assert cost >= 0  # uses default, doesn't crash


def test_health_reports_ocr_flag():
    r = client.get("/health")
    body = r.json()
    assert "ocr" in body
    assert isinstance(body["ocr"], bool)
    assert "models" in body


def test_get_verdict_returns_full_record_when_seeded():
    db = database.SessionLocal()
    try:
        row = database.VerdictCache(
            pdf_filename="test.pdf",
            content_hash="test-hash-xyz",
            model_used="gemini-2.5-flash",
            alex_output="alex text",
            sam_output="sam text",
            maya_output="maya text",
            verdict="GO",
            risk_score=20,
            headline="all clear",
            structured_json='{"verdict":"GO","risk_score":20,"headline":"all clear","risks":[],"conditions":[]}',
            execution_time=12.3,
            total_tokens=500,
            input_tokens=300,
            output_tokens=200,
            cost_usd=0.0015,
            truncated=False,
            chunked=False,
            pdf_chars=1000,
        )
        db.add(row)
        db.commit()
        cache_id = row.id
    finally:
        db.close()

    r = client.get(f"/api/verdict/{cache_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["verdict"] == "GO"
    assert data["risk_score"] == 20
    assert data["transcripts"]["alex"] == "alex text"
    assert data["transcripts"]["maya"] == "maya text"
    assert data["structured"]["verdict"] == "GO"


# ---------------------------------------------------------------------------
# Multi-agent additions: knowledge base, tools, graph
# ---------------------------------------------------------------------------

from knowledge_base import KNOWLEDGE_BASE, search as kb_search, categories  # noqa: E402
from tools import ALL_TOOLS, tool_call_summary  # noqa: E402
import agents as agents_mod  # noqa: E402


def test_knowledge_base_loaded():
    assert len(KNOWLEDGE_BASE) >= 30
    cats = categories()
    for required in ("liability", "indemnification", "data", "sla"):
        assert required in cats


def test_kb_search_returns_relevant_for_liability_query():
    hits = kb_search("liability cap 3 months fees")
    assert hits, "expected at least one hit"
    ids = [p.id for p in hits]
    assert "liability_cap_short" in ids


def test_kb_search_returns_relevant_for_data_query():
    hits = kb_search("perpetual licence anonymised customer data ML training")
    assert hits
    ids = [p.id for p in hits]
    assert "perpetual_data_license" in ids


def test_kb_search_empty_query():
    assert kb_search("") == []


def test_kb_search_unmatched_query():
    # A query with no overlap with any KB entry should return empty.
    hits = kb_search("xyzzy plover frobnicate")
    assert hits == []


def test_tools_registered():
    names = [t.name for t in ALL_TOOLS]
    assert "search_precedent" in names


def test_search_precedent_tool_returns_json():
    tool = next(t for t in ALL_TOOLS if t.name == "search_precedent")
    result = tool.invoke({"query": "liability cap"})
    data = json.loads(result)
    assert "matches" in data
    assert len(data["matches"]) >= 1
    assert "title" in data["matches"][0]


def test_tool_call_summary_renders_search_call():
    s = tool_call_summary({"name": "search_precedent", "args": {"query": "data licence"}})
    assert "search_precedent" in s
    assert "data licence" in s


def test_specialists_have_required_fields():
    for sid, spec in agents_mod.SPECIALISTS.items():
        assert "name" in spec and "focus" in spec
        assert len(spec["focus"]) > 50  # not a stub


def test_extract_json_helper_finds_last_block():
    text = (
        "## REASON\nsomething\n"
        "```json\n{\"verdict\": \"GO\"}\n```\n"
        "more text\n"
        "```json\n{\"verdict\": \"NO-GO\", \"risk_score\": 90}\n```"
    )
    out = agents_mod._extract_json(text)
    assert out is not None and out["verdict"] == "NO-GO"


def test_build_graph_compiles():
    # Build with a dummy LLM stand-in. We never call ainvoke here, just
    # confirm the graph topology compiles without raising.
    class _DummyLLM:
        def bind_tools(self, _tools):
            return self
    async def _noop(_payload):
        pass
    graph = agents_mod.build_graph(_DummyLLM(), _noop)
    # The compiled graph exposes nodes via its underlying definition.
    node_set = set(graph.get_graph().nodes)
    assert "planner" in node_set
    assert "alex" in node_set and "sam" in node_set and "maya" in node_set
    assert "critique" in node_set and "revise" in node_set
    for sid in agents_mod.SPECIALISTS:
        assert f"spec_{sid}" in node_set
