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
    FinalAnswer, _extract_final_json, _hash_content, _heuristic_answer,
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
    for needle in ("Alex", "Sam", "Maya", "THE AEGIS", "Your Gemini API Key"):
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
