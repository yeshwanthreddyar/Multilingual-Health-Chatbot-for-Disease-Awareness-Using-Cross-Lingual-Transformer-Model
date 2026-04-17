from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    """Basic health-check endpoint should be up and return JSON."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert data.get("service") == "healthbot"


def test_chat_offline_endpoint_end_to_end() -> None:
    """
    End-to-end API check: /api/chat-offline should run the full
    NLP → ML → Dialogue pipeline without requiring Ollama.
    """
    payload = {"message": "I have fever and cough", "session_id": "pytest-api"}
    resp = client.post("/api/chat-offline", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    # Response shape
    assert "response" in data
    assert "intent" in data
    assert "lang" in data
    assert "is_emergency" in data

    assert isinstance(data["response"], str)
    assert data["response"].strip()
    assert isinstance(data["is_emergency"], bool)

