from __future__ import annotations

from typing import List, Dict

from fastapi.testclient import TestClient

from main import app
from app.integrations.government_mock import MOCK_ALERTS
from app.integrations.alert_sender import region_matches
from app.outbreak import case_records


client = TestClient(app)


def test_health_endpoint_ok() -> None:
    """Basic health-check endpoint should respond with status ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert data.get("service") == "healthbot"


def test_region_matching_and_mock_alerts() -> None:
    """
    Unit-level check for region matching logic and presence of mock alerts.
    Ensures external integrations for alerts are wired with reasonable defaults.
    """
    # region_matches should always accept National / Multiple states for any region
    assert region_matches("National", "Bengaluru")
    assert region_matches("Multiple states", "Hyderabad")

    # For a specific mock alert region, only close matches should be true
    assert region_matches("Bengaluru", "Bengaluru")
    assert not region_matches("Hyderabad", "Bengaluru")

    # Sanity check that mock alerts are non-empty and have required fields
    assert MOCK_ALERTS
    for alert in MOCK_ALERTS:
        assert "title" in alert
        assert "region" in alert
        assert "source" in alert


def test_alerts_endpoint_includes_outbreak_alerts(tmp_path) -> None:
    """
    Integration test: /api/alerts should merge HealthBot outbreak alerts
    with government-style alerts when enough recent cases are recorded.
    """
    # Isolate case storage to a temporary file
    case_records.CASES_FILE = tmp_path / "cases_test.json"

    # Add enough recent cases to cross the outbreak threshold
    for _ in range(6):
        case_records.add_case(region="Bengaluru", disease="dengue")

    resp = client.get("/api/alerts", params={"limit": 10})
    assert resp.status_code == 200
    alerts: List[Dict[str, object]] = resp.json()
    assert alerts

    sources = {a.get("source") for a in alerts}
    # At least one alert should come from the HealthBot outbreak detector
    assert "HealthBot Outbreak" in sources
    # And at least one from the mocked government-style feeds
    assert any(s in {"MoHFW", "ICMR", "WHO", "State Health", "BBMP"} for s in sources if isinstance(s, str))


def test_alerts_by_location_filters_region() -> None:
    """
    /api/alerts/by-location should prefer alerts that match the requested region,
    while still falling back to broader alerts when necessary.
    """
    resp = client.get("/api/alerts/by-location", params={"region": "Bengaluru", "limit": 10})
    assert resp.status_code == 200
    alerts: List[Dict[str, object]] = resp.json()
    assert alerts

    for alert in alerts:
        region = (alert.get("region") or "").lower()
        # Either region-specific, or a broad national/multi-state alert
        assert "bengaluru" in region or region in ("national", "multiple states", "delhi")


def test_chat_offline_basic_flow() -> None:
    """
    End-to-end API test: /api/chat-offline should process a simple symptom
    query and return a non-empty response with intent and language fields.
    Uses only the offline path (no Ollama dependency).
    """
    payload = {
        "message": "I have fever and cough",
        "session_id": "pytest-session",
        "location": "Bengaluru",
    }
    resp = client.post("/api/chat-offline", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert isinstance(data.get("response"), str)
    assert data["response"].strip()
    assert isinstance(data.get("intent"), str)
    assert isinstance(data.get("lang"), str)
    assert data.get("session_id") == "pytest-session"
    # Emergency flag should always be present (may be True/False)
    assert "is_emergency" in data

