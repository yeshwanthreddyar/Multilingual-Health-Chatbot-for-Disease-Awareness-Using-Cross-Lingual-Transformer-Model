from __future__ import annotations

from typing import List, Tuple

from app.knowledge_base.graph import (
    DISEASE_GRAPH,
    get_disease_info,
    get_disease_name,
    get_diseases_by_symptoms,
    get_prevention,
    get_rich_context_for_topk,
    get_vaccination_guidelines,
)
from app.integrations.health_portals import load_health_portal_alerts
from app.integrations import health_portals as hp


def test_knowledge_base_basic_queries() -> None:
    info = get_disease_info("common_cold")
    assert info is not None
    assert info.get("name_en") == "Common Cold"
    assert "R05" in info.get("symptoms", [])

    # Localised name should fall back gracefully
    assert get_disease_name("common_cold", lang="hi")
    assert get_disease_name("common_cold", lang="en")

    prevention = get_prevention("flu")
    assert isinstance(prevention, list)
    assert prevention, "Prevention list for flu should not be empty"

    vaccines = get_vaccination_guidelines("typhoid")
    assert isinstance(vaccines, list)


def test_knowledge_base_ranking_and_enrichment() -> None:
    """
    Verify that symptom-based retrieval and rich-context enrichment return
    diseases that are consistent with the in-memory graph schema.
    """
    symptom_codes: List[str] = ["R05", "R50"]  # cough + fever style symptoms
    ranked = get_diseases_by_symptoms(symptom_codes)
    assert ranked, "Expected at least one disease for common symptoms"

    graph_keys = set(DISEASE_GRAPH.keys())
    for key, score in ranked:
        assert key in graph_keys
        assert 0.0 <= score <= 1.0

    # Enrich top-k predictions with structured KB info
    topk: List[Tuple[str, float]] = ranked[:3]
    enriched = get_rich_context_for_topk(topk)
    assert len(enriched) == len(topk)
    for item in enriched:
        assert item["key"] in graph_keys
        assert isinstance(item["prevention"], list)
        assert isinstance(item["symptoms"], list)


def test_health_portal_alerts_fallback() -> None:
    """
    With default config (no external URLs), the loader should return a list and
    never raise, allowing the system to fall back to mock alerts.
    """
    alerts = load_health_portal_alerts(limit=5)
    assert isinstance(alerts, list)
    # Length may be zero if no feeds configured; that's acceptable here.


def test_health_portal_alerts_normalization(monkeypatch) -> None:
    """
    When external feeds are configured, load_health_portal_alerts should
    normalise heterogeneous upstream JSON structures into a common schema.
    """

    def fake_fetch_json_feed(url: str):
        if url == "WHO_URL":
            return [
                {"id": "who-1", "title": "WHO Alert", "location": "Global", "severity": "HIGH"},
            ]
        if url == "MOHFW_URL":
            return {
                "items": [
                    {"id": "mohfw-1", "headline": "MoHFW Advisory", "state": "India"},
                ]
            }
        if url == "ICMR_URL":
            return {
                "results": [
                    {"id": "icmr-1", "title": "ICMR Update", "region": "National"},
                ]
            }
        return []

    monkeypatch.setattr(hp, "WHO_FEED_URL", "WHO_URL")
    monkeypatch.setattr(hp, "MOHFW_API_URL", "MOHFW_URL")
    monkeypatch.setattr(hp, "ICMR_API_URL", "ICMR_URL")
    monkeypatch.setattr(hp, "_fetch_json_feed", fake_fetch_json_feed)

    alerts = hp.load_health_portal_alerts(limit=10)
    assert alerts, "Expected at least one alert from mocked feeds"

    sources = {a.get("source") for a in alerts}
    assert {"WHO", "MoHFW", "ICMR"}.issubset(sources)

    # Check normalised keys are present for at least one alert
    first = alerts[0]
    assert "id" in first
    assert "title" in first
    assert "region" in first
    assert "severity" in first
    assert "summary" in first
