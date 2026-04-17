"""
Government Data Integration - MOCK.
WHO / MoHFW / ICMR APIs - mocked.
Expose: /alerts (outbreak alerts), /vaccines (schedules), /advisories (public health).
"""
from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime

from app.config import HEALTH_PORTALS_ENABLED
from app.integrations.health_portals import load_health_portal_alerts

# Mock outbreak alerts (aligned with WHO/MoHFW style). Include region for location-based alerts.
MOCK_ALERTS: List[Dict[str, Any]] = [
    {
        "id": "ALT-001",
        "title": "Seasonal flu advisory",
        "region": "National",
        "source": "MoHFW",
        "issued_at": "2024-01-15T00:00:00Z",
        "severity": "medium",
        "summary": "Increased flu activity; vaccination and hygiene advised.",
    },
    {
        "id": "ALT-002",
        "title": "Dengue prevention campaign",
        "region": "Multiple states",
        "source": "ICMR",
        "issued_at": "2024-02-01T00:00:00Z",
        "severity": "medium",
        "summary": "Mosquito control and removal of stagnant water recommended.",
    },
    {
        "id": "ALT-003",
        "title": "Dengue cases in Bengaluru",
        "region": "Bengaluru",
        "source": "BBMP",
        "issued_at": "2024-02-02T00:00:00Z",
        "severity": "high",
        "summary": "Rise in dengue cases in Bengaluru; avoid stagnant water, use mosquito nets.",
    },
    {
        "id": "ALT-004",
        "title": "Heat wave advisory – Hyderabad",
        "region": "Hyderabad",
        "source": "State Health",
        "issued_at": "2024-02-01T00:00:00Z",
        "severity": "medium",
        "summary": "Stay hydrated, avoid outdoor exposure during peak hours.",
    },
    {
        "id": "ALT-005",
        "title": "Respiratory infections – Delhi NCR",
        "region": "Delhi",
        "source": "MoHFW",
        "issued_at": "2024-01-28T00:00:00Z",
        "severity": "medium",
        "summary": "Increase in respiratory illness; masks and hand hygiene recommended.",
    },
]

# Mock vaccination schedules (MoHFW/WHO style)
MOCK_VACCINES: List[Dict[str, Any]] = [
    {
        "id": "VAC-001",
        "name": "Routine Immunization - Children",
        "age_group": "0-5 years",
        "vaccines": ["BCG", "OPV", "DPT", "Measles", "Hep B", "PCV", "Rota"],
        "source": "MoHFW",
    },
    {
        "id": "VAC-002",
        "name": "Influenza (Flu)",
        "age_group": "6 months+",
        "vaccines": ["Annual flu vaccine"],
        "source": "WHO",
    },
    {
        "id": "VAC-003",
        "name": "COVID-19",
        "age_group": "12+",
        "vaccines": ["As per national guidelines"],
        "source": "MoHFW",
    },
]

# Mock public health advisories
MOCK_ADVISORIES: List[Dict[str, Any]] = [
    {
        "id": "ADV-001",
        "title": "Hand hygiene and respiratory etiquette",
        "category": "Prevention",
        "source": "WHO",
        "issued_at": "2024-01-01T00:00:00Z",
        "summary": "Wash hands frequently; cover cough/sneeze; avoid touching face.",
    },
    {
        "id": "ADV-002",
        "title": "Safe drinking water",
        "category": "Prevention",
        "source": "MoHFW",
        "issued_at": "2024-01-10T00:00:00Z",
        "summary": "Use boiled or filtered water; store safely; avoid contamination.",
    },
]


def get_alerts(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Return outbreak/public-health alerts.

    If HEALTH_PORTALS_ENABLED is true and external feeds are configured, their
    alerts are included first, followed by local mock alerts as fallback.
    """
    alerts: List[Dict[str, Any]] = []

    if HEALTH_PORTALS_ENABLED:
        try:
            alerts.extend(load_health_portal_alerts(limit=limit))
        except Exception:
            # Fail closed: rely on mock alerts if external integration fails.
            alerts = []

    # Always append mock alerts (used either as primary or fallback)
    for a in MOCK_ALERTS:
        if len(alerts) >= limit:
            break
        alerts.append(a)

    return alerts[:limit]


def get_vaccines(age_group: str | None = None) -> List[Dict[str, Any]]:
    """Return mock vaccination schedules."""
    if age_group:
        return [v for v in MOCK_VACCINES if age_group.lower() in (v.get("age_group") or "").lower()]
    return MOCK_VACCINES


def get_advisories(category: str | None = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Return mock public health advisories."""
    out = MOCK_ADVISORIES
    if category:
        out = [a for a in out if (a.get("category") or "").lower() == category.lower()]
    return out[:limit]
