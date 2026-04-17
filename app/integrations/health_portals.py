from __future__ import annotations

"""
Integration layer for external health portals (MoHFW / WHO / ICMR or proxies).

This module is intentionally conservative: if external feeds are unavailable or
change format, it fails closed and the system falls back to mock alerts.
"""

from typing import Any, Dict, List

import json

import requests

from app.config import (
    MOHFW_API_URL,
    WHO_FEED_URL,
    ICMR_API_URL,
    HEALTH_PORTAL_TIMEOUT_SEC,
)


def _fetch_json_feed(url: str) -> List[Dict[str, Any]]:
    """
    Fetch a JSON-like feed from the given URL.

    The exact schema depends on the upstream provider; we normalise into a list
    of dicts below. On any error, returns [] so callers can safely fall back.
    """
    if not url:
        return []
    try:
        resp = requests.get(url, timeout=HEALTH_PORTAL_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Common patterns: {"items": [...]} or {"results": [...]}
        for key in ("items", "results", "alerts"):
            items = data.get(key)
            if isinstance(items, list):
                return items
    return []


def load_health_portal_alerts(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Load outbreak/public-health style alerts from configured portals.

    Normalised fields:
      - id, title, region, source, severity, summary, issued_at
    """
    alerts: List[Dict[str, Any]] = []

    # WHO feed
    for raw in _fetch_json_feed(WHO_FEED_URL):
        alerts.append(
            {
                "id": str(raw.get("id") or raw.get("guid") or f"WHO-{len(alerts)+1}"),
                "title": raw.get("title") or "WHO health advisory",
                "region": raw.get("region") or raw.get("location") or "Global",
                "source": "WHO",
                "severity": (raw.get("severity") or "info").lower(),
                "summary": raw.get("summary") or raw.get("description") or "",
                "issued_at": raw.get("published") or raw.get("date") or "",
            }
        )

    # MoHFW feed
    for raw in _fetch_json_feed(MOHFW_API_URL):
        alerts.append(
            {
                "id": str(raw.get("id") or f"MoHFW-{len(alerts)+1}"),
                "title": raw.get("title") or raw.get("headline") or "MoHFW advisory",
                "region": raw.get("region") or raw.get("state") or "National",
                "source": "MoHFW",
                "severity": (raw.get("severity") or "info").lower(),
                "summary": raw.get("summary") or raw.get("description") or "",
                "issued_at": raw.get("date") or raw.get("published") or "",
            }
        )

    # ICMR or other national research body
    for raw in _fetch_json_feed(ICMR_API_URL):
        alerts.append(
            {
                "id": str(raw.get("id") or f"ICMR-{len(alerts)+1}"),
                "title": raw.get("title") or "ICMR health update",
                "region": raw.get("region") or "National",
                "source": "ICMR",
                "severity": (raw.get("severity") or "info").lower(),
                "summary": raw.get("summary") or raw.get("description") or "",
                "issued_at": raw.get("date") or "",
            }
        )

    # De-duplicate by id while preserving order
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for a in alerts:
        aid = a.get("id")
        if aid in seen:
            continue
        seen.add(aid)
        deduped.append(a)
        if len(deduped) >= limit:
            break

    return deduped

