"""
Load WHO and Indian health updates from local JSON files.
No external API - manually update files from WHO/MoHFW/ICMR sources.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
WHO_FILE = DATA_DIR / "who_updates.json"
INDIA_FILE = DATA_DIR / "india_health_updates.json"

_updates_cache: List[Dict[str, Any]] = []
_cache_loaded = False


def _load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _ensure_cache() -> None:
    global _updates_cache, _cache_loaded
    if _cache_loaded:
        return
    who = _load_json(WHO_FILE)
    india = _load_json(INDIA_FILE)
    for u in who + india:
        u.setdefault("source", "WHO")
        u.setdefault("region", "Global")
        u.setdefault("tags", [])
    _updates_cache = sorted(who + india, key=lambda x: x.get("date", ""), reverse=True)
    _cache_loaded = True


def get_health_updates(
    disease: str | None = None,
    region: str | None = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Get health updates, optionally filtered by disease tag or region."""
    _ensure_cache()
    out = _updates_cache
    if disease:
        d = disease.lower().replace("_", " ")
        out = [u for u in out if any(d in (t or "").lower() for t in u.get("tags", []))]
    if region:
        r = region.lower()
        out = [u for u in out if r in (u.get("region") or "").lower()]
    return out[:limit]


def get_updates_for_disease(disease_key: str) -> List[Dict[str, Any]]:
    """Get latest updates relevant to a disease (for inclusion in advisories)."""
    return get_health_updates(disease=disease_key, limit=3)


def reload_updates() -> None:
    """Force reload from files (call after updating JSON files)."""
    global _cache_loaded
    _cache_loaded = False
    _ensure_cache()
