"""
Case records - people affected by disease (by region).
Stored in data/cases.json. Can be migrated to DB later.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CASES_FILE = Path(__file__).resolve().parents[2] / "data" / "cases.json"


def _ensure_file() -> None:
    CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CASES_FILE.exists():
        CASES_FILE.write_text("[]", encoding="utf-8")


def _load_cases() -> List[Dict[str, Any]]:
    _ensure_file()
    try:
        data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_cases(cases: List[Dict[str, Any]]) -> None:
    _ensure_file()
    CASES_FILE.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")


def add_case(
    region: str,
    disease: str,
    source: str = "self_report",
    user_id: Optional[str] = None,
    severity: str = "unknown",
) -> Dict[str, Any]:
    """Record a disease case. Returns the created case."""
    cases = _load_cases()
    case = {
        "id": f"CASE-{len(cases) + 1:06d}",
        "region": region.strip(),
        "disease": disease.strip(),
        "source": source,
        "user_id": user_id,
        "severity": severity,
        "date_reported": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    cases.append(case)
    _save_cases(cases)
    return case


def get_cases(
    region: Optional[str] = None,
    disease: Optional[str] = None,
    days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Get cases, optionally filtered by region, disease, or last N days."""
    cases = _load_cases()
    if region:
        r = region.strip().lower()
        cases = [c for c in cases if r in (c.get("region") or "").lower()]
    if disease:
        d = disease.strip().lower()
        cases = [c for c in cases if d in (c.get("disease") or "").lower()]
    if days:
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        cases = [c for c in cases if (c.get("date_reported") or "")[:10] >= cutoff]
    return cases


def get_case_counts_by_region_disease(days: int = 7) -> Dict[str, Dict[str, int]]:
    """Returns {region: {disease: count}} for last N days."""
    cases = get_cases(days=days)
    counts: Dict[str, Dict[str, int]] = {}
    for c in cases:
        reg = c.get("region") or "Unknown"
        dis = c.get("disease") or "unknown"
        if reg not in counts:
            counts[reg] = {}
        counts[reg][dis] = counts[reg].get(dis, 0) + 1
    return counts
