"""
Outbreak detector - detect rapid spread by region+disease.
Generates alerts when case counts exceed threshold or growth rate is high.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.outbreak.case_records import get_case_counts_by_region_disease

# Thresholds
OUTBREAK_THRESHOLD = 5  # cases in region+disease in last 7 days
OUTBREAK_DAYS = 7


def detect_outbreaks() -> List[Dict[str, Any]]:
    """
    Detect outbreaks: region+disease combinations exceeding threshold.
    Returns list of alert dicts compatible with government_mock alerts.
    """
    counts = get_case_counts_by_region_disease(days=OUTBREAK_DAYS)
    alerts = []
    for region, disease_counts in counts.items():
        for disease, count in disease_counts.items():
            if count >= OUTBREAK_THRESHOLD:
                alerts.append({
                    "id": f"OUT-{region}_{disease}".replace(" ", "_"),
                    "title": f"Rise in {disease.replace('_', ' ').title()} cases in {region}",
                    "region": region,
                    "source": "HealthBot Outbreak",
                    "severity": "high" if count >= 10 else "medium",
                    "summary": f"{count} cases reported in the last {OUTBREAK_DAYS} days. "
                    "Please take preventive measures and consult a healthcare provider if you have symptoms.",
                    "disease": disease,
                    "case_count": count,
                })
    return alerts


def merge_with_existing_alerts(existing_alerts: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    """Merge outbreak alerts with existing government alerts. Outbreak alerts first."""
    outbreaks = detect_outbreaks()
    seen_ids = {a.get("id") for a in outbreaks}
    for a in existing_alerts:
        if a.get("id") not in seen_ids:
            outbreaks.append(a)
            seen_ids.add(a.get("id"))
    return outbreaks[:limit]
