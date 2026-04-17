"""
Knowledge Base Retriever - Retrieval ONLY.
Disease info, symptom-disease mapping, prevention & vaccination guidelines.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.knowledge_base.graph import (
    get_disease_info,
    get_diseases_by_symptoms,
    get_prevention,
    get_vaccination_guidelines,
    get_disease_name,
)


def retrieve_disease_advisory(disease_key: str, lang: str = "en") -> Dict:
    """Retrieve advisory content for a disease (no diagnosis). Includes WHO/India updates."""
    info = get_disease_info(disease_key)
    if not info:
        return {
            "name": "Other",
            "prevention": ["Consult a healthcare provider."],
            "vaccines": [],
            "updates": [],
        }
    try:
        from app.data.load_health_updates import get_updates_for_disease

        updates = get_updates_for_disease(disease_key)
    except Exception:
        updates = []
    return {
        "name": get_disease_name(disease_key, lang=lang),
        "prevention": get_prevention(disease_key),
        "vaccines": get_vaccination_guidelines(disease_key),
        "updates": updates,
    }


def retrieve_for_symptoms(symptom_codes: List[str], top_k: int = 3, lang: str = "en") -> List[Dict]:
    """Retrieve top-k disease advisories by symptom codes."""
    ranked = get_diseases_by_symptoms(symptom_codes)[:top_k]
    return [retrieve_disease_advisory(d, lang=lang) for d, _ in ranked]
