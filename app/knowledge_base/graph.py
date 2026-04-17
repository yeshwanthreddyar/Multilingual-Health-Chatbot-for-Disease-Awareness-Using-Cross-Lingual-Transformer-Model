"""
Medical Knowledge Base - Retrieval ONLY.
Disease-centric knowledge graph (in-memory), symptom-disease mapping,
prevention & vaccination guidelines, multilingual disease names.
Aligned with WHO / MoHFW / ICMR concepts.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# In-memory disease knowledge graph.
# Node type: "disease"; edges expressed as symptom codes + prevention/vaccine lists.
DISEASE_GRAPH: Dict[str, Dict] = {
    "common_cold": {
        "type": "disease",
        "name_en": "Common Cold",
        "names": {
            "en": "Common Cold",
            "hi": "साधारण जुकाम",
            "bn": "সাধারণ সর্দি",
            "te": "సాధారణ జలుబు",
        },
        "symptoms": ["R05", "R09.8", "R07.0", "J00"],
        "prevention": ["Hand hygiene", "Avoid close contact with sick", "Rest and fluids"],
        "vaccines": [],
        "who_code": "J00",
    },
    "flu": {
        "type": "disease",
        "name_en": "Influenza",
        "names": {
            "en": "Influenza (Flu)",
            "hi": "फ़्लू (इन्फ्लुएंज़ा)",
            "bn": "ইনফ্লুয়েঞ্জা (ফ্লু)",
            "te": "ఇన్ఫ్లుయెంజా (ఫ్లూ)",
        },
        "symptoms": ["R50", "R51", "R05", "R53", "R52"],
        "prevention": ["Annual flu vaccine", "Hand washing", "Cover cough/sneeze"],
        "vaccines": ["Influenza vaccine"],
        "who_code": "J11",
    },
    "gastroenteritis": {
        "type": "disease",
        "name_en": "Gastroenteritis",
        "names": {
            "en": "Gastroenteritis",
            "hi": "गैस्ट्रोएन्टेराइटिस",
            "bn": "গ্যাস্ট্রোএন্টেরাইটিস",
            "te": "గ్యాస్ట్రోఎంటెరైటిస్",
        },
        "symptoms": ["R19", "R11", "R10", "R53"],
        "prevention": ["Safe water", "Hand hygiene", "Safe food handling"],
        "vaccines": ["Rotavirus (children)"],
        "who_code": "A09",
    },
    "dengue": {
        "type": "disease",
        "name_en": "Dengue",
        "names": {
            "en": "Dengue",
            "hi": "डेंगू",
            "bn": "ডেঙ্গু",
            "te": "డెంగ్యూ",
        },
        "symptoms": ["R50", "R21", "R52", "R53", "R11"],
        "prevention": ["Mosquito control", "Use repellent", "Cover water containers"],
        "vaccines": ["Dengvaxia (in some programs)"],
        "who_code": "A90",
    },
    "malaria": {
        "type": "disease",
        "name_en": "Malaria",
        "names": {
            "en": "Malaria",
            "hi": "मलेरिया",
            "bn": "ম্যালেরিয়া",
            "te": "మలేరియా",
        },
        "symptoms": ["R50", "R53", "R11", "R52"],
        "prevention": ["Mosquito nets", "Indoor spraying", "Early treatment"],
        "vaccines": ["RTS,S (pilot)"],
        "who_code": "B54",
    },
    "typhoid": {
        "type": "disease",
        "name_en": "Typhoid",
        "names": {
            "en": "Typhoid",
            "hi": "टाइफॉइड",
            "bn": "টাইফয়েড",
            "te": "టైఫాయిడ్",
        },
        "symptoms": ["R50", "R10", "R19", "R53"],
        "prevention": ["Safe water and food", "Hand washing", "Vaccination"],
        "vaccines": ["Typhoid vaccine"],
        "who_code": "A01.0",
    },
    "respiratory_infection": {
        "type": "disease",
        "name_en": "Respiratory Infection",
        "names": {
            "en": "Respiratory Infection",
            "hi": "श्वसन संक्रमण",
            "bn": "শ্বাসযন্ত্রের সংক্রমণ",
            "te": "శ్వాసకోశ సంక్రమణ",
        },
        "symptoms": ["R05", "R06", "R50", "R07"],
        "prevention": ["Vaccination (pneumococcal, flu)", "Avoid smoke", "Ventilation"],
        "vaccines": ["PCV", "Influenza"],
        "who_code": "J22",
    },
    "hypertension": {
        "type": "disease",
        "name_en": "Hypertension",
        "names": {
            "en": "Hypertension",
            "hi": "हाई ब्लड प्रेशर",
            "bn": "উচ্চ রক্তচাপ",
            "te": "అధిక రక్తపోటు",
        },
        "symptoms": ["R51", "R42", "R07"],
        "prevention": ["Low salt", "Exercise", "No tobacco", "Regular check-up"],
        "vaccines": [],
        "who_code": "I10",
    },
    "diabetes_awareness": {
        "type": "disease",
        "name_en": "Diabetes (awareness)",
        "names": {
            "en": "Diabetes (awareness)",
            "hi": "मधुमेह (जागरूकता)",
            "bn": "ডায়াবেটিস (সচেতনতা)",
            "te": "మధుమేహం (జాగ్రత్త)",
        },
        "symptoms": ["R53", "R63", "R35"],
        "prevention": ["Healthy diet", "Physical activity", "Screening"],
        "vaccines": [],
        "who_code": "E14",
    },
    "skin_infection": {
        "type": "disease",
        "name_en": "Skin Infection",
        "names": {
            "en": "Skin Infection",
            "hi": "त्वचा संक्रमण",
            "bn": "ত্বকের সংক্রমণ",
            "te": "చర్మ సంక్రమణ",
        },
        "symptoms": ["R21", "L29", "R22"],
        "prevention": ["Clean wounds", "Avoid sharing personal items", "Hygiene"],
        "vaccines": [],
        "who_code": "L08",
    },
    "other": {
        "type": "disease",
        "name_en": "Other",
        "names": {
            "en": "Other",
        },
        "symptoms": [],
        "prevention": ["Consult a healthcare provider"],
        "vaccines": [],
        "who_code": "",
    },
}

# Symptom code → list of disease keys (for retrieval)
SYMPTOM_TO_DISEASE: Dict[str, List[str]] = {}
for d, info in DISEASE_GRAPH.items():
    for s in info["symptoms"]:
        SYMPTOM_TO_DISEASE.setdefault(s, []).append(d)


def get_disease_info(disease_key: str) -> Optional[Dict]:
    """Retrieve disease info by key (advisory only)."""
    return DISEASE_GRAPH.get(disease_key)


def get_disease_name(disease_key: str, lang: str = "en") -> str:
    """
    Retrieve localized disease display name.

    Falls back to English name and then the raw key if no localized name exists.
    """
    info = DISEASE_GRAPH.get(disease_key) or {}
    names = info.get("names") or {}
    # Exact language match
    if lang in names:
        return names[lang]
    # Fallback for regional variants (e.g. 'hi-IN' → 'hi')
    base = lang.split("-")[0]
    if base in names:
        return names[base]
    return info.get("name_en") or disease_key


def get_diseases_by_symptoms(symptom_codes: List[str]) -> List[Tuple[str, float]]:
    """Retrieve diseases associated with symptoms (for ranking)."""
    counts: Dict[str, int] = {}
    for code in symptom_codes:
        for d in SYMPTOM_TO_DISEASE.get(code, []):
            counts[d] = counts.get(d, 0) + 1
    total = sum(counts.values()) or 1
    return [(d, c / total) for d, c in sorted(counts.items(), key=lambda x: -x[1])]


def get_prevention(disease_key: str) -> List[str]:
    """Prevention guidelines for disease."""
    info = DISEASE_GRAPH.get(disease_key)
    return (info or {}).get("prevention", [])


def get_vaccination_guidelines(disease_key: str) -> List[str]:
    """Vaccination guidelines for disease."""
    info = DISEASE_GRAPH.get(disease_key)
    return (info or {}).get("vaccines", [])


def get_rich_context_for_topk(
    topk: List[Tuple[str, float]],
) -> List[Dict[str, object]]:
    """
    Enrich top-k disease predictions from the ML layer with KB content.

    Input: [(disease_key, score), ...]
    Output: list of dicts with:
      - key, score, name_en, who_code, symptoms, prevention, vaccines
      - names (multilingual map, if available)
    """
    enriched: List[Dict[str, object]] = []
    for key, score in topk:
        info = DISEASE_GRAPH.get(key) or {}
        enriched.append(
            {
                "key": key,
                "score": float(score),
                "name_en": info.get("name_en", key),
                "who_code": info.get("who_code", ""),
                "symptoms": list(info.get("symptoms", [])),
                "prevention": list(info.get("prevention", [])),
                "vaccines": list(info.get("vaccines", [])),
                "names": dict(info.get("names", {})),
            }
        )
    return enriched


def get_localised_rich_context_for_topk(
    topk: List[Tuple[str, float]],
    lang: str = "en",
) -> List[Dict[str, object]]:
    """
    Enrich top-k disease predictions from the ML layer with KB content and
    include a language-specific display name.

    Input: [(disease_key, score), ...]
    Output: list of dicts with:
      - key, score, display_name, who_code, symptoms, prevention, vaccines, names
    """
    enriched_base = get_rich_context_for_topk(topk)
    for item in enriched_base:
        key = item.get("key", "")
        item["display_name"] = get_disease_name(key, lang=lang)
    return enriched_base
