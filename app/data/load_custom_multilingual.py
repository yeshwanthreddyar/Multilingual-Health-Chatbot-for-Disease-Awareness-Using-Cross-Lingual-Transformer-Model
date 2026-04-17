from __future__ import annotations

"""
Load a simple, user-created multilingual conversational dataset.

Expected CSV/JSON schema (one row per utterance):
- text: user message (string)
- language: ISO code like "en", "hi", "bn", "te" (string)
- intent: one of the 6 intents used in the project, e.g.:
    disease_information, symptom_reporting, prevention_guidance,
    vaccination_schedule, emergency_assessment, general_health_query
- symptoms: optional; list of symptom phrases or codes
- disease: optional; single disease key aligned with DISEASE_LABELS / DISEASE_GRAPH
- diseases: optional; list of disease keys
- answer: optional; assistant/doctor reply
- split: optional; "train" / "valid" / "test" (defaults to "train")
"""

from pathlib import Path
from typing import Dict, Any
import json

import pandas as pd


def _parse_maybe_list(value: Any) -> list:
    """
    Accepts common CSV encodings and returns a Python list:
    - empty / NaN -> []
    - list/tuple -> list(value)
    - JSON list string -> parsed list
    - semicolon-delimited string -> split
    - single string -> [string]
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
                # If dict, keep values as a list (rare case)
                if isinstance(parsed, dict):
                    return list(parsed.values())
            except Exception:
                pass
        if ";" in s:
            return [p.strip() for p in s.split(";") if p.strip()]
        return [s]
    # Fallback: best-effort cast
    try:
        return [str(value)]
    except Exception:
        return []


def _normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    text = str(row.get("text", "")).strip()
    lang = str(row.get("language", "en")).strip() or "en"
    intent = str(row.get("intent", "general_health_query")).strip() or "general_health_query"

    # Normalise optional fields
    symptoms = _parse_maybe_list(row.get("symptoms", []))

    disease = row.get("disease", "")
    if pd.isna(disease):
        disease = ""

    diseases = _parse_maybe_list(row.get("diseases", []))

    answer = row.get("answer", "")
    if pd.isna(answer):
        answer = ""

    split = str(row.get("split", "train")).strip() or "train"

    return {
        "text": text,
        "language": lang,
        "intent": intent,
        "symptoms": symptoms,
        "disease": disease,
        "diseases": diseases,
        "answer": answer,
        "source": "custom_multilingual",
        "split": split,
    }


def load_custom_multilingual(path: str = "data/custom_multilingual.csv") -> pd.DataFrame:
    """
    Load a custom multilingual dataset from CSV or JSON.

    - If `path` ends with .csv → uses pandas.read_csv.
    - If `path` ends with .json or .jsonl → uses pandas.read_json.
    """
    p = Path(path)
    print(f"Loading custom multilingual dataset from {p}...")
    if not p.exists():
        raise FileNotFoundError(f"Custom multilingual dataset not found at: {p}")

    if p.suffix.lower() == ".csv":
        raw_df = pd.read_csv(p)
    elif p.suffix.lower() in {".json", ".jsonl"}:
        raw_df = pd.read_json(p, lines=p.suffix.lower() == ".jsonl")
    else:
        raise ValueError(f"Unsupported custom dataset format: {p.suffix}")

    processed = [_normalise_row(r) for _, r in raw_df.iterrows()]
    df = pd.DataFrame(processed)
    print(f"Loaded {len(df)} samples from custom multilingual dataset")
    return df

