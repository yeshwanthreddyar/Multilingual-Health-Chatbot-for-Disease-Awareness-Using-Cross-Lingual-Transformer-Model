# app/data/__init__.py
"""Data loading modules and helpers for HealthBot training datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import pandas as pd


class CanonicalSample(TypedDict, total=False):
    """
    Canonical schema for a single conversational sample used across loaders.

    This matches the schema documented in DATASET.md and is the expected
    representation for the combined dataset.
    """

    text: str
    intent: str
    language: str
    split: str
    symptoms: List[str]
    disease: str
    diseases: List[str]
    is_emergency: bool
    answer: str
    context: str
    source: str


CANONICAL_COLUMNS: List[str] = [
    "text",
    "intent",
    "language",
    "split",
    "symptoms",
    "disease",
    "diseases",
    "is_emergency",
    "answer",
    "context",
    "source",
]


def load_combined_conversations(path: str = "data/processed/combined_dataset.json") -> pd.DataFrame:
    """
    Load the combined multilingual conversational dataset produced by train.py.

    Returns an empty DataFrame if the file is missing or invalid.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_json(p)
    except Exception:
        return pd.DataFrame()

    # Basic sanity check: ensure we have a text column
    if "text" not in df.columns:
        return pd.DataFrame()

    return as_canonical_dataframe(df)


def as_canonical_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project a DataFrame to the canonical schema.

    - Ensures all canonical columns exist (filled with sensible defaults).
    - Preserves any extra columns present in the input.
    """
    if df.empty:
        return df

    df = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            # Fill with appropriate defaults
            if col in {"symptoms", "diseases"}:
                df[col] = [[] for _ in range(len(df))]
            elif col == "is_emergency":
                df[col] = False
            elif col in {"intent", "language", "split", "disease", "answer", "context", "source"}:
                df[col] = ""

    # Reorder canonical columns first, keep any additional columns afterwards
    other_cols = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    ordered_cols = CANONICAL_COLUMNS + other_cols
    return df[ordered_cols]


def summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """
    Compute basic dataset properties for documentation and monitoring.

    Returns a dict with:
    - total_samples
    - languages: {lang: count}
    - intents: {intent: count}
    - by_language_and_intent: nested counts
    """
    if df.empty:
        return {
            "total_samples": 0,
            "languages": {},
            "intents": {},
            "by_language_and_intent": {},
        }

    df = as_canonical_dataframe(df)

    total_samples = int(len(df))
    languages = (
        df["language"].value_counts(dropna=False).to_dict()
        if "language" in df.columns
        else {}
    )
    intents = (
        df["intent"].value_counts(dropna=False).to_dict()
        if "intent" in df.columns
        else {}
    )

    by_lang_intent: Dict[str, Dict[str, int]] = {}
    if "language" in df.columns and "intent" in df.columns:
        for (lang, intent), count in (
            df.groupby(["language", "intent"]).size().items()
        ):
            by_lang_intent.setdefault(str(lang), {})[str(intent)] = int(count)

    return {
        "total_samples": total_samples,
        "languages": {str(k): int(v) for k, v in languages.items()},
        "intents": {str(k): int(v) for k, v in intents.items()},
        "by_language_and_intent": by_lang_intent,
    }

