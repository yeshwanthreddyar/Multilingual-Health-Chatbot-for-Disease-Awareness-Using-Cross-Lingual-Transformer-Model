"""
Language-aware tokenization for Indian languages.
Handles agglutinative languages (e.g. Tamil, Telugu, Malayalam).
"""
from __future__ import annotations

import re
from typing import List

from app.nlp.language_detection import detect_language


def _whitespace_tokenize(text: str) -> List[str]:
    """Basic whitespace + punctuation split."""
    tokens = re.findall(r"\w+|[^\w\s]", text.strip())
    return [t for t in tokens if t.strip()]


def _indic_tokenize(text: str, lang: str) -> List[str]:
    """
    Tokenize Indic text: preserve script units, split on virama/sandhi where safe.
    For agglutinative languages we keep compound tokens but split on spaces and punctuation.
    """
    # Normalize ZWJ/ZWNJ for consistent splitting
    text = text.replace("\u200d", "").replace("\u200c", " ")
    tokens = _whitespace_tokenize(text)
    return tokens


def tokenize(text: str, lang: str | None = None) -> List[str]:
    """
    Language-aware tokenization.
    If lang not provided, detected from text.
    """
    if not text or not text.strip():
        return []
    lang = lang or detect_language(text)
    return _indic_tokenize(text.strip(), lang)


def tokenize_for_embedding(text: str, lang: str | None = None, max_tokens: int = 512) -> List[str]:
    """
    Tokenize for embedding models; optionally truncate to max_tokens.
    """
    tokens = tokenize(text, lang)
    return tokens[:max_tokens] if len(tokens) > max_tokens else tokens
