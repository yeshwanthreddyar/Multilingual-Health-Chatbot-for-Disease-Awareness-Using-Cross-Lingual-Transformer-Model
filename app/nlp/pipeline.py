"""
NLP Processing Pipeline: language detection → tokenization → embedding generation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.nlp.language_detection import detect_language
from app.nlp.tokenization import tokenize
from app.nlp.embeddings import get_embedding_generator


class NLPPipeline:
    """Single entry for NLP: detect lang → tokenize → embed."""

    def __init__(self):
        self._embedder = get_embedding_generator()

    def process(self, text: str) -> dict:
        """
        Returns: {
            "lang": ISO code,
            "tokens": list of tokens,
            "embedding": 1D numpy array,
        }
        """
        lang = detect_language(text)
        tokens = tokenize(text, lang)
        embedding = self._embedder.encode_single(text, lang_hint=lang)
        return {
            "lang": lang,
            "tokens": tokens,
            "embedding": embedding,
        }

    def process_batch(self, texts: List[str], lang_hint: Optional[str] = None) -> List[dict]:
        """Process multiple texts; lang_hint optional for batch."""
        if not texts:
            return []
        lang = lang_hint or (detect_language(texts[0]) if texts else "en")
        tokens_list = [tokenize(t, lang) for t in texts]
        embeddings = self._embedder.encode(texts, lang_hint=lang_hint)
        return [
            {"lang": lang, "tokens": tok, "embedding": emb}
            for tok, emb in zip(tokens_list, embeddings)
        ]


_pipeline: Optional[NLPPipeline] = None


def get_nlp_pipeline() -> NLPPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = NLPPipeline()
    return _pipeline
