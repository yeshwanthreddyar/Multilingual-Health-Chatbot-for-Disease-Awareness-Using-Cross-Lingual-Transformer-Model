"""
Embedding Generation - IndicBERT (primary), mBERT (fallback).
Uses sentence-transformers. No text generation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

# Model names for sentence-transformers
INDICBERT_MODEL = "ai4bharat/indic-bert"
MBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Target embedding dimensionality used across training and runtime.
# All encoder outputs are padded or truncated to this size.
EMBEDDING_DIM = 384


def _ensure_dim(arr: np.ndarray, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    Ensure embeddings have a fixed dimensionality (pad/truncate to `dim`).

    This keeps the representation consistent across:
      - IndicBERT / multilingual MiniLM (which may emit 768-dim vectors)
      - Synthetic/random fallback embeddings during development
    """
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    d = arr.shape[1]
    if d == dim:
        return arr
    if d > dim:
        return arr[:, :dim]
    pad = np.zeros((arr.shape[0], dim - d), dtype=arr.dtype)
    return np.hstack([arr, pad])


class EmbeddingGenerator:
    """IndicBERT primary, mBERT fallback. No text generation. Fixed 384-dim output."""

    def __init__(self, use_indic_first: bool = True):
        self._indic_model = None
        self._mbert_model = None
        self._use_indic_first = use_indic_first
        self._primary_ready = False
        self._fallback_ready = False

    def _load_indic(self) -> bool:
        if self._indic_model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._indic_model = SentenceTransformer(INDICBERT_MODEL)
            self._primary_ready = True
            return True
        except Exception:
            self._primary_ready = False
            return False

    def _load_mbert(self) -> bool:
        if self._mbert_model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._mbert_model = SentenceTransformer(MBERT_MODEL)
            self._fallback_ready = True
            return True
        except Exception:
            self._fallback_ready = False
            return False

    def encode(
        self,
        texts: List[str],
        lang_hint: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode texts to embeddings. IndicBERT first, then mBERT fallback.
        No text generation. Always returns (n, EMBEDDING_DIM).
        """
        if not texts:
            return np.array([]).reshape(0, EMBEDDING_DIM)

        use_indic = self._use_indic_first and (lang_hint != "en" or not lang_hint)
        if use_indic and self._load_indic():
            try:
                raw = self._indic_model.encode(texts, convert_to_numpy=True)
                return _ensure_dim(np.asarray(raw))
            except Exception:
                pass
        if self._load_mbert():
            raw = self._mbert_model.encode(texts, convert_to_numpy=True)
            return _ensure_dim(np.asarray(raw))
        # Last resort: small random embedding for testing without models
        rand = np.random.randn(len(texts), EMBEDDING_DIM).astype(np.float32) * 0.01
        return rand

    def encode_single(self, text: str, lang_hint: Optional[str] = None) -> np.ndarray:
        """Encode a single string. Returns 1D array of size EMBEDDING_DIM."""
        arr = self.encode([text], lang_hint=lang_hint)
        return arr[0] if len(arr) else np.zeros(EMBEDDING_DIM, dtype=np.float32)


_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator(use_indic_first=True)
    return _embedding_generator
