from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_logistic_baseline() -> Pipeline:
    """Return a sklearn Pipeline: CountVectorizer -> Tfidf -> LogisticRegression"""
    pipe = Pipeline(
        [
            ("vect", CountVectorizer(ngram_range=(1, 2), max_features=20000)),
            ("tfidf", TfidfTransformer()),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )
    return pipe


def train_logistic_baseline(texts: Iterable[str], labels: Iterable[str]) -> Pipeline:
    """Train and return a logistic regression baseline pipeline."""
    pipe = build_logistic_baseline()
    pipe.fit(list(texts), list(labels))
    return pipe


def predict(pipe: Pipeline, texts: Iterable[str]) -> np.ndarray:
    return pipe.predict(list(texts))


def predict_proba(pipe: Pipeline, texts: Iterable[str]) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(list(texts))
    # fallback: return zeros
    return np.zeros((len(list(texts)), 1))
