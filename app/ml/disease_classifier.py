"""
Disease classification: P(disease | symptoms) ∝ P(symptoms | disease) × P(disease).
Naive Bayes + Random Forest + Gradient Boosting, ensemble weighted average.
Output: top-3 probable diseases, confidence scores, advisory wording only.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Placeholder disease labels (replace with real KB codes)
DISEASE_LABELS = [
    "common_cold",
    "flu",
    "gastroenteritis",
    "dengue",
    "malaria",
    "typhoid",
    "respiratory_infection",
    "hypertension",
    "diabetes_awareness",
    "skin_infection",
    "other",
]


class DiseaseClassifier:
    """
    Bayes: P(disease | symptoms) ∝ P(symptoms | disease) × P(disease).
    Ensemble: NB + RF + GB, weighted average.
    """

    def __init__(self, weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)):
        self._nb = GaussianNB()  # GaussianNB for binary/continuous features
        self._rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        self._le = LabelEncoder()
        self._le.fit(DISEASE_LABELS)
        self._weights = weights
        self._fitted = False

    def fit(self, X: np.ndarray, y: List[str]) -> None:
        """X: symptom features (e.g. multi-hot or embeddings), y: disease labels."""
        y_enc = self._le.transform(y)
        self._nb.fit(X, y_enc)
        self._rf.fit(X, y_enc)
        self._gb.fit(X, y_enc)
        self._fitted = True

    def predict_proba_ensemble(self, X: np.ndarray) -> np.ndarray:
        w1, w2, w3 = self._weights
        p_nb = self._nb.predict_proba(X)
        p_rf = self._rf.predict_proba(X)
        p_gb = self._gb.predict_proba(X)
        return w1 * p_nb + w2 * p_rf + w3 * p_gb

    def top_k(self, X: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Returns top-k (disease, confidence) for advisory only.
        """
        if not self._fitted or X.size == 0:
            return [("other", 0.0)] * min(k, 1)
        probs = self.predict_proba_ensemble(X)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        order = np.argsort(-probs[0])
        labels = self._le.inverse_transform(order[:k])
        scores = probs[0][order[:k]]
        return list(zip(labels.tolist(), scores.tolist()))
