"""
Intent Recognition - 6 classes from PDF.
Disease Information, Symptom Reporting, Prevention Guidance,
Vaccination Schedule, Emergency Assessment, General Health Query.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Intent classes from PDF
INTENT_CLASSES = [
    "disease_information",
    "symptom_reporting",
    "prevention_guidance",
    "vaccination_schedule",
    "emergency_assessment",
    "general_health_query",
]


class Intent(str, Enum):
    DISEASE_INFORMATION = "disease_information"
    SYMPTOM_REPORTING = "symptom_reporting"
    PREVENTION_GUIDANCE = "prevention_guidance"
    VACCINATION_SCHEDULE = "vaccination_schedule"
    EMERGENCY_ASSESSMENT = "emergency_assessment"
    GENERAL_HEALTH_QUERY = "general_health_query"


class IntentClassifier:
    """Ensemble: Naive Bayes + Random Forest + Gradient Boosting, weighted average."""

    def __init__(self, weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)):
        self._nb = GaussianNB()  # Use GaussianNB for real-valued embeddings
        self._rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        self._le = LabelEncoder()
        self._le.fit(INTENT_CLASSES)
        self._weights = weights
        self._fitted = False

    def fit(self, X: np.ndarray, y: List[str]) -> None:
        """X: feature matrix (e.g. embeddings), y: intent labels."""
        y_enc = self._le.transform(y)
        self._nb.fit(X, y_enc)
        self._rf.fit(X, y_enc)
        self._gb.fit(X, y_enc)
        self._fitted = True

    def predict_proba_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of NB, RF, GB probabilities."""
        p_nb = self._nb.predict_proba(X)
        p_rf = self._rf.predict_proba(X)
        p_gb = self._gb.predict_proba(X)
        w1, w2, w3 = self._weights
        return w1 * p_nb + w2 * p_rf + w3 * p_gb

    def predict(self, X: np.ndarray) -> List[str]:
        if not self._fitted:
            return [INTENT_CLASSES[0]] * len(X)
        probs = self.predict_proba_ensemble(X)
        indices = np.argmax(probs, axis=1)
        return list(self._le.inverse_transform(indices))

    def predict_single(self, X: np.ndarray) -> Tuple[str, float]:
        """Returns (intent, confidence)."""
        if not self._fitted:
            return INTENT_CLASSES[0], 0.0
        probs = self.predict_proba_ensemble(X)
        idx = int(np.argmax(probs[0]))
        return self._le.inverse_transform([idx])[0], float(probs[0][idx])
