"""
ML Pipeline: intent recognition + symptom extraction + disease classification.
Uses embeddings from NLP layer; ensemble NB + RF + GB.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ml.intent_classifier import IntentClassifier, INTENT_CLASSES
from app.ml.symptom_extractor import extract_symptoms, is_emergency_symptom
from app.ml.disease_classifier import DiseaseClassifier, DISEASE_LABELS
from app.nlp.embeddings import EMBEDDING_DIM


def _symptom_vector(symptom_codes: List[str], all_codes: List[str]) -> np.ndarray:
    """Multi-hot from symptom codes for disease classifier."""
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    vec = np.zeros(len(all_codes), dtype=np.float32)
    for c in symptom_codes:
        if c in code_to_idx:
            vec[code_to_idx[c]] = 1.0
    return vec


# Canonical symptom codes used in disease model
CANONICAL_SYMPTOM_CODES = [
    "R50", "R51", "R05", "J00", "R52", "R10", "R19", "R11", "R53",
    "R06", "R07", "R21", "L29", "R09.8", "R63", "R42", "M25.5", "M54", "R22", "R20", "H10", "H92", "K08.8",
]


class MLPipeline:
    """Intent + symptom extraction + disease top-3 (advisory only)."""

    def __init__(self):
        self._intent_clf = IntentClassifier()
        self._disease_clf = DiseaseClassifier()
        self._intent_fitted = False
        self._disease_fitted = False

    def fit_intent(self, X: np.ndarray, y: List[str]) -> None:
        self._intent_clf.fit(X, y)
        self._intent_fitted = True

    def fit_disease(self, X: np.ndarray, y: List[str]) -> None:
        self._disease_clf.fit(X, y)
        self._disease_fitted = True

    def run(
        self,
        text: str,
        tokens: List[str],
        embedding: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Returns:
          intent, intent_confidence, symptoms (list of (code, phrase)),
          top3_diseases [(name, score)], is_emergency, advisory_only.
        """
        # Intent from embedding (reshape to 2D; pad/truncate to EMBEDDING_DIM for fitted classifier)
        emb_2d = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding
        dim_expected = EMBEDDING_DIM
        if emb_2d.shape[1] != dim_expected:
            if emb_2d.shape[1] > dim_expected:
                emb_2d = emb_2d[:, :dim_expected]
            else:
                pad = np.zeros((1, dim_expected - emb_2d.shape[1]), dtype=emb_2d.dtype)
                emb_2d = np.hstack([emb_2d, pad])
        intent, intent_conf = self._intent_clf.predict_single(emb_2d)

        # Symptom extraction
        symptoms = extract_symptoms(text, tokens)
        symptom_codes = [s[0] for s in symptoms]
        emergency = is_emergency_symptom(symptom_codes)

        # --- Intent override rules ---
        # Rule 1: If symptoms found but wrong intent, force symptom_reporting
        SYMPTOM_INTENTS = {"symptom_reporting", "emergency_assessment", "disease_information"}
        if symptom_codes and intent not in SYMPTOM_INTENTS:
            intent = "symptom_reporting"
            intent_conf = max(intent_conf, 0.6)

        # Rule 2: Non-English scripts are almost always symptom reports or health queries.
        # The ML classifier is trained mostly on English so it misclassifies Indian scripts
        # as prevention_guidance. Override it for all Indian language scripts.
        NON_ENGLISH_LANGS = {"hi","bn","te","ta","kn","ml","mr","gu","pa","or","as","ur","ne","kok"}
        from app.nlp.language_detection import detect_language
        detected_lang = detect_language(text)
        if detected_lang in NON_ENGLISH_LANGS and intent == "prevention_guidance":
            intent = "symptom_reporting" if symptom_codes else "general_health_query"
            intent_conf = max(intent_conf, 0.55)

        # Disease top-3:
        # - Some trained disease classifiers are fit on 384-dim text embeddings (train_disease_classifier.py / train.py multilingual).
        # - The online symptom extractor produces a 23-dim multi-hot symptom vector.
        # Choose the feature representation that matches the loaded model.
        vec_sym = (
            _symptom_vector(symptom_codes, CANONICAL_SYMPTOM_CODES).reshape(1, -1)
            if symptom_codes
            else np.zeros((1, len(CANONICAL_SYMPTOM_CODES)), dtype=np.float32)
        )
        dis_features = vec_sym
        try:
            expected = getattr(getattr(self._disease_clf, "_nb", None), "n_features_in_", None)
            if expected == EMBEDDING_DIM:
                dis_features = emb_2d
        except Exception:
            pass
        top3 = self._disease_clf.top_k(dis_features, k=3)

        return {
            "intent": intent,
            "intent_confidence": intent_conf,
            "symptoms": symptoms,
            "top3_diseases": top3,
            "is_emergency": emergency,
            "advisory_only": True,
        }


_ml_pipeline: Optional[MLPipeline] = None


def get_ml_pipeline() -> MLPipeline:
    global _ml_pipeline
    if _ml_pipeline is None:
        _ml_pipeline = MLPipeline()
        # Load trained intent classifier if available (from train.py or train_from_logs.py)
        try:
            import joblib
            from pathlib import Path

            for p in ["models/intent_classifier_trained_from_logs.pkl", "models/intent_classifier_trained.pkl"]:
                if Path(p).exists():
                    _ml_pipeline._intent_clf = joblib.load(p)
                    _ml_pipeline._intent_fitted = True
                    break
        except Exception:
            pass

        # Load trained disease classifier if available (from train_disease_classifier.py)
        try:
            import joblib
            from pathlib import Path

            for p in [
                "models/disease_classifier_trained_multilingual.pkl",
                "models/disease_classifier_trained.pkl",
            ]:
                if Path(p).exists():
                    _ml_pipeline._disease_clf = joblib.load(p)
                    _ml_pipeline._disease_fitted = True
                    break
        except Exception:
            pass

        # Fallback: minimal synthetic fit only if nothing real is available
        if not _ml_pipeline._intent_fitted and not _ml_pipeline._disease_fitted:
            _fit_synthetic(_ml_pipeline)
    return _ml_pipeline


def _fit_synthetic(pipeline: MLPipeline) -> None:
    """Minimal synthetic fit so classifiers don't fail."""
    np.random.seed(42)
    n = 100
    X_emb = np.random.randn(n, EMBEDDING_DIM).astype(np.float32) * 0.1
    y_intent = [INTENT_CLASSES[i % len(INTENT_CLASSES)] for i in range(n)]
    pipeline.fit_intent(X_emb, y_intent)

    n_d = 80
    dim_sym = len(CANONICAL_SYMPTOM_CODES)
    X_sym = (np.random.rand(n_d, dim_sym) > 0.7).astype(np.float32)
    y_disease = [DISEASE_LABELS[i % len(DISEASE_LABELS)] for i in range(n_d)]
    pipeline.fit_disease(X_sym, y_disease)