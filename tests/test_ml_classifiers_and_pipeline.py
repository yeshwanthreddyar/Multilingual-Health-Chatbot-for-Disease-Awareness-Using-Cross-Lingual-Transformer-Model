from __future__ import annotations

from typing import List

import numpy as np

from app.ml.intent_classifier import INTENT_CLASSES, IntentClassifier
from app.ml.disease_classifier import DISEASE_LABELS, DiseaseClassifier
from app.ml.pipeline import get_ml_pipeline
from app.nlp.pipeline import get_nlp_pipeline


def _cycle_labels(labels: List[str], n: int) -> List[str]:
    """Utility: repeat labels to length n."""
    out: List[str] = []
    for i in range(n):
        out.append(labels[i % len(labels)])
    return out


def test_intent_classifier_basic_fit_and_predict() -> None:
    """
    Unit-level check: IntentClassifier can be fitted on synthetic embeddings
    and returns a valid intent label with a probability in [0, 1].
    """
    dim = 32
    n = 60
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = _cycle_labels(INTENT_CLASSES, n)

    clf = IntentClassifier()
    clf.fit(X, y)

    x_test = rng.normal(size=(1, dim)).astype(np.float32)
    intent, conf = clf.predict_single(x_test)

    assert intent in INTENT_CLASSES
    assert 0.0 <= conf <= 1.0


def test_disease_classifier_basic_fit_and_topk() -> None:
    """
    Unit-level check: DiseaseClassifier can be fitted on synthetic symptom
    vectors and returns a top-k list with valid disease labels.
    """
    dim = 10
    n = 80
    rng = np.random.default_rng(123)
    X = (rng.random(size=(n, dim)) > 0.7).astype(np.float32)
    y = _cycle_labels(DISEASE_LABELS, n)

    clf = DiseaseClassifier()
    clf.fit(X, y)

    x_test = (rng.random(size=(1, dim)) > 0.5).astype(np.float32)
    top3 = clf.top_k(x_test, k=3)

    assert len(top3) == 3
    for disease, score in top3:
        assert disease in DISEASE_LABELS
        assert 0.0 <= score <= 1.0


def test_ml_pipeline_run_structure() -> None:
    """
    Integration-level check: NLP → ML pipeline for a simple sentence
    produces structured outputs consistent with the research claims.
    """
    nlp = get_nlp_pipeline()
    ml = get_ml_pipeline()

    text = "I have fever and cough"
    nlp_out = nlp.process(text)
    ml_out = ml.run(text, nlp_out["tokens"], nlp_out["embedding"])

    # Intent / confidence
    assert ml_out["intent"] in INTENT_CLASSES
    assert isinstance(ml_out["intent_confidence"], float)

    # Symptoms and diseases
    assert isinstance(ml_out["symptoms"], list)
    assert isinstance(ml_out["top3_diseases"], list)
    assert len(ml_out["top3_diseases"]) == 3
    for disease, score in ml_out["top3_diseases"]:
        # In debug mode the classifier may fall back to "other"
        assert disease in DISEASE_LABELS or disease == "other"
        assert 0.0 <= float(score) <= 1.0

    # Emergency flag and advisory-only behaviour
    assert isinstance(ml_out["is_emergency"], bool)
    assert ml_out["advisory_only"] is True

