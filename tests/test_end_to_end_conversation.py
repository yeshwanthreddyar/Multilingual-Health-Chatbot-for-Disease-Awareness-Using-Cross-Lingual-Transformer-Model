from __future__ import annotations

from app.nlp.pipeline import get_nlp_pipeline
from app.ml.pipeline import get_ml_pipeline
from app.dialog.manager import DialogueManager
from app.ml.intent_classifier import INTENT_CLASSES
from app.ml.disease_classifier import DISEASE_LABELS


def test_simple_english_flow_without_ollama() -> None:
    """
    End-to-end check: NLP → ML → Dialogue for a simple English symptom query.

    Uses template/KB-based responses only (no Ollama dependency).
    """
    nlp = get_nlp_pipeline()
    ml = get_ml_pipeline()
    dialog = DialogueManager()

    text = "I have fever and cough"
    nlp_out = nlp.process(text)
    ml_out = ml.run(text, nlp_out["tokens"], nlp_out["embedding"])

    action = dialog.next_action(
        session_id="pytest-session",
        intent=ml_out["intent"],
        symptoms=ml_out["symptoms"],
        top3_diseases=ml_out["top3_diseases"],
        is_emergency=ml_out["is_emergency"],
        lang=nlp_out["lang"],
        user_message=text,
    )

    response = dialog.build_response(
        session_id="pytest-session",
        action=action,
        intent=ml_out["intent"],
        symptoms=ml_out["symptoms"],
        top3_diseases=ml_out["top3_diseases"],
        is_emergency=ml_out["is_emergency"],
        lang=nlp_out["lang"],
        use_ollama_phrasing=False,
        user_message=text,
    )

    assert action in {"prevention", "general", "follow_up", "emergency", "location", "vaccination"}
    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_ml_pipeline_outputs_valid_labels_and_scores() -> None:
    """
    Unit-style check that the ML pipeline returns intents and disease labels
    that are consistent with the configured label sets and well-formed scores.
    """
    nlp = get_nlp_pipeline()
    ml = get_ml_pipeline()

    text = "I have fever, cough and body pain"
    nlp_out = nlp.process(text)
    ml_out = ml.run(text, nlp_out["tokens"], nlp_out["embedding"])

    # Intent should always be one of the known classes
    assert ml_out["intent"] in INTENT_CLASSES

    # Disease predictions should use known labels and bounded scores
    top3 = ml_out["top3_diseases"]
    assert isinstance(top3, list)
    assert len(top3) <= 3
    for disease_key, score in top3:
        assert disease_key in DISEASE_LABELS
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # Symptom extraction output should be a list of (code, phrase) tuples
    symptoms = ml_out["symptoms"]
    assert isinstance(symptoms, list)
    for code, phrase in symptoms:
        assert isinstance(code, str)
        assert isinstance(phrase, str)

