"""
Train intent classifier from health-related chat logs.
Run: python train_from_logs.py
Uses logs from DB where is_health_related=True, merges with existing dataset, retrains.
"""
from __future__ import annotations

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from app.db import SessionLocal, init_db
from app.models.log import Log
from app.ml.intent_classifier import IntentClassifier
from app.ml.intent_classifier import INTENT_CLASSES
from app.nlp.embeddings import get_embedding_generator

INTENT_MAP = {
    "disease_information": "disease_information",
    "symptom_reporting": "symptom_reporting",
    "prevention_guidance": "prevention_guidance",
    "vaccination_schedule": "vaccination_schedule",
    "emergency_assessment": "emergency_assessment",
    "general_health_query": "general_health_query",
}


def load_logs() -> pd.DataFrame:
    init_db()
    db = SessionLocal()
    rows = db.query(Log).filter(Log.is_health_related == True).all()
    db.close()
    records = []
    for r in rows:
        if not r.message or len(str(r.message).strip()) < 3:
            continue
        intent = r.intent or "general_health_query"
        if intent not in INTENT_MAP:
            intent = "general_health_query"
        records.append({"text": r.message, "intent": INTENT_MAP[intent], "language": "en"})
    return pd.DataFrame(records)


def load_existing() -> pd.DataFrame:
    path = Path("data/processed/combined_dataset.json")
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_json(path)
        if "intent" not in df.columns:
            return pd.DataFrame()
        df["intent"] = df["intent"].apply(
            lambda x: INTENT_MAP.get(x, "general_health_query") if x in INTENT_MAP else "general_health_query"
        )
        return df[["text", "intent", "language"]].dropna(subset=["text"])
    except Exception:
        return pd.DataFrame()


def main() -> None:
    print("Loading health-related logs...")
    logs_df = load_logs()
    print(f"  Logs: {len(logs_df)} health-related samples")

    print("Loading existing dataset...")
    existing_df = load_existing()
    print(f"  Existing: {len(existing_df)} samples")

    combined = pd.concat([existing_df, logs_df], ignore_index=True).drop_duplicates(subset=["text"])
    print(f"  Combined: {len(combined)} samples")

    if len(combined) < 10:
        print("Not enough samples to train. Need at least 10.")
        return

    print("Generating embeddings...")
    emb = get_embedding_generator()
    X, y = [], []
    for _, row in combined.iterrows():
        text = str(row["text"]).strip()
        if not text or text.isdigit() or len(text) < 3:
            continue
        intent = row["intent"]
        if intent not in INTENT_CLASSES:
            intent = "general_health_query"
        try:
            vec = emb.encode_single(text, lang_hint=row.get("language"))
            if len(vec) > 0:
                X.append(vec)
                y.append(intent)
        except Exception:
            continue

    if len(X) < 10:
        print("Not enough valid embeddings.")
        return

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training intent classifier...")
    clf = IntentClassifier()
    clf.fit(X_train, y_train.tolist())

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"  Accuracy: {acc:.2%}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, "models/intent_classifier_trained_from_logs.pkl")
    print("  Saved to models/intent_classifier_trained_from_logs.pkl")


if __name__ == "__main__":
    main()
