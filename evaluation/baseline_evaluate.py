"""Train and evaluate simple baselines on Symptom2Disease.csv or expanded CSV.

Produces a classification report for a logistic-regression baseline and a simple
keyword-based rule baseline.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path as _Path

# Ensure repo root is on path so `app` package can be imported when running as script
ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.baseline import train_logistic_baseline, predict


def load_csv(path: Path) -> (List[str], List[str]):
    texts = []
    labels = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            # support formats: (label, text) or (idx, label, text)
            if len(row) == 2:
                label = row[0]
                text = row[1]
            else:
                label = row[1]
                text = row[2]
            labels.append(label)
            texts.append(text)
    return texts, labels


def build_keyword_baseline(texts: List[str], labels: List[str], top_k: int = 20):
    # For each label, count top tokens
    token_counts = defaultdict(Counter)
    for t, l in zip(texts, labels):
        tokens = [w.lower().strip(".,") for w in t.split() if len(w) > 2]
        token_counts[l].update(tokens)
    keywords = {l: [w for w, _ in token_counts[l].most_common(top_k)] for l in token_counts}
    return keywords


def predict_keyword(keywords, text: str) -> str:
    tokens = [w.lower().strip(".,") for w in text.split() if len(w) > 2]
    scores = Counter()
    for t in tokens:
        for label, kws in keywords.items():
            if t in kws:
                scores[label] += 1
    if not scores:
        # fallback: choose most common label
        return max(keywords.keys(), key=lambda k: len(keywords[k]))
    return scores.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/expanded_symptoms.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        # fall back to root CSV
        fallback = Path("Symptom2Disease.csv")
        if fallback.exists():
            args.input = fallback
        else:
            raise SystemExit("No dataset found. Run scripts/augment_dataset.py first or provide path.")

    texts, labels = load_csv(args.input)
    # stratify when feasible; fall back when there are too many classes with few examples
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=args.test_size, random_state=args.random_state, stratify=labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=args.test_size, random_state=args.random_state
        )

    # train logistic baseline
    lr = train_logistic_baseline(X_train, y_train)
    y_pred_lr = predict(lr, X_test)

    # keyword baseline
    keywords = build_keyword_baseline(X_train, y_train)
    y_pred_kw = [predict_keyword(keywords, t) for t in X_test]

    # report
    print("=== Logistic Regression Baseline ===")
    print(classification_report(y_test, y_pred_lr, zero_division=0))
    print("=== Keyword Baseline ===")
    print(classification_report(y_test, y_pred_kw, zero_division=0))

    # save confusion matrix for logistic
    cm = confusion_matrix(y_test, y_pred_lr, labels=sorted(set(labels)))
    out = Path("evaluation/baseline_confusion.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, cm)
    print(f"Saved confusion matrix to {out}")


if __name__ == "__main__":
    main()
