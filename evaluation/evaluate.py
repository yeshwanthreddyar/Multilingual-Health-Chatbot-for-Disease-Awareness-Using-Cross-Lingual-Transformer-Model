"""
Evaluation - Accuracy ≥ 80%, Mean latency ≤ 2 seconds, Language-wise accuracy report.
Sample conversations in 15 languages (placeholder).
"""
from __future__ import annotations

import time
from typing import List, Dict, Any

# Placeholder: run against test set and measure accuracy + latency
SUPPORTED_LANGS = [
    "en", "hi", "bn", "te", "ta", "kn", "ml", "mr", "gu", "pa", "or", "as", "ur", "ne", "kok",
]


def run_latency_check(chat_fn, samples: List[Dict[str, str]], max_latency_sec: float = 2.0) -> Dict[str, Any]:
    """Measure mean latency; target ≤ 2 seconds."""
    times_sec = []
    for s in samples:
        t0 = time.perf_counter()
        chat_fn(s.get("message", "hello"))
        times_sec.append(time.perf_counter() - t0)
    mean_latency = sum(times_sec) / len(times_sec) if times_sec else 0.0
    return {
        "mean_latency_sec": mean_latency,
        "target_sec": max_latency_sec,
        "passed": mean_latency <= max_latency_sec,
        "n_samples": len(samples),
    }


def run_accuracy_check(predict_fn, X, y_true: List[str]) -> Dict[str, Any]:
    """Accuracy ≥ 80%."""
    y_pred = predict_fn(X)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / len(y_true) if y_true else 0.0
    return {
        "accuracy": acc,
        "target": 0.80,
        "passed": acc >= 0.80,
        "n_samples": len(y_true),
    }


def language_wise_accuracy(predict_fn, lang_to_samples: Dict[str, List[tuple]]) -> Dict[str, Dict]:
    """Language-wise accuracy report."""
    report = {}
    for lang, samples in lang_to_samples.items():
        if not samples:
            continue
        X = [s[0] for s in samples]
        y_true = [s[1] for s in samples]
        y_pred = predict_fn(X)
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        report[lang] = {
            "accuracy": correct / len(y_true),
            "n_samples": len(y_true),
        }
    return report
