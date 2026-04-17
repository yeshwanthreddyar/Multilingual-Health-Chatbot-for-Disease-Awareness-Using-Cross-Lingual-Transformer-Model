"""Simple dataset augmentation utilities.

Reads `Symptom2Disease.csv` by default and writes augmented CSV to `data/expanded_symptoms.csv`.
This is intentionally lightweight: it performs random shuffling of clauses and small token deletions
to increase dataset size while preserving label distribution.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List


def simple_augment(text: str) -> str:
    # split on commas and periods to get clauses
    clauses = [c.strip() for c in text.replace(";", ",").split(",") if c.strip()]
    if not clauses:
        tokens = text.split()
        if len(tokens) > 1:
            # randomly drop one token
            i = random.randrange(len(tokens))
            del tokens[i]
            return " ".join(tokens)
        return text
    random.shuffle(clauses)
    # optionally drop a short clause
    if len(clauses) > 1 and random.random() < 0.25:
        idx = random.randrange(len(clauses))
        clauses.pop(idx)
    return ", ".join(clauses)


def augment_file(in_csv: Path, out_csv: Path, multiplier: int = 5, seed: int = 42) -> None:
    random.seed(seed)
    out_rows: List[List[str]] = []
    with in_csv.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # preserve header
        for row in reader:
            if len(row) < 3:
                continue
            label = row[1]
            text = row[2]
            out_rows.append([label, text])
            # generate augmented versions
            for i in range(multiplier - 1):
                aug = simple_augment(text)
                out_rows.append([label, aug])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "text"])
        for r in out_rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("Symptom2Disease.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/expanded_symptoms.csv"))
    parser.add_argument("--multiplier", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    augment_file(args.input, args.output, multiplier=args.multiplier, seed=args.seed)


if __name__ == "__main__":
    main()
