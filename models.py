import csv
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


# -----------------------------
# Load dataset used in project
# -----------------------------

ROOT = Path(__file__).resolve().parent

# Prefer expanded dataset if available (created by scripts/augment_dataset.py)
expanded = ROOT / "data" / "expanded_symptoms.csv"
source = ROOT / "Symptom2Disease.csv"

def load_dataset(path: Path):
    texts = []
    labels = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        hdr = next(reader, None)
        for row in reader:
            if not row:
                continue
            if len(row) == 2:
                labels.append(row[0])
                texts.append(row[1])
            elif len(row) >= 3:
                # tolerate (idx,label,text)
                labels.append(row[1])
                texts.append(row[2])
    return pd.DataFrame({"text": texts, "label": labels})


if expanded.exists():
    data = load_dataset(expanded)
else:
    data = load_dataset(source)

X = data["text"]
y = data["label"]


# -----------------------------
# Train / Test split
# -----------------------------

# If dataset is small, apply lightweight augmentation to increase samples
def simple_augment(text: str) -> str:
    clauses = [c.strip() for c in text.replace(";", ",").split(",") if c.strip()]
    if not clauses:
        tokens = text.split()
        if len(tokens) > 1:
            i = random.randrange(len(tokens))
            del tokens[i]
            return " ".join(tokens)
        return text
    random.shuffle(clauses)
    if len(clauses) > 1 and random.random() < 0.25:
        clauses.pop(random.randrange(len(clauses)))
    return ", ".join(clauses)


if len(data) < 2000:
    # expand to approximately 5x but keep label distribution
    rows = []
    for _, r in data.iterrows():
        rows.append((r["text"], r["label"]))
        for _ in range(4):
            rows.append((simple_augment(r["text"]), r["label"]))
    aug_df = pd.DataFrame(rows, columns=["text", "label"])
    X = aug_df["text"]
    y = aug_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# TF-IDF Vectorization
# -----------------------------

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -----------------------------
# Model 1 — TF-IDF + Naive Bayes
# (Baseline chatbot)
# -----------------------------

nb_model = MultinomialNB()

nb_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)

nb_accuracy = accuracy_score(y_test, nb_pred)


# -----------------------------
# Model 2 — Random Forest
# -----------------------------

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train_vec, y_train)

rf_pred = rf_model.predict(X_test_vec)

rf_accuracy = accuracy_score(y_test, rf_pred)


# -----------------------------
# Model 3 — Gradient Boosting
# -----------------------------

gb_model = GradientBoostingClassifier()

gb_model.fit(X_train_vec.toarray(), y_train)

gb_pred = gb_model.predict(X_test_vec.toarray())

gb_accuracy = accuracy_score(y_test, gb_pred)

# --- Logistic regression baseline (Bag-of-words + TF-IDF)
lr_pipe = Pipeline(
    [("vect", CountVectorizer(ngram_range=(1, 2), max_features=20000)), ("tfidf", TfidfTransformer()), ("clf", LogisticRegression(max_iter=1000))]
)
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

# --- Simple keyword baseline
def build_keyword_map(texts, labels, top_k=20):
    from collections import defaultdict, Counter

    token_counts = defaultdict(Counter)
    for t, l in zip(texts, labels):
        tokens = [w.lower().strip(".,") for w in str(t).split() if len(w) > 2]
        token_counts[l].update(tokens)
    return {l: [w for w, _ in token_counts[l].most_common(top_k)] for l in token_counts}

def predict_keyword(keywords, text: str) -> str:
    from collections import Counter

    tokens = [w.lower().strip(".,") for w in str(text).split() if len(w) > 2]
    scores = Counter()
    for t in tokens:
        for label, kws in keywords.items():
            if t in kws:
                scores[label] += 1
    if not scores:
        # fallback: most common label in training
        return Counter(y_train).most_common(1)[0][0]
    return scores.most_common(1)[0][0]

keywords = build_keyword_map(X_train, y_train)
kw_pred = [predict_keyword(keywords, t) for t in X_test]
kw_accuracy = accuracy_score(y_test, kw_pred)


# -----------------------------
# Results table
# -----------------------------

models = [
    "TF-IDF + Naive Bayes",
    "Random Forest",
    "Gradient Boosting (Proposed)",
    "Logistic Regression (Baseline)",
    "Keyword Baseline",
]

accuracies = [nb_accuracy, rf_accuracy, gb_accuracy, lr_accuracy, kw_accuracy]


results = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracies
})

print("\nModel Comparison Results\n")
print(results)

print("\nDetailed accuracies:\n")
print(f"Naive Bayes: {nb_accuracy:.4f}")
print(f"Random Forest: {rf_accuracy:.4f}")
print(f"Gradient Boosting: {gb_accuracy:.4f}")
print(f"Logistic Regression baseline: {lr_accuracy:.4f}")
print(f"Keyword baseline: {kw_accuracy:.4f}")


# -----------------------------
# Plot graph for research paper
# -----------------------------

plt.figure(figsize=(6,5))

plt.bar(models, accuracies)

plt.ylim(0,1)

plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.title("Performance Comparison of Disease Prediction Models")

plt.savefig("model_accuracy_comparison.png", dpi=300)

plt.show()