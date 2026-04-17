import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("Symptom2Disease.csv")

X = data["text"]
y = data["label"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Text Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test_vec)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.xlabel("Predicted Disease")
plt.ylabel("Actual Disease")
plt.title("Confusion Matrix - AI Health Chatbot")

plt.savefig("confusion_matrix_healthbot.png", dpi=300)

plt.show()

# -----------------------------
# Evaluation Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

metrics = ["Accuracy","Precision","Recall","F1 Score"]
scores = [accuracy, precision, recall, f1]

print("\nEvaluation Matrix\n")
for m,s in zip(metrics,scores):
    print(f"{m}: {s:.3f}")

# -----------------------------
# Evaluation Graph
# -----------------------------
plt.figure(figsize=(6,5))

plt.bar(metrics, scores)

plt.ylim(0,1)

plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.title("Evaluation Matrix - AI Health Chatbot")

plt.savefig("evaluation_matrix_healthbot.png", dpi=300)

plt.show()