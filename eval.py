import pandas as pd
import json
import time
import matplotlib.pyplot as plt
from langdetect import detect

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


print("\n========= HEALTHBOT PROJECT EVALUATION =========\n")


# ------------------------------------------------
# 1 LOAD SYMPTOM DATASET
# ------------------------------------------------

data = pd.read_csv("Symptom2Disease.csv")

print("Dataset loaded:", len(data))

texts = data["text"]
labels = data["label"]


# ------------------------------------------------
# 2 TRAIN TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


# ------------------------------------------------
# 3 TEXT VECTORIZATION
# ------------------------------------------------

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ------------------------------------------------
# 4 TRAIN MODEL
# ------------------------------------------------

model = RandomForestClassifier()

model.fit(X_train_vec, y_train)

print("Model trained successfully")


# ------------------------------------------------
# 5 MODEL PREDICTIONS
# ------------------------------------------------

predictions = model.predict(X_test_vec)


# ------------------------------------------------
# 6 MODEL PERFORMANCE
# ------------------------------------------------

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted")
recall = recall_score(y_test, predictions, average="weighted")
f1 = f1_score(y_test, predictions, average="weighted")

print("\nMODEL PERFORMANCE")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ------------------------------------------------
# 7 CONFUSION MATRIX
# ------------------------------------------------

cm = confusion_matrix(y_test, predictions)

plt.figure()

plt.imshow(cm)

plt.title("Disease Classification Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.colorbar()

plt.show()


# ------------------------------------------------
# 8 TABLE 4 : RESPONSE LATENCY
# ------------------------------------------------

users = [50,100,150,200]

latency_results = []

for u in users:

    start = time.time()

    for i in range(u):

        sample = vectorizer.transform(["fever headache cough"])

        model.predict(sample)

    end = time.time()

    latency = (end-start)/u

    latency_results.append(latency)


table4 = pd.DataFrame({
    "Users":users,
    "Avg_Response_Time_sec":latency_results
})

print("\nTABLE 4 : SYSTEM LATENCY")

print(table4)


# latency graph

plt.figure()

plt.plot(users, latency_results, marker="o")

plt.xlabel("Number of Users")

plt.ylabel("Response Time (seconds)")

plt.title("Chatbot Response Latency")

plt.show()


# ------------------------------------------------
# 9 TABLE 6 : MULTILINGUAL PERFORMANCE
# ------------------------------------------------

languages = []
correct = []

for text,true,pred in zip(X_test, y_test, predictions):

    try:
        lang = detect(text)
    except:
        lang = "unknown"

    languages.append(lang)

    correct.append(true==pred)


lang_df = pd.DataFrame({
    "language":languages,
    "correct":correct
})


lang_accuracy = lang_df.groupby("language")["correct"].mean()

table6 = pd.DataFrame({
    "Language":lang_accuracy.index,
    "Accuracy":lang_accuracy.values
})


print("\nTABLE 6 : MULTILINGUAL PERFORMANCE")

print(table6)


# language graph

plt.figure()

plt.bar(table6["Language"], table6["Accuracy"])

plt.xlabel("Language")

plt.ylabel("Accuracy")

plt.title("Multilingual Performance")

plt.show()


print("\nEvaluation completed successfully")