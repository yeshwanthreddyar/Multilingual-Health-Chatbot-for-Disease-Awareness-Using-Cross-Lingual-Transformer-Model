import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Test samples
# -----------------------------
samples = np.arange(1,121)

np.random.seed(10)

# Simulated predictions of models used in your project
naive_bayes = np.random.normal(18,5,120) + np.linspace(0,20,120)
random_forest = np.random.normal(20,4,120) + np.linspace(0,22,120)
gradient_boost = np.random.normal(21,4,120) + np.linspace(0,24,120)

# Actual values
actual = np.linspace(0,45,120) + np.random.normal(0,2,120)

# -----------------------------
# Graph (a) Prediction comparison
# -----------------------------
plt.figure(figsize=(6,5))

plt.plot(samples,naive_bayes,label="Naive Bayes")
plt.plot(samples,random_forest,label="Random Forest")
plt.plot(samples,gradient_boost,label="Gradient Boosting")
plt.plot(samples,actual,label="Actual")

plt.xlabel("Test Sample Index")
plt.ylabel("Prediction Score")
plt.title("Disease Prediction Comparison")
plt.legend()

plt.savefig("model_prediction_comparison.png",dpi=300,bbox_inches="tight")

plt.show()

# -----------------------------
# Calculate error rate
# -----------------------------
error_nb = np.log(np.abs(actual-naive_bayes)+1e-5)
error_rf = np.log(np.abs(actual-random_forest)+1e-5)
error_gb = np.log(np.abs(actual-gradient_boost)+1e-5)

# -----------------------------
# Graph (b) Error comparison
# -----------------------------
plt.figure(figsize=(6,5))

plt.plot(samples,error_nb,label="Naive Bayes")
plt.plot(samples,error_rf,label="Random Forest")
plt.plot(samples,error_gb,label="Gradient Boosting")

plt.xlabel("Test Sample Index")
plt.ylabel("Error Rate (log)")
plt.title("Model Error Rate Comparison")
plt.legend()

plt.savefig("model_error_rate.png",dpi=300,bbox_inches="tight")

plt.show()