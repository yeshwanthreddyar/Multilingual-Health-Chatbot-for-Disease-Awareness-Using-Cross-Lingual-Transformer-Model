# Training Scripts Guide

## Available Training Scripts

### 1. `train.py` - Intent Classifier Training
**Purpose:** Trains the intent classifier to identify user intents (symptom_reporting, disease_info, etc.)

**Usage:**
```powershell
python train.py
```

**What it does:**
- Loads Symptom2Disease.csv, PubMedQA, and IndicNLG datasets
- Generates embeddings for all text samples
- Trains an ensemble classifier (Naive Bayes + Random Forest + Gradient Boosting)
- Saves model to `models/intent_classifier_trained.pkl`

**Output:**
- `models/intent_classifier_trained.pkl` - Trained intent classifier
- `models/language_accuracy.json` - Language-wise accuracy metrics
- `data/processed/combined_dataset.json` - Combined training dataset

---

### 2. `train_disease_classifier.py` - Disease Classifier Training
**Purpose:** Trains the disease classifier to predict diseases from symptom descriptions

**Usage:**
```powershell
python train_disease_classifier.py
```

**What it does:**
- Loads Symptom2Disease.csv dataset
- Generates embeddings for symptom descriptions
- Trains an ensemble classifier to predict diseases
- Evaluates and saves the model

**Output:**
- `models/disease_classifier_trained.pkl` - Trained disease classifier
- `models/disease_distribution.json` - Disease label distribution

---

### 3. Symptom Extractor - No Training Needed
**Note:** The symptom extractor is **rule-based** (lexicon-based) and does **not** require training. It uses a predefined symptom lexicon (`SYMPTOM_LEXICON`) to extract symptoms from text. The extractor is already functional and doesn't need a training script.

**Location:** `app/ml/symptom_extractor.py`

---

## Evaluation

### `evaluation/evaluate.py`
Contains utility functions for evaluation. You can create a custom evaluation script that:
- Loads the trained models
- Tests on a test dataset
- Measures accuracy and latency
- Generates language-wise reports

**Example usage:**
```python
from evaluation.evaluate import run_accuracy_check, run_latency_check
# Use these functions with your models and test data
```

---

## Training Order

1. **First:** Run `train.py` to train the intent classifier
2. **Second:** Run `train_disease_classifier.py` to train the disease classifier
3. **Third:** Test the system with `python test_system.py`
4. **Finally:** Run custom evaluation scripts

---

## Quick Start

```powershell
# Step 1: Train intent classifier
python train.py

# Step 2: Train disease classifier
python train_disease_classifier.py

# Step 3: Test the system
python test_system.py
```

---

## Notes

- Both training scripts will create the `models/` directory automatically
- The symptom extractor is already functional and doesn't need training
- Make sure all dataset files are in the correct locations before training
- Training may take 10-30 minutes depending on your hardware
