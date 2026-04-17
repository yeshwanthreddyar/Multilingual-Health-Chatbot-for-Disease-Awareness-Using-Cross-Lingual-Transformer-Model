# Quick Start Guide - HealthBot Training

## 🚀 Quick Execution Steps

### 1. Navigate to Project Directory
```powershell
cd "c:\Users\Yeshwanth Reddy A R\Downloads\healthbot"
```

### 2. Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### 3. Run Training
```powershell
python train.py
```

That's it! The script will:
- ✅ Load Symptom2Disease.csv (~1200 samples)
- ✅ Load PubMedQA from ori_pqau.json
- ✅ Load IndicNLG datasets from __MACOSX folder
- ✅ Combine all datasets
- ✅ Train intent classifier
- ✅ Save models to `models/` directory

## 📁 Required Files Location

Make sure these files are in the project root:
- ✅ `Symptom2Disease.csv`
- ✅ `__MACOSX/` folder (with train.json, dev.json, test.json)
- ✅ `ori_pqau.json`

## 📊 What Gets Created

After training, you'll find:
- `data/processed/combined_dataset.json` - Combined dataset
- `models/intent_classifier_trained.pkl` - Trained model
- `models/language_accuracy.json` - Performance metrics

## ⚠️ Common Issues

**Problem:** `ModuleNotFoundError`
- **Fix:** Run `pip install -r requirements.txt`

**Problem:** `FileNotFoundError`
- **Fix:** Ensure all dataset files are in the project root directory

**Problem:** JSON decode errors
- **Fix:** Already handled - JSONL format is automatically parsed

## 📖 Full Documentation

See `TRAINING_GUIDE.md` for detailed documentation and troubleshooting.
