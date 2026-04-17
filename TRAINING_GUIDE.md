# HealthBot Training Guide - Step by Step Procedure

This guide will walk you through training the HealthBot model using your datasets:
- **Symptom2Disease.csv** - Symptom to disease mapping dataset
- **__MACOSX folder** - Contains IndicNLG medical question datasets (train.json, dev.json, test.json)
- **ori_pqau.json** - PubMedQA medical question-answer dataset

## Prerequisites

1. **Python 3.10+** installed
2. **All required packages** installed (see requirements.txt)
3. **Dataset files** in the correct locations:
   - `Symptom2Disease.csv` in the project root
   - `__MACOSX/` folder in the project root (contains train.json, dev.json, test.json)
   - `ori_pqau.json` in the project root

## Step-by-Step Execution

### Step 1: Verify Dataset Files

Check that all dataset files are present:

```powershell
cd "c:\Users\Yeshwanth Reddy A R\Downloads\healthbot"
dir Symptom2Disease.csv
dir __MACOSX
dir ori_pqau.json
```

### Step 2: Install Dependencies

Install all required Python packages:

```powershell
pip install -r requirements.txt
```

**Note:** If you encounter issues with PyTorch or transformers, you may need to install them separately:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
```

### Step 3: Verify Project Structure

Ensure the following directory structure exists:
```
healthbot/
├── app/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_indicnlg.py
│   │   ├── load_pubmedqa.py
│   │   └── load_symptom2disease.py
│   ├── ml/
│   ├── nlp/
│   └── ...
├── Symptom2Disease.csv
├── __MACOSX/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── ori_pqau.json
├── train.py
└── requirements.txt
```

### Step 4: Create Required Directories

The training script will create these automatically, but you can create them manually:

```powershell
mkdir -p data\processed
mkdir -p models
```

### Step 5: Run Training

Execute the training script:

```powershell
python train.py
```

## What the Training Script Does

1. **Loads Symptom2Disease.csv**
   - Extracts symptom descriptions and disease labels
   - Processes ~1200 samples

2. **Loads PubMedQA (ori_pqau.json)**
   - Processes medical question-answer pairs
   - Classifies intents (symptom_reporting, treatment_info, etc.)

3. **Loads IndicNLG datasets from __MACOSX**
   - Processes multilingual medical questions
   - Supports 11 Indian languages (hi, bn, te, ta, mr, gu, kn, ml, pa, or, as)

4. **Combines all datasets**
   - Creates a unified training dataset
   - Splits into train/test sets

5. **Generates embeddings**
   - Uses language-aware embeddings for multilingual support

6. **Trains Intent Classifier**
   - Trains a classifier to identify user intents
   - Evaluates on test set
   - Saves model to `models/intent_classifier_trained.pkl`

7. **Saves processed datasets**
   - Saves combined dataset to `data/processed/combined_dataset.json`
   - Creates language-specific datasets

## Expected Output

You should see output like:

```
============================================================
Preparing Combined Dataset
============================================================

1. Loading Symptom2Disease.csv...
   ✓ Loaded 1200 Symptom2Disease samples

2. Loading PubMedQA (ori_pqau.json)...
   ✓ Loaded XXXX PubMedQA samples

3. Loading IndicNLG datasets from __MACOSX folder...
   Processing train split (XXXX samples)...
   Processing dev split (XXXX samples)...
   Processing test split (XXXX samples)...
   ✓ Total IndicNLG samples: XXXX

============================================================
Dataset Summary:
Total samples: XXXX
Languages: ['en', 'hi', 'bn', ...]
Sources: ['symptom2disease', 'pubmedqa', 'indicnlg']
Intents: ['symptom_reporting', 'disease_info', ...]
============================================================

============================================================
Training Models
============================================================

1. Preparing data for intent classification...
   Using XXXX samples for intent training

2. Generating embeddings...
   Generated XXXX embeddings

3. Splitting data...
   Train: XXXX samples
   Test: XXXX samples

4. Training Intent Classifier...

5. Evaluating...
   Overall Accuracy: XX.XX%

6. Language-wise Accuracy:
   en: XX.XX% (XXXX samples)
   hi: XX.XX% (XXXX samples)
   ...

7. Saving models...
✓ Training complete!
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Ensure you're running from the project root directory:
```powershell
cd "c:\Users\Yeshwanth Reddy A R\Downloads\healthbot"
python train.py
```

### Issue: FileNotFoundError for datasets

**Solution:** Verify file paths:
- `Symptom2Disease.csv` should be in the root directory
- `__MACOSX` folder should be in the root directory
- `ori_pqau.json` should be in the root directory

### Issue: JSON decode errors

**Solution:** The JSON files in `__MACOSX` are JSONL format (one JSON object per line). The loader handles this automatically.

### Issue: Memory errors during embedding generation

**Solution:** If you have limited RAM, you may need to:
1. Process datasets in batches
2. Use smaller embedding models
3. Reduce the dataset size for initial testing

### Issue: Missing dependencies

**Solution:** Install missing packages:
```powershell
pip install <package-name>
```

## Next Steps After Training

1. **Test the trained model:**
   ```powershell
   python test_system.py
   ```

2. **Train additional models:**
   - Symptom extractor
   - Disease classifier

3. **Run evaluation:**
   ```powershell
   python evaluation/evaluate.py
   ```

## Files Created During Training

- `data/processed/combined_dataset.json` - Combined training dataset
- `data/processed/{lang}_dataset.json` - Language-specific datasets
- `models/intent_classifier_trained.pkl` - Trained intent classifier
- `models/language_accuracy.json` - Language-wise accuracy metrics

## Notes

- The training process may take 10-30 minutes depending on your hardware
- Embedding generation is the most time-consuming step
- The model supports multilingual inputs (English + 11 Indian languages)
- All datasets are automatically preprocessed and converted to a unified format
