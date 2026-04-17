#!/bin/bash
# setup_and_train.sh

echo "Setting up HealthBot training with your datasets..."

# 1. Organize your files
mkdir -p data/datasets
mkdir -p data/pubmedqa
mkdir -p data/indicnlp

echo "Copy your files to:"
echo "  PubMedQA: data/pubmedqa/ori_pqau.json"
echo "  IndicNLG: data/indicnlp/{lang}.{train,dev,test}.json"

# 2. Install dependencies
pip install -r requirements.txt
pip install scikit-learn==1.3.0 pandas==2.0.3 joblib==1.3.2

# 3. Run training
echo "Starting training..."
python app/training/train_with_your_files.py

# 4. Test the system
echo "Testing the trained system..."
python test_system.py