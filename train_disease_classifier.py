"""
Train Disease Classifier using Symptom2Disease dataset
Maps symptoms to diseases using the combined dataset
"""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.ml.disease_classifier import DiseaseClassifier
from app.ml.disease_classifier import DISEASE_LABELS
from app.data.load_symptom2disease import load_symptom2disease
from app.nlp.embeddings import get_embedding_generator

# Note: The DiseaseClassifier uses a fixed set of DISEASE_LABELS.
# Diseases not in DISEASE_LABELS will be mapped to "other"
def map_disease_to_classifier_label(disease: str) -> str:
    """Map dataset disease names to classifier labels"""
    disease_lower = disease.lower().strip()
    
    # Direct mappings
    mapping = {
        'psoriasis': 'skin_infection',
        'varicose veins': 'other',
        'peptic ulcer disease': 'gastroenteritis',
        'drug reaction': 'other',
        'gastroesophageal reflux disease': 'gastroenteritis',
        'allergy': 'other',
        'urinary tract infection': 'other',
        'malaria': 'malaria',
        'jaundice': 'other',
        'cervical spondylosis': 'other',
        'migraine': 'other',
        'hypertension': 'hypertension',
        'bronchial asthma': 'respiratory_infection',
        'acne': 'skin_infection',
        'arthritis': 'other',
        'dimorphic hemorrhoids': 'other',
        'pneumonia': 'respiratory_infection',
        'common cold': 'common_cold',
        'fungal infection': 'skin_infection',
        'dengue': 'dengue',
        'typhoid': 'typhoid',
        'diabetes': 'diabetes_awareness',
        'flu': 'flu',
    }
    
    # Check direct mapping
    if disease_lower in mapping:
        return mapping[disease_lower]
    
    # Check if any classifier label is in the disease name
    for label in DISEASE_LABELS:
        if label.replace('_', ' ') in disease_lower or disease_lower in label.replace('_', ' '):
            return label
    
    # Default to "other"
    return 'other'

def prepare_disease_training_data():
    """Prepare training data for disease classification"""
    print("=" * 60)
    print("Preparing Disease Classification Dataset")
    print("=" * 60)
    
    # Load Symptom2Disease dataset
    print("\n1. Loading Symptom2Disease.csv...")
    df = load_symptom2disease("Symptom2Disease.csv")
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Filter to samples with disease labels
    df = df[df['disease'].notna() & (df['disease'] != '')].copy()
    print(f"   ✓ {len(df)} samples with disease labels")
    
    # Show disease distribution
    print("\n2. Disease distribution:")
    disease_counts = df['disease'].value_counts()
    print(f"   Total unique diseases: {len(disease_counts)}")
    for disease, count in disease_counts.head(15).items():
        mapped = map_disease_to_classifier_label(disease)
        print(f"   {disease} → {mapped}: {count} samples")
    
    return df

def train_disease_classifier(df: pd.DataFrame):
    """Train disease classifier on symptom-disease pairs"""
    
    print("\n" + "=" * 60)
    print("Training Disease Classifier")
    print("=" * 60)
    
    # 1. Prepare features (symptom embeddings)
    print("\n1. Generating symptom embeddings...")
    emb_generator = get_embedding_generator()
    
    X = []
    y = []
    
    for idx, row in df.iterrows():
        try:
            text = str(row['text']).strip()
            disease = str(row['disease']).strip()
            
            # Skip if text is empty or too short
            if not text or len(text) < 3:
                continue
            
            # Generate embedding for symptom description
            embedding = emb_generator.encode_single(text, lang_hint='en')
            
            if len(embedding) == 0:
                continue
            
            # Map disease to classifier label
            mapped_disease = map_disease_to_classifier_label(disease)
            
            X.append(embedding)
            y.append(mapped_disease)
        except Exception as e:
            if len(X) < 5:
                print(f"   Warning: Could not process row {idx}: {str(e)[:100]}")
            continue
    
    if len(X) == 0:
        print("\n   ERROR: No valid samples for training!")
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   Generated {len(X)} embeddings")
    
    # 2. Split data
    print("\n2. Splitting data...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # If stratification fails (not enough samples per class), split without it
        print("   Warning: Stratification failed, splitting without stratification")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # 3. Train classifier
    print("\n3. Training Disease Classifier...")
    classifier = DiseaseClassifier()
    classifier.fit(X_train, y_train.tolist())
    
    # 4. Evaluate
    print("\n4. Evaluating...")
    y_pred = []
    for x in X_test:
        top_diseases = classifier.top_k(x.reshape(1, -1), k=1)
        y_pred.append(top_diseases[0][0] if top_diseases else "other")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Overall Accuracy: {accuracy:.2%}")
    
    # Show classification report
    print("\n5. Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 6. Save model
    print("\n6. Saving model...")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(classifier, 'models/disease_classifier_trained.pkl')
    print("   ✓ Saved to models/disease_classifier_trained.pkl")
    
    # Save disease distribution
    disease_dist = pd.Series(y).value_counts().to_dict()
    with open('models/disease_distribution.json', 'w') as f:
        json.dump(disease_dist, f, indent=2)
    print("   ✓ Saved disease distribution to models/disease_distribution.json")
    
    print("\n✓ Training complete!")
    return classifier

def main():
    """Main execution"""
    
    # Create directories if they don't exist
    Path("models").mkdir(exist_ok=True)
    
    # 1. Prepare dataset
    df = prepare_disease_training_data()
    
    if len(df) == 0:
        print("\nERROR: No data to train on!")
        return
    
    # 2. Train classifier
    classifier = train_disease_classifier(df)
    
    if classifier:
        # 3. Test with examples
        print("\n" + "=" * 60)
        print("Testing with Examples")
        print("=" * 60)
        
        test_examples = [
            "I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches.",
            "I have been experiencing joint pain in my fingers, wrists, and knees. The pain is often achy and throbbing.",
            "I have fever and headache with body pain."
        ]
        
        emb_generator = get_embedding_generator()
        
        for text in test_examples:
            embedding = emb_generator.encode_single(text, lang_hint='en')
            if len(embedding) > 0:
                top_diseases = classifier.top_k(embedding.reshape(1, -1), k=3)
                print(f"\nText: {text[:80]}...")
                print("Top predicted diseases:")
                for disease, confidence in top_diseases:
                    print(f"  - {disease}: {confidence:.2%}")
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("1. Run evaluation: python evaluation/evaluate.py")
        print("2. Test the system: python test_system.py")
        print("=" * 60)

if __name__ == "__main__":
    main()
