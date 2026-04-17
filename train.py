# app/training/train_with_your_files.py
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.data.load_pubmedqa import load_pubmedqa
from app.data.load_symptom2disease import load_symptom2disease
from app.data.load_custom_multilingual import load_custom_multilingual
from train_disease_classifier import map_disease_to_classifier_label

def prepare_combined_dataset() -> pd.DataFrame:
    """Combine PubMedQA, IndicNLG, and Symptom2Disease for training"""
    
    print("=" * 60)
    print("Preparing Combined Dataset")
    print("=" * 60)
    
    all_samples = []
    
    # 1. Load Symptom2Disease.csv
    print("\n1. Loading Symptom2Disease.csv...")
    try:
        symptom2disease_df = load_symptom2disease("Symptom2Disease.csv")
        print(f"   ✓ Loaded {len(symptom2disease_df)} Symptom2Disease samples")
        
        # Add to training data
        for _, row in symptom2disease_df.iterrows():
            all_samples.append({
                'text': row['text'],
                'language': row['language'],
                'intent': row['intent'],
                'symptoms': row.get('symptoms', []),
                'disease': row.get('disease', ''),
                'is_emergency': row.get('is_emergency', False),
                'source': 'symptom2disease',
                'split': row.get('split', 'train')
            })
    except Exception as e:
        print(f"   ✗ Error loading Symptom2Disease: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Load PubMedQA
    print("\n2. Loading PubMedQA (ori_pqau.json)...")
    try:
        pubmedqa_df = load_pubmedqa("ori_pqau.json")
        print(f"   ✓ Loaded {len(pubmedqa_df)} PubMedQA samples")
        
        # Add to training data
        for _, row in pubmedqa_df.iterrows():
            all_samples.append({
                'text': row['text'],
                'language': 'en',
                'intent': row['intent'],
                'symptoms': row.get('symptoms', []),
                'diseases': row.get('diseases', []),
                'answer': row.get('answer', ''),
                'source': 'pubmedqa',
                'split': 'train'  # Use all for training initially
            })
    except Exception as e:
        print(f"   ✗ Error loading PubMedQA: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Load custom multilingual dataset (user-provided)
    print("\n3. Loading custom multilingual dataset (if present)...")
    try:
        custom_df = load_custom_multilingual("data/custom_multilingual.csv")
        print(f"   ✓ Loaded {len(custom_df)} custom multilingual samples")
        for _, row in custom_df.iterrows():
            all_samples.append({
                "text": row["text"],
                "language": row["language"],
                "intent": row["intent"],
                "symptoms": row.get("symptoms", []),
                "disease": row.get("disease", ""),
                "diseases": row.get("diseases", []),
                "answer": row.get("answer", ""),
                "source": row.get("source", "custom_multilingual"),
                "split": row.get("split", "train"),
            })
    except FileNotFoundError:
        print("   No custom multilingual dataset found at data/custom_multilingual.csv (skipping).")
    except Exception as e:
        print(f"   ✗ Error loading custom multilingual dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert to DataFrame
    if len(all_samples) == 0:
        print("\n   ERROR: No samples loaded! Check your dataset files.")
        return pd.DataFrame()
    
    combined_df = pd.DataFrame(all_samples)
    
    # Validate that text column exists and has valid data
    if 'text' not in combined_df.columns:
        print("\n   ERROR: 'text' column missing from combined dataset!")
        return pd.DataFrame()
    
    # Show sample of text to verify it's not IDs
    print("\n   Sample texts (first 3):")
    for i, text in enumerate(combined_df['text'].head(3)):
        print(f"     {i+1}. {str(text)[:100]}...")
    
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Languages: {combined_df['language'].unique().tolist()}")
    print(f"Sources: {combined_df['source'].unique().tolist()}")
    print(f"Intents: {combined_df['intent'].unique().tolist()}")
    print("=" * 60)
    
    return combined_df

def create_language_specific_datasets(df: pd.DataFrame):
    """Create separate datasets for each language"""
    
    print("\nCreating language-specific datasets...")
    
    languages = df['language'].unique()
    language_datasets = {}
    
    for lang in languages:
        lang_df = df[df['language'] == lang].copy()
        language_datasets[lang] = lang_df
        
        print(f"  {lang}: {len(lang_df)} samples")
        
        # Save language-specific dataset
        output_path = f"data/processed/{lang}_dataset.json"
        lang_df.to_json(output_path, orient='records', force_ascii=False)
        print(f"    Saved to {output_path}")
    
    return language_datasets

def map_intent_to_classifier_format(intent: str) -> str:
    """Map training intents to IntentClassifier's expected format"""
    intent_mapping = {
        'disease_info': 'disease_information',
        'disease_information': 'disease_information',
        'symptom_reporting': 'symptom_reporting',
        'prevention_guidance': 'prevention_guidance',
        'treatment_info': 'disease_information',  # Map treatment to disease_info
        'diagnosis_info': 'disease_information',    # Map diagnosis to disease_info
        'cause_info': 'disease_information',       # Map cause to disease_info
        'general_health_query': 'general_health_query',
        'vaccination_schedule': 'vaccination_schedule',
        'emergency_assessment': 'emergency_assessment'
    }
    return intent_mapping.get(intent, 'general_health_query')

def train_models_on_combined(df: pd.DataFrame):
    """Train ensemble intent and disease models on the combined multilingual dataset."""
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from app.ml.intent_classifier import IntentClassifier
    from app.ml.disease_classifier import DiseaseClassifier
    from app.nlp.embeddings import get_embedding_generator
    import joblib
    
    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)
    
    # 1. Prepare data for intent classification
    print("\n1. Preparing data for intent classification...")
    
    filtered_df = df.copy()
    
    # Show intent distribution before mapping
    print(f"   Intent distribution before mapping:")
    intent_counts = filtered_df['intent'].value_counts()
    for intent, count in intent_counts.items():
        print(f"     {intent}: {count}")
    
    print(f"   Using {len(filtered_df)} samples for intent training")
    
    # 2. Get embeddings (language-aware) for intent model
    print("\n2. Generating embeddings for intent model...")
    emb_generator = get_embedding_generator()
    
    X_intent = []
    y_intent = []
    language_labels = []
    
    for idx, row in filtered_df.iterrows():
        try:
            text = str(row['text']).strip()
            if not text or text.isdigit():
                continue
            lang = str(row['language']) if 'language' in row and pd.notna(row['language']) else None
            if len(text) < 3:
                continue
            embedding = emb_generator.encode_single(text, lang_hint=lang)
            if len(embedding) == 0:
                continue
            X_intent.append(embedding)
            y_intent.append(row['intent'])
            language_labels.append(lang if lang else 'en')
        except Exception as e:
            if len(X_intent) < 5:
                print(f"   Warning: Could not embed row {idx} for intent: {str(e)[:100]}")
            continue
    
    if len(X_intent) == 0:
        print("\n   ERROR: No embeddings generated for intent model! Skipping training.")
        return None
    
    X_intent = np.array(X_intent)
    y_intent = np.array(y_intent)
    
    print(f"   Generated {len(X_intent)} embeddings for intent model")
    
    # 3. Split data for intent classifier
    print("\n3. Splitting data for intent classifier...")
    
    if len(set(y_intent)) > 1 and len(X_intent) >= 10:
        try:
            X_train_i, X_test_i, y_train_i, y_test_i, lang_train, lang_test = train_test_split(
                X_intent, y_intent, language_labels, test_size=0.2, random_state=42, stratify=y_intent
            )
        except ValueError:
            print("   Warning: Stratification failed for intent model, splitting without stratification")
            X_train_i, X_test_i, y_train_i, y_test_i, lang_train, lang_test = train_test_split(
                X_intent, y_intent, language_labels, test_size=0.2, random_state=42
            )
    else:
        print("   Warning: Not enough samples for proper intent train/test split")
        split_idx = int(len(X_intent) * 0.8)
        X_train_i, X_test_i = X_intent[:split_idx], X_intent[split_idx:]
        y_train_i, y_test_i = y_intent[:split_idx], y_intent[split_idx:]
        lang_train, lang_test = language_labels[:split_idx], language_labels[split_idx:]
    
    print(f"   Intent Train: {len(X_train_i)} samples")
    print(f"   Intent Test: {len(X_test_i)} samples")
    
    # 4. Train intent classifier (ensemble)
    print("\n4. Training Intent Classifier (ensemble)...")
    intent_classifier = IntentClassifier()
    
    y_train_mapped = [map_intent_to_classifier_format(intent) for intent in y_train_i]
    y_test_mapped = [map_intent_to_classifier_format(intent) for intent in y_test_i]
    
    intent_classifier.fit(X_train_i, y_train_mapped)
    
    # 5. Evaluate intent classifier
    print("\n5. Evaluating Intent Classifier...")
    
    y_pred_i = intent_classifier.predict(X_test_i)
    accuracy_i = accuracy_score(y_test_mapped, y_pred_i)
    print(f"   Overall Intent Accuracy: {accuracy_i:.2%}")
    
    print("\n6. Language-wise Intent Accuracy:")
    lang_accuracies = {}
    for lang in set(lang_test):
        lang_mask = [l == lang for l in lang_test]
        if sum(lang_mask) > 10:
            lang_X = X_test_i[lang_mask]
            lang_y = [y_test_mapped[i] for i in range(len(lang_test)) if lang_mask[i]]
            lang_pred = intent_classifier.predict(lang_X)
            lang_accuracy = accuracy_score(lang_y, lang_pred)
            lang_accuracies[lang] = lang_accuracy
            print(f"   {lang}: {lang_accuracy:.2%} ({sum(lang_mask)} samples)")
    
    # 7. Save intent model and metrics
    print("\n7. Saving intent model...")
    joblib.dump(intent_classifier, 'models/intent_classifier_trained.pkl')
    with open('models/language_accuracy.json', 'w') as f:
        json.dump(lang_accuracies, f, indent=2)
    
    # 8. Train disease classifier on multilingual conversational data (where available)
    print("\n8. Preparing data for Disease Classifier from combined dataset...")
    
    disease_texts = []
    disease_labels = []
    
    for idx, row in df.iterrows():
        disease_value = None
        if 'disease' in row and pd.notna(row['disease']) and str(row['disease']).strip():
            disease_value = str(row['disease']).strip()
        elif 'diseases' in row and isinstance(row['diseases'], (list, tuple)) and row['diseases']:
            disease_value = str(row['diseases'][0]).strip()
        
        if not disease_value:
            continue
        
        text = str(row['text']).strip()
        if not text or len(text) < 3:
            continue
        
        lang = str(row['language']) if 'language' in row and pd.notna(row['language']) else None
        
        try:
            emb = emb_generator.encode_single(text, lang_hint=lang)
            if len(emb) == 0:
                continue
        except Exception as e:
            if len(disease_texts) < 5:
                print(f"   Warning: Could not embed row {idx} for disease: {str(e)[:100]}")
            continue
        
        mapped_label = map_disease_to_classifier_label(disease_value)
        disease_texts.append(emb)
        disease_labels.append(mapped_label)
    
    if not disease_texts:
        print("   No disease-labelled multilingual samples in combined dataset; skipping disease training here.")
        print("\n✓ Intent training complete!")
        return intent_classifier
    
    X_disease = np.array(disease_texts)
    y_disease = np.array(disease_labels)
    
    print(f"   Found {len(X_disease)} multilingual samples with disease labels for training")
    
    print("\n9. Splitting data for Disease Classifier...")
    try:
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
            X_disease, y_disease, test_size=0.2, random_state=42, stratify=y_disease
        )
    except ValueError:
        print("   Warning: Stratification failed for disease model, splitting without stratification")
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
            X_disease, y_disease, test_size=0.2, random_state=42
        )
    
    print(f"   Disease Train: {len(X_train_d)} samples")
    print(f"   Disease Test: {len(X_test_d)} samples")
    
    print("\n10. Training Disease Classifier (ensemble) on multilingual conversational data...")
    disease_classifier = DiseaseClassifier()
    disease_classifier.fit(X_train_d, y_train_d.tolist())
    
    print("\n11. Evaluating Disease Classifier...")
    y_pred_d = []
    for x in X_test_d:
        top = disease_classifier.top_k(x.reshape(1, -1), k=1)
        y_pred_d.append(top[0][0] if top else "other")
    
    accuracy_d = accuracy_score(y_test_d, y_pred_d)
    print(f"   Overall Disease Accuracy (multilingual training set): {accuracy_d:.2%}")
    print("\nDisease Classification Report (multilingual training set):")
    print(classification_report(y_test_d, y_pred_d, zero_division=0))
    
    print("\n12. Saving Disease Classifier trained on multilingual conversational dataset...")
    joblib.dump(disease_classifier, 'models/disease_classifier_trained_multilingual.pkl')
    
    disease_dist = pd.Series(y_disease).value_counts().to_dict()
    with open('models/disease_distribution_multilingual.json', 'w') as f:
        json.dump(disease_dist, f, indent=2)
    
    print("\n✓ Intent and disease training complete!")
    return intent_classifier

def main():
    """Main execution"""
    
    # Create directories if they don't exist
    Path("data/processed").mkdir(exist_ok=True, parents=True)
    Path("models").mkdir(exist_ok=True)
    
    # 1. Prepare combined dataset
    combined_df = prepare_combined_dataset()
    
    # 2. Save combined dataset
    combined_df.to_json("data/processed/combined_dataset.json", 
                       orient='records', force_ascii=False)
    print(f"\nSaved combined dataset to data/processed/combined_dataset.json")
    
    # 3. Create language-specific datasets
    language_datasets = create_language_specific_datasets(combined_df)
    
    # 4. Train models
    intent_classifier = train_models_on_combined(combined_df)
    
    # 5. Test with examples
    print("\n" + "=" * 60)
    print("Testing with Examples")
    print("=" * 60)
    
    test_examples = [
        ("I have fever and headache", "en"),
        ("मुझे बुखार और सरदर्द है", "hi"),
        ("আমার জ্বর এবং মাথা ব্যাথা আছে", "bn"),
        ("నాకు జ్వరం మరియు తలనొప్పి ఉంది", "te")
    ]
    
    for text, lang in test_examples:
        from app.nlp.pipeline import get_nlp_pipeline
        nlp = get_nlp_pipeline()
        nlp_out = nlp.process(text)
        
        # Use predict_single for single predictions (returns tuple)
        intent, confidence = intent_classifier.predict_single(nlp_out['embedding'].reshape(1, -1))
        print(f"\nText: {text}")
        print(f"Language: {lang} (detected: {nlp_out['lang']})")
        print(f"Predicted Intent: {intent} (confidence: {confidence:.2%})")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Train disease classifier: python train_disease_classifier.py")
    print("   (Symptom extractor is rule-based and doesn't need training)")
    print("2. Test the system: python test_system.py")
    print("3. See TRAINING_SCRIPTS.md for more details")
    print("=" * 60)

if __name__ == "__main__":
    main()