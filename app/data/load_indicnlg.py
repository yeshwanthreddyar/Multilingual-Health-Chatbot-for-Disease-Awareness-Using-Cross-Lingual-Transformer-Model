# app/data/load_indicnlg.py
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

def load_indicnlg_dataset(base_path: str = "__MACOSX") -> Dict[str, pd.DataFrame]:
    """
    Load IndicNLG dataset files. Supports two formats:
    1. Language-specific: {lang}.{split}.json (e.g., hi.train.json)
    2. Generic: {split}.json (e.g., train.json) - treats as English medical Q&A
    """
    languages = ['hi', 'bn', 'te', 'ta', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'as']
    
    datasets = {
        'train': [],
        'dev': [],
        'test': []
    }
    
    # First, try to load generic format files (train.json, dev.json, test.json)
    generic_files_found = False
    for split in ['train', 'dev', 'test']:
        generic_file_path = Path(base_path) / f"{split}.json"
        if generic_file_path.exists():
            generic_files_found = True
            print(f"\nLoading generic {split}.json file...")
            data = []
            try:
                with open(generic_file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                if line_num <= 5:  # Only print first few errors
                                    print(f"  Warning: JSON decode error on line {line_num}: {e}")
                                continue
                
                print(f"  Loaded {len(data)} records from {split}.json")
                
                # Convert to HealthBot format - treat as English medical Q&A
                converted_data = convert_indicnlg_format(data, 'en', split)
                datasets[split].extend(converted_data)
                print(f"  Converted to {len(converted_data)} samples")
            except Exception as e:
                print(f"  Error loading {split}.json: {e}")
    
    # If generic files found, skip language-specific loading
    if generic_files_found:
        print("\nUsing generic format files (train.json, dev.json, test.json)")
    else:
        # Try language-specific format
        print("\nTrying language-specific format...")
        for lang in languages:
            print(f"\nProcessing {lang}...")
            
            # Load train, dev, test files (ignore hidden ._ files)
            for split in ['train', 'dev', 'test']:
                file_path = Path(base_path) / f"{lang}.{split}.json"
                hidden_file_path = Path(base_path) / f"._{lang}.{split}.json"
                
                if file_path.exists():
                    print(f"  Loading {split}...")
                    # Handle JSONL format (one JSON object per line)
                    data = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    data.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                    
                    # Convert to HealthBot format based on dataset type
                    converted_data = convert_indicnlg_format(data, lang, split)
                    datasets[split].extend(converted_data)
                
                # Optional: Log if hidden file exists (usually macOS artifacts)
                if hidden_file_path.exists():
                    print(f"  Note: Found hidden file {hidden_file_path.name} (likely macOS artifact)")
    
    # Convert to DataFrames
    result = {}
    for split, data_list in datasets.items():
        if data_list:
            result[split] = pd.DataFrame(data_list)
            print(f"{split}: {len(data_list)} samples")
    
    return result

def convert_indicnlg_format(data: List[Dict], lang: str, split: str) -> List[Dict]:
    """Convert IndicNLG format to HealthBot training format"""
    converted = []
    
    for item in data:
        # Handle different IndicNLG dataset structures
        if isinstance(item, dict):
            # Check structure type
            if 'question' in item:
                # Q&A format - use 'exp' as answer if available
                question_text = str(item['question'])
                answer = str(item.get('exp', item.get('answer', '')))
                
                # Classify intent based on question content
                intent = classify_medical_question_intent(question_text)
                
                converted.append({
                    'text': question_text,
                    'answer': answer,
                    'language': lang,
                    'intent': intent,
                    'symptoms': extract_symptoms_indic(question_text),
                    'is_emergency': False,
                    'split': split,
                    'source': f'indicnlg_{lang}',
                    'subject': item.get('subject_name', ''),
                    'topic': item.get('topic_name', '')
                })
            elif 'text1' in item and 'text2' in item:
                # Paraphrase format - use as augmentation
                converted.append({
                    'text': item['text1'],
                    'paraphrase': item['text2'],
                    'language': lang,
                    'intent': 'general_health_query',
                    'split': split,
                    'source': f'indicnlg_paraphrase_{lang}'
                })
            elif 'text' in item and 'label' in item:
                # Sentiment/classification format
                converted.append({
                    'text': item['text'],
                    'label': item['label'],
                    'language': lang,
                    'intent': map_label_to_intent(item['label']),
                    'split': split,
                    'source': f'indicnlg_{split}_{lang}'
                })
        elif isinstance(item, str):
            # Simple text format
            converted.append({
                'text': item,
                'language': lang,
                'intent': 'general_health_query',
                'split': split,
                'source': f'indicnlg_text_{lang}'
            })
    
    return converted

def extract_symptoms_indic(text: str) -> List[str]:
    """Extract symptoms from Indic language text"""
    # Simple keyword-based extraction (can be enhanced with NER)
    symptom_keywords = [
        'fever', 'headache', 'cough', 'cold', 'pain', 'rash', 'itching',
        'vomiting', 'diarrhea', 'weakness', 'dizziness', 'fatigue',
        'bukhār', 'sardī', 'khāsī', 'pet dard', 'dast', 'ultī', 'kamzorī', 'sir dard'
    ]
    
    text_lower = text.lower()
    found_symptoms = []
    
    for keyword in symptom_keywords:
        if keyword in text_lower:
            found_symptoms.append(keyword)
    
    return found_symptoms

def map_label_to_intent(label: str) -> str:
    """Map dataset labels to HealthBot intents"""
    label_lower = label.lower()
    
    if 'symptom' in label_lower or 'report' in label_lower:
        return 'symptom_reporting'
    elif 'disease' in label_lower or 'condition' in label_lower:
        return 'disease_info'
    elif 'treatment' in label_lower or 'therapy' in label_lower:
        return 'treatment_info'
    elif 'prevent' in label_lower or 'prevention' in label_lower:
        return 'prevention_guidance'
    else:
        return 'general_health_query'

def classify_medical_question_intent(question: str) -> str:
    """Classify intent from medical exam questions"""
    question_lower = question.lower()
    
    # Intent mapping based on question patterns
    if any(word in question_lower for word in ['symptom', 'sign', 'present', 'manifest', 'complaint']):
        return 'symptom_reporting'
    elif any(word in question_lower for word in ['treatment', 'therapy', 'drug', 'medication', 'prescribe']):
        return 'treatment_info'
    elif any(word in question_lower for word in ['cause', 'etiology', 'risk factor', 'due to', 'leads to']):
        return 'cause_info'
    elif any(word in question_lower for word in ['diagnosis', 'test', 'detect', 'diagnose', 'investigation']):
        return 'diagnosis_info'
    elif any(word in question_lower for word in ['prevent', 'prophylaxis', 'vaccine', 'prevention']):
        return 'prevention_guidance'
    elif any(word in question_lower for word in ['anatomy', 'structure', 'location', 'organ']):
        return 'disease_info'  # Anatomy questions
    elif any(word in question_lower for word in ['physiology', 'function', 'mechanism', 'how does']):
        return 'disease_info'  # Physiology questions
    else:
        return 'disease_info'  # Default for medical questions