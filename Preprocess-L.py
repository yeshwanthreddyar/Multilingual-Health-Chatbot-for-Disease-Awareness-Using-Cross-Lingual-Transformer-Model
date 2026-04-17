# app/data/load_indicnlg.py
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

def load_indicnlg_dataset(base_path: str = "__MACOSX") -> Dict[str, pd.DataFrame]:
    """
    Load all IndicNLG dataset files with exact structure:
    - {lang}.dev.json
    - {lang}.test.json  
    - {lang}.train.json
    - ._{lang}.dev.json (ignore these hidden files)
    """
    languages = ['hi', 'bn', 'te', 'ta', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'as']
    
    datasets = {
        'train': [],
        'dev': [],
        'test': []
    }
    
    for lang in languages:
        print(f"\nProcessing {lang}...")
        
        # Load train, dev, test files (ignore hidden ._ files)
        for split in ['train', 'dev', 'test']:
            file_path = Path(base_path) / f"{lang}.{split}.json"
            hidden_file_path = Path(base_path) / f"._{lang}.{split}.json"
            
            if file_path.exists():
                print(f"  Loading {split}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
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
            if 'question' in item and 'answer' in item:
                # Q&A format
                converted.append({
                    'text': item['question'],
                    'answer': item['answer'],
                    'language': lang,
                    'intent': 'disease_info',
                    'symptoms': extract_symptoms_indic(item['question']),
                    'is_emergency': False,
                    'split': split,
                    'source': f'indicnlg_{lang}'
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