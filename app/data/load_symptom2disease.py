# app/data/load_symptom2disease.py
import pandas as pd
from typing import Dict, List
import re

def load_symptom2disease(file_path: str = "Symptom2Disease.csv") -> pd.DataFrame:
    """
    Load Symptom2Disease.csv dataset
    Structure: label (disease name), text (symptom description)
    """
    print(f"Loading Symptom2Disease from {file_path}...")
    
    df = pd.read_csv(file_path)
    
    processed_samples = []
    
    for _, row in df.iterrows():
        text = str(row['text'])
        label = str(row['label'])
        
        # Extract symptoms from text
        symptoms = extract_symptoms_from_text(text)
        
        # Process for HealthBot training
        processed = {
            'text': text,
            'disease': label,
            'language': 'en',
            'intent': 'symptom_reporting',
            'symptoms': symptoms,
            'is_emergency': check_emergency_symptoms(symptoms),
            'source': 'symptom2disease',
            'split': 'train'  # Will be split later
        }
        processed_samples.append(processed)
    
    print(f"Loaded {len(processed_samples)} samples from Symptom2Disease")
    return pd.DataFrame(processed_samples)

def extract_symptoms_from_text(text: str) -> List[str]:
    """Extract symptoms from symptom description text"""
    symptom_keywords = [
        'fever', 'headache', 'cough', 'cold', 'pain', 'rash', 'itching',
        'vomiting', 'diarrhea', 'weakness', 'dizziness', 'fatigue',
        'chest pain', 'stomach pain', 'joint pain', 'back pain',
        'breathing difficulty', 'shortness of breath', 'nausea',
        'swelling', 'bleeding', 'burning sensation', 'peeling',
        'inflammation', 'red', 'itchy', 'scaly', 'dry', 'cracked',
        'tender', 'sensitive', 'discomfort', 'ache', 'throbbing'
    ]
    
    text_lower = text.lower()
    found_symptoms = []
    
    for keyword in symptom_keywords:
        if keyword in text_lower:
            found_symptoms.append(keyword)
    
    return list(set(found_symptoms))  # Remove duplicates

def check_emergency_symptoms(symptoms: List[str]) -> bool:
    """Check if symptoms indicate emergency"""
    emergency_keywords = [
        'chest pain', 'breathing difficulty', 'shortness of breath',
        'severe pain', 'bleeding', 'unconscious', 'severe'
    ]
    
    symptoms_str = ' '.join(symptoms).lower()
    
    for keyword in emergency_keywords:
        if keyword in symptoms_str:
            return True
    
    return False
