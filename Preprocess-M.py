# app/data/load_pubmedqa.py
import json
import pandas as pd
from typing import Dict, List, Any

def load_pubmedqa(file_path: str = "data/pubmedqa/ori_pqau.json") -> pd.DataFrame:
    """
    Load PubMedQA dataset from ori_pqau.json
    Structure: {question: {context: str, long_answer: str, ...}}
    """
    print(f"Loading PubMedQA from {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    processed_samples = []
    
    for question, details in data.items():
        if isinstance(details, dict):
            # Extract medical information
            context = details.get('context', '')
            long_answer = details.get('long_answer', '')
            final_decision = details.get('final_decision', '')
            
            # Process for HealthBot training
            processed = {
                'text': question,
                'context': context,
                'answer': long_answer,
                'language': 'en',
                'intent': classify_pubmed_intent(question, context),
                'symptoms': extract_symptoms_from_qa(question, long_answer),
                'diseases': extract_diseases_from_context(context),
                'is_emergency': check_emergency_context(context),
                'decision_type': final_decision,
                'source': 'pubmedqa_ori_pqau',
                'metadata': {
                    'has_context': bool(context),
                    'answer_length': len(long_answer)
                }
            }
            processed_samples.append(processed)
    
    print(f"Loaded {len(processed_samples)} samples from PubMedQA")
    return pd.DataFrame(processed_samples)

def classify_pubmed_intent(question: str, context: str) -> str:
    """Classify intent from PubMedQA questions"""
    question_lower = question.lower()
    
    # Intent mapping based on question patterns
    if any(word in question_lower for word in ['symptom', 'sign', 'present', 'manifest']):
        return 'symptom_reporting'
    elif any(word in question_lower for word in ['treatment', 'therapy', 'drug', 'medication']):
        return 'treatment_info'
    elif any(word in question_lower for word in ['cause', 'etiology', 'risk factor']):
        return 'cause_info'
    elif any(word in question_lower for word in ['diagnosis', 'test', 'detect']):
        return 'diagnosis_info'
    elif any(word in question_lower for word in ['prevent', 'prophylaxis', 'vaccine']):
        return 'prevention_guidance'
    else:
        return 'disease_info'

def extract_symptoms_from_qa(question: str, answer: str) -> List[str]:
    """Extract symptoms from question and answer text"""
    import re
    
    symptom_keywords = [
        'fever', 'headache', 'cough', 'cold', 'pain', 'rash', 'itching',
        'vomiting', 'diarrhea', 'weakness', 'dizziness', 'fatigue',
        'chest pain', 'stomach pain', 'joint pain', 'back pain',
        'breathing difficulty', 'shortness of breath', 'nausea',
        'swelling', 'bleeding', 'burning sensation'
    ]
    
    combined_text = (question + ' ' + answer).lower()
    found_symptoms = []
    
    for keyword in symptom_keywords:
        if keyword in combined_text:
            found_symptoms.append(keyword)
    
    return list(set(found_symptoms))  # Remove duplicates

def extract_diseases_from_context(context: str) -> List[str]:
    """Extract disease names from context text"""
    import re
    
    # Common disease patterns (can be enhanced with NER)
    disease_keywords = [
        'diabetes', 'hypertension', 'asthma', 'pneumonia', 'tuberculosis',
        'malaria', 'dengue', 'covid', 'flu', 'influenza', 'cancer',
        'heart disease', 'stroke', 'arthritis', 'psoriasis', 'eczema'
    ]
    
    context_lower = context.lower()
    found_diseases = []
    
    for keyword in disease_keywords:
        if keyword in context_lower:
            found_diseases.append(keyword)
    
    return list(set(found_diseases))  # Remove duplicates

def check_emergency_context(context: str) -> bool:
    """Check if context indicates emergency situation"""
    emergency_keywords = [
        'emergency', 'urgent', 'severe', 'critical', 'acute',
        'chest pain', 'breathing difficulty', 'unconscious',
        'bleeding', 'severe pain', 'heart attack', 'stroke'
    ]
    
    context_lower = context.lower()
    
    for keyword in emergency_keywords:
        if keyword in context_lower:
            return True
    
    return False