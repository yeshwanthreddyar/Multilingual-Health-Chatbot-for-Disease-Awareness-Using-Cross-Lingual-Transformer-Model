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
    
    for item_id, details in data.items():
        if isinstance(details, dict):
            # Extract medical information - handle both uppercase and lowercase keys
            question = details.get('QUESTION', details.get('question', ''))
            context = details.get('CONTEXTS', details.get('contexts', details.get('context', '')))
            long_answer = details.get('LONG_ANSWER', details.get('long_answer', details.get('FINAL_ANSWER', details.get('final_answer', ''))))
            final_decision = details.get('FINAL_DECISION', details.get('final_decision', ''))
            
            # Handle CONTEXTS as list - take first context if it's a list
            if isinstance(context, list) and len(context) > 0:
                context = context[0] if isinstance(context[0], str) else str(context[0])
            elif not isinstance(context, str):
                context = str(context) if context else ''
            
            # Skip if no question text
            if not question or len(str(question).strip()) == 0:
                continue
            
            # Process for HealthBot training
            processed = {
                'text': str(question),
                'context': str(context),
                'answer': str(long_answer),
                'language': 'en',
                'intent': classify_pubmed_intent(str(question), str(context)),
                'symptoms': extract_symptoms_from_qa(str(question), str(long_answer)),
                'diseases': extract_diseases_from_context(str(context)),
                'is_emergency': check_emergency_context(str(context)),
                'decision_type': str(final_decision),
                'source': 'pubmedqa_ori_pqau',
                'metadata': {
                    'has_context': bool(context),
                    'answer_length': len(str(long_answer))
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