"""
Symptom Extraction - NER-based.
Map colloquial phrases → ICD-10 / SNOMED-style labels.
Multilingual symptom lexicons.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# Colloquial → canonical (ICD-10/SNOMED-style) symptom mapping (English + transliterations)
SYMPTOM_LEXICON: Dict[str, str] = {
    # English
    "fever": "R50",
    "headache": "R51",
    "cough": "R05",
    "cold": "J00",
    "body pain": "R52",
    "stomach pain": "R10",
    "diarrhea": "R19",
    "vomiting": "R11",
    "weakness": "R53",
    "breathing difficulty": "R06",
    "chest pain": "R07",
    "rash": "R21",
    "itching": "L29",
    "sore throat": "R07.0",
    "runny nose": "R09.8",
    "loss of appetite": "R63",
    "dizziness": "R42",
    "fatigue": "R53",
    "joint pain": "M25.5",
    "back pain": "M54",
    "nausea": "R11",
    "swelling": "R22",
    "bleeding": "R58",
    "burning sensation": "R20",
    "red eyes": "H10",
    "ear pain": "H92",
    "toothache": "K08.8",
    # Common Indian language equivalents (transliterated / keywords)
    # Hindi / Hinglish - many spelling variants people actually type
    "bukhār": "R50",
    "bukhar": "R50",
    "bukar": "R50",
    "buukhar": "R50",
    "bhukar": "R50",
    "jwar": "R50",
    "jwara": "R50",
    "tap": "R50",
    "taap": "R50",
    "sardī": "J00",
    "sardi": "J00",
    "zukam": "J00",
    "jukam": "J00",
    "jukaam": "J00",
    "nazla": "J00",
    "khāsī": "R05",
    "khansi": "R05",
    "khaansi": "R05",
    "khaasi": "R05",
    "pet dard": "R10",
    "pet me dard": "R10",
    "pet mein dard": "R10",
    "pait dard": "R10",
    "dast": "R19",
    "loose motion": "R19",
    "loose motions": "R19",
    "ultī": "R11",
    "ulti": "R11",
    "vomit": "R11",
    "uski": "R11",
    "kamzorī": "R53",
    "kamzori": "R53",
    "kamjori": "R53",
    "thakaan": "R53",
    "thakan": "R53",
    "sir dard": "R51",
    "sar dard": "R51",
    "sir me dard": "R51",
    "sardard": "R51",
    "saans phoolna": "R06",
    "saans phulna": "R06",
    "saans lene me takleef": "R06",
    "chati me dard": "R07",
    "chaati me dard": "R07",
    "seene me dard": "R07",
    "dard": "R52",
    "takleef": "R52",
    "pareshani": "R52",
    # Hindi / Marathi / Nepali / Konkani (Devanagari)
    "बुखार": "R50", "ज्वर": "R50", "तेज बुखार": "R50", "ताप": "R50", "ज्वरो": "R50",
    "खांसी": "R05", "खोकला": "R05", "खोकी": "R05",
    "सर्दी": "J00", "जुकाम": "J00", "नाक बहना": "J00", "सर्दी खांसी": "J00",
    "पेट दर्द": "R10", "पोटदुखी": "R10", "पेट दुख": "R10",
    "उल्टी": "R11", "उलटी": "R11", "वमन": "R11",
    "दस्त": "R19", "जुलाब": "R19", "पतले दस्त": "R19",
    "थकान": "R53", "कमजोरी": "R53", "थकावट": "R53",
    "सिर दर्द": "R51", "सर दर्द": "R51",
    "सांस फूलना": "R06", "श्वास त्रास": "R06",
    "छाती में दर्द": "R07", "छातीत दुखणे": "R07",
    "शरीर दर्द": "R52", "अंग दर्द": "R52",
    "चक्कर": "R42", "घाम": "R21",

    # Bengali
    "জ্বর": "R50", "তীব্র জ্বর": "R50",
    "কাশি": "R05", "সর্দি": "J00", "নাক দিয়ে জল": "J00",
    "পেট ব্যথা": "R10", "বমি": "R11", "ডায়রিয়া": "R19",
    "দুর্বলতা": "R53", "মাথা ব্যথা": "R51",
    "শ্বাসকষ্ট": "R06", "বুকে ব্যথা": "R07",
    "গা ব্যথা": "R52", "মাথা ঘোরা": "R42",

    # Assamese
    "জ্বৰ": "R50", "কাহ": "R05", "পেট বিষ": "R10",
    "বমি": "R11", "পাতল পায়খানা": "R19", "দুৰ্বলতা": "R53",
    "মূৰ বিষ": "R51", "উশাহ লোৱাত কষ্ট": "R06",

    # Telugu
    "జ్వరం": "R50", "దగ్గు": "R05", "జలుబు": "J00",
    "కడుపు నొప్పి": "R10", "వాంతి": "R11", "విరేచనాలు": "R19",
    "అలసట": "R53", "తలనొప్పి": "R51",
    "శ్వాస తీసుకోవడం కష్టం": "R06", "ఛాతి నొప్పి": "R07",
    "ఒళ్ళు నొప్పి": "R52", "తల తిరగడం": "R42",

    # Tamil
    "காய்ச்சல்": "R50", "இருமல்": "R05", "சளி": "J00",
    "வயிற்று வலி": "R10", "வாந்தி": "R11", "வயிற்றுப்போக்கு": "R19",
    "சோர்வு": "R53", "தலைவலி": "R51",
    "மூச்சுத் திணறல்": "R06", "மார்பு வலி": "R07",
    "உடல் வலி": "R52", "தலைசுற்றல்": "R42",

    # Kannada
    "ಜ್ವರ": "R50", "ಕೆಮ್ಮು": "R05", "ಶೀತ": "J00",
    "ಹೊಟ್ಟೆ ನೋವು": "R10", "ವಾಂತಿ": "R11", "ಭೇದಿ": "R19",
    "ದಣಿವು": "R53", "ತಲೆನೋವು": "R51",
    "ಉಸಿರಾಟದ ತೊಂದರೆ": "R06", "ಎದೆ ನೋವು": "R07",
    "ಮೈ ನೋವು": "R52", "ತಲೆ ತಿರುಗುವಿಕೆ": "R42",

    # Malayalam
    "പനി": "R50", "ചുമ": "R05", "ജലദോഷം": "J00",
    "വയറുവേദന": "R10", "ഛർദി": "R11", "വയറിളക്കം": "R19",
    "ക്ഷീണം": "R53", "തലവേദന": "R51",
    "ശ്വാസതടസ്സം": "R06", "നെഞ്ചുവേദന": "R07",
    "ദേഹവേദന": "R52", "തലകറക്കം": "R42",

    # Gujarati
    "તાવ": "R50", "ઉધરસ": "R05", "શરદી": "J00",
    "પેટ દુખાવો": "R10", "ઊલટી": "R11", "ઝાડા": "R19",
    "થાક": "R53", "માથાનો દુખાવો": "R51",
    "શ્વાસ લેવામાં તકલીફ": "R06", "છાતીમાં દુખાવો": "R07",
    "અંગ દુખાવો": "R52", "ચક્કર": "R42",

    # Punjabi (Gurmukhi)
    "ਬੁਖਾਰ": "R50", "ਖੰਘ": "R05", "ਜ਼ੁਕਾਮ": "J00",
    "ਪੇਟ ਦਰਦ": "R10", "ਉਲਟੀ": "R11", "ਦਸਤ": "R19",
    "ਕਮਜ਼ੋਰੀ": "R53", "ਸਿਰ ਦਰਦ": "R51",
    "ਸਾਹ ਲੈਣ ਵਿੱਚ ਤਕਲੀਫ਼": "R06", "ਛਾਤੀ ਦਰਦ": "R07",
    "ਸਰੀਰ ਦਰਦ": "R52",

    # Odia
    "ଜ୍ୱର": "R50", "କାଶ": "R05", "ସର୍ଦି": "J00",
    "ପେଟ ଯନ୍ତ୍ରଣା": "R10", "ବାନ୍ତି": "R11", "ତରଳ ଝାଡ଼ା": "R19",
    "ଦୁର୍ବଳତା": "R53", "ମୁଣ୍ଡ ବ୍ୟଥା": "R51",
    "ଶ୍ୱାସ ନେବାରେ କଷ୍ଟ": "R06", "ଛାତି ଯନ୍ତ୍ରଣା": "R07",

    # Urdu
    "بخار": "R50", "کھانسی": "R05", "نزلہ": "J00",
    "پیٹ درد": "R10", "الٹی": "R11", "دست": "R19",
    "کمزوری": "R53", "سر درد": "R51",
    "سانس لینے میں تکلیف": "R06", "سینے میں درد": "R07",
    "جسم میں درد": "R52", "چکر": "R42",
}

# Emergency symptom codes → escalate to hospital
EMERGENCY_SYMPTOMS: Set[str] = {
    "R06",   # breathing difficulty
    "R07",   # chest pain
    "R58",   # bleeding
    "R10.0", # acute abdomen
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_symptoms(text: str, tokens: List[str]) -> List[Tuple[str, str]]:
    """
    NER-style extraction: match against lexicon, return list of (canonical_code, raw_phrase).
    """
    normalized = _normalize(text)
    found: List[Tuple[str, str]] = []
    seen_codes: Set[str] = set()

    # Phrase-level matches first
    for phrase, code in sorted(SYMPTOM_LEXICON.items(), key=lambda x: -len(x[0])):
        if phrase in normalized and code not in seen_codes:
            found.append((code, phrase))
            seen_codes.add(code)

    # Token-level
    for tok in tokens:
        t = tok.lower()
        if t in SYMPTOM_LEXICON:
            code = SYMPTOM_LEXICON[t]
            if code not in seen_codes:
                found.append((code, t))
                seen_codes.add(code)

    return found


def is_emergency_symptom(codes: List[str]) -> bool:
    """Check if any extracted symptom is emergency (hospital recommendation)."""
    for c in codes:
        if c in EMERGENCY_SYMPTOMS or any(c.startswith(e) for e in EMERGENCY_SYMPTOMS):
            return True
    return False