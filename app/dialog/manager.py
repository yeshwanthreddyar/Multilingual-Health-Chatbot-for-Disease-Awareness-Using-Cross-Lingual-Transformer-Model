"""
Context-Aware Dialogue Manager.
Multi-turn memory, carry forward symptoms, decide next action:
ask follow-up, give prevention info, emergency escalation.
Ollama used ONLY for: response phrasing, context summarization.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.llm.ollama_client import call_ollama
from app.knowledge_base.retriever import retrieve_disease_advisory, retrieve_for_symptoms
from app.integrations.location_service import (
    find_nearby_hospitals,
    extract_location_from_message,
    format_hospital_response,
)
from app.config import DISCLAIMER


def _normalize_region_from_location(user_location: Optional[str]) -> Optional[str]:
    """
    Best-effort region extraction from a user-provided location string.

    - If coordinates like "lat,lon", return None (no human-readable region).
    - If "City, State" or "City", return the leading part as region.
    """
    if not user_location:
        return None
    loc = user_location.strip()
    if not loc:
        return None
    # Coordinate-like: first two comma-separated parts are numeric → skip
    if "," in loc:
        parts = [p.strip() for p in loc.split(",")]
        head = parts[0] if parts else ""
        try:
            if len(parts) >= 2:
                float(parts[0])
                float(parts[1])
                return None
        except ValueError:
            return head or None
        return None
    return loc


_LANG_LABELS: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "ne": "Nepali",
    "kok": "Konkani",
}


def _language_instruction(lang: str) -> str:
    """
    Short instruction for Ollama to answer in the same language as the user.
    """
    label = _LANG_LABELS.get(lang, "the user's language")
    return f"Respond in {label} (same language as the user's message)."


def _get_region_alert_snippets(
    region: Optional[str],
    top3_diseases: List[tuple],
    limit: int = 2,
) -> List[str]:
    """
    Retrieve short outbreak / public-health alert snippets for a region and (optionally)
    the ML-predicted diseases. Uses the same underlying sources as the /alerts endpoints.
    """
    try:
        from app.integrations.government_mock import get_alerts
        from app.integrations.alert_sender import region_matches
        from app.outbreak.detector import merge_with_existing_alerts
    except Exception:
        return []

    try:
        base_alerts = get_alerts(limit=limit * 3)
        all_alerts = merge_with_existing_alerts(base_alerts, limit=limit * 3)
    except Exception:
        return []

    # Filter by region if we have one
    if region:
        try:
            filtered = [
                a for a in all_alerts
                if region_matches(a.get("region") or "", region)
            ]
            alerts = filtered or all_alerts
        except Exception:
            alerts = all_alerts
    else:
        alerts = all_alerts

    disease_keys = {d for d, _ in (top3_diseases or [])}
    snippets: List[str] = []
    for a in alerts:
        disease_key = a.get("disease")
        if disease_keys and disease_key and disease_key not in disease_keys:
            continue
        src = a.get("source") or "Health authority"
        title = a.get("title") or ""
        summary = a.get("summary") or ""
        region_label = a.get("region") or "National"
        text = f"Alert ({src}, {region_label}): {title}. {summary}".strip()
        if text and text not in snippets:
            snippets.append(text)
        if len(snippets) >= limit:
            break
    return snippets


def _is_advice_seeking(text: str) -> bool:
    """True if the message is asking what to do / for advice."""
    t = (text or "").lower().strip()
    return any(
        phrase in t
        for phrase in (
            "what should i", "what to do", "what can i do", "what do i do",
            "how to", "should i", "what do you recommend", "advice", "suggest",
            "i have ", "i am having ", "i feel ", "suffering from", "experiencing ",
        )
    )


def _is_valid_answer(answer: str) -> bool:
    """False if Ollama failed or the model clearly refused to answer."""
    if not answer or "[Ollama" in answer:
        return False
    a = answer.strip()
    if len(a) < 10:
        return False
    # Only treat as refusal if the start of the answer looks like a refusal
    a_lower = a.lower()[:200]
    refusal = any(
        x in a_lower
        for x in (
            "i can't answer", "i cannot answer", "i'm not able to answer",
            "i do not answer", "cannot answer that", "can't answer that",
            "unable to answer", "not able to provide", "refuse to answer",
        )
    )
    return not refusal


def _is_student_query(text: Optional[str]) -> bool:
    """
    Heuristic: True if the message looks like it is from a medical / nursing student
    asking for deeper academic or exam-style explanation.
    """
    if not text:
        return False
    t = text.lower()
    return any(
        phrase in t
        for phrase in (
            "pathophysiology",
            "pathogenesis",
            "mechanism of action",
            "moa",
            "classification of",
            "differential diagnosis",
            "difference between",
            "compare",
            "contrast",
            "mbbs",
            "for my exam",
            "for exam",
            "final year",
            "medical student",
            "nursing student",
        )
    )


def _build_rule_based_symptom_explanation(user_message: Optional[str]) -> str:
    """
    Analyse the free-text user message for common symptom patterns and return
    a safe, human-readable description of *possible* conditions.

    This does NOT give a confirmed diagnosis. It only explains that symptoms
    can be seen in certain illnesses.
    """
    if not user_message:
        return ""
    t = user_message.lower()

    # English keywords
    has_fever = "fever" in t or "temperature" in t
    has_cough = "cough" in t
    has_cold = "cold" in t or "runny nose" in t or "running nose" in t
    has_sore_throat = "sore throat" in t or "throat pain" in t
    has_body_pain = "body pain" in t or "body ache" in t or "joint pain" in t or "muscle pain" in t
    has_headache = "headache" in t or "head pain" in t
    has_behind_eyes = "behind my eyes" in t or "behind the eyes" in t
    has_rash = "rash" in t
    has_chills = "chills" in t or "shivering" in t or "rigor" in t
    has_vomit = "vomit" in t or "vomiting" in t or "nausea" in t
    has_loose_stools = (
        "loose motion" in t
        or "loose motions" in t
        or "diarrhoea" in t
        or "diarrhea" in t
        or "loose stool" in t
    )
    has_abd_pain = "stomach pain" in t or "abdominal pain" in t or "tummy pain" in t
    has_burning_urine = (
        "burning urine" in t
        or "burning while passing urine" in t
        or "pain while passing urine" in t
        or "burning while urinating" in t
    )
    has_urine_freq = "frequent urination" in t or "passing urine again and again" in t
    has_breath = (
        "shortness of breath" in t
        or "breathless" in t
        or "difficulty breathing" in t
        or "breathing difficulty" in t
    )
    has_chest_pain = "chest pain" in t or "pain in chest" in t
    has_travel = "travel" in t or "village" in t or "forest" in t
    has_mosquito = "mosquito" in t or "mosquitoes" in t or "stagnant water" in t

    # Hindi symptom keywords (simple substring checks)
    # fever: bukhar / bukar / buukhar / bhukar / jwar / tap / taap
    if ("बुखार" in t) or ("bukhar" in t) or ("bukar" in t) or ("buukhar" in t) or ("bhukar" in t) or ("jwar" in t) or ("taap" in t) or ("tap" in t and "fever" not in t):
        has_fever = True
    # cough: "khansi", "खांसी"
    if ("खांसी" in t) or ("खँसी" in t) or ("khansi" in t):
        has_cough = True
    # cold / runny nose: "sardi", "जुकाम", "नाक बह"
    if ("सर्दी" in t) or ("जुकाम" in t) or ("जुकाम" in t) or ("sardi" in t) or ("zukam" in t) or ("नाक बह" in t):
        has_cold = True
    # sore throat: "gale me dard", "gala dard", "गले में दर्द", "गला दर्द"
    if ("गले में दर्द" in t) or ("गला दर्द" in t) or ("gale me dard" in t) or ("gala dard" in t):
        has_sore_throat = True
    # body pain: "sarir dard", "shareer dard", "शरीर में दर्द", "शरीर दर्द", "jodo me dard"
    if (
        ("शरीर में दर्द" in t)
        or ("शरीर दर्द" in t)
        or ("jodo me dard" in t)
        or ("jodon me dard" in t)
        or ("shareer dard" in t)
        or ("sarir dard" in t)
    ):
        has_body_pain = True
    # headache: "sar dard", "सर दर्द", "सिर दर्द"
    if ("सर दर्द" in t) or ("सिर दर्द" in t) or ("sar dard" in t) or ("sir dard" in t):
        has_headache = True
    # chills / shivering: "kaap", "kapkapi", "sardi lagna", "कांपना", "कंपकंपी"
    if ("कांप" in t) or ("कंपकंपी" in t) or ("kapkapi" in t) or ("sardi lag" in t):
        has_chills = True
    # loose motions / diarrhoea: "dast", "पतले दस्त", "पेट खराब", "loose motion" in Hindi
    if ("दस्त" in t) or ("पतले दस्त" in t) or ("पेट खराब" in t) or ("pet kharab" in t):
        has_loose_stools = True
    # vomiting / nausea: "ulti", "उल्टी", "man ghabrana", "मतली"
    if ("उल्टी" in t) or ("ulti" in t) or ("मतली" in t) or ("man ghabra" in t):
        has_vomit = True
    # stomach pain: "pet dard", "पेट दर्द", "पेट में दर्द"
    if ("पेट दर्द" in t) or ("पेट में दर्द" in t) or ("pet dard" in t):
        has_abd_pain = True
    # burning urine: "peshab me jalan", " पेशाब में जलन"
    if ("पेशाब में जलन" in t) or ("peshab me jalan" in t) or ("peshab mein jalan" in t):
        has_burning_urine = True
    # frequent urination: "bar bar peshab", "बार बार पेशाब"
    if ("बार बार पेशाब" in t) or ("bar bar peshab" in t):
        has_urine_freq = True
    # breathlessness: "saans phoolna", "saas phoolna", "सांस फूलना", "saans lene me takleef"
    if ("सांस फूल" in t) or ("saans phool" in t) or ("saas phool" in t) or ("saans lene me takleef" in t):
        has_breath = True
    # chest pain: "chati me dard", "छाती में दर्द"
    if ("छाती में दर्द" in t) or ("chati me dard" in t) or ("chaati me dard" in t):
        has_chest_pain = True
    # mosquito / stagnant water: "machchar", "मच्छर", "पानी जमा", "keede", etc.
    if ("मच्छर" in t) or ("machchar" in t) or ("पानी जमा" in t):
        has_mosquito = True

    # Bengali
    if "জ্বর" in t:
        has_fever = True
    if "কাশি" in t:
        has_cough = True
    if ("ঠান্ডা" in t) or ("সর্দি" in t):
        has_cold = True
    if ("পেট ব্যথা" in t) or ("পেট ব্যথা" in t):
        has_abd_pain = True
    if ("পাতলা পায়খানা" in t) or ("ডায়রিয়া" in t) or ("ডায়রিয়া" in t):
        has_loose_stools = True
    if "বমি" in t:
        has_vomit = True
    if ("শ্বাসকষ্ট" in t) or ("নিশ্বাস" in t):
        has_breath = True
    if "বুকের ব্যথা" in t:
        has_chest_pain = True
    if "মশা" in t:
        has_mosquito = True

    # Telugu
    if "జ్వరం" in t:
        has_fever = True
    if "దగ్గు" in t:
        has_cough = True
    if "జలుబు" in t:
        has_cold = True
    if ("డయ్యేరియా" in t) or ("అతిసారం" in t):
        has_loose_stools = True
    if ("వాంతి" in t) or ("వాంతులు" in t):
        has_vomit = True
    if "కడుపు నొప్పి" in t:
        has_abd_pain = True
    if ("ఉబ్బసం" in t) or ("శ్వాస" in t and "కష్టం" in t):
        has_breath = True
    if "ఛాతి నొప్పి" in t:
        has_chest_pain = True
    if "దోమ" in t:
        has_mosquito = True

    # Tamil
    if "காய்ச்சல்" in t:
        has_fever = True
    if "இருமல்" in t:
        has_cough = True
    if "சளி" in t:
        has_cold = True
    if "வயிற்றுப்போக்கு" in t:
        has_loose_stools = True
    if "வாந்தி" in t:
        has_vomit = True
    if "வயிற்று வலி" in t:
        has_abd_pain = True
    if ("மூச்சுத் திணறல்" in t) or ("சுவாச" in t and "பிரச்சனை" in t):
        has_breath = True
    if "மார்பு வலி" in t:
        has_chest_pain = True
    if "கொசு" in t:
        has_mosquito = True

    # Kannada
    if "ಜ್ವರ" in t:
        has_fever = True
    if "ಕೆಮ್ಮು" in t:
        has_cough = True
    if "ಜಲದೋಷ" in t:
        has_cold = True
    if ("ಅತಿಸಾರ" in t) or ("ಡಯೇರಿಯಾ" in t):
        has_loose_stools = True
    if ("ಓಕು" in t) or ("ಓಕಳಿಕೆ" in t):
        has_vomit = True
    if "ಹೊಟ್ಟೆ ನೋವು" in t:
        has_abd_pain = True
    if ("ಉಸಿರಾಟದ ತೊಂದರೆ" in t) or ("ಉಸಿರು" in t and "ಕಷ್ಟ" in t):
        has_breath = True
    if "ಎದೆಯ ನೋವು" in t:
        has_chest_pain = True
    if "ಸೊಳ್ಳೆ" in t:
        has_mosquito = True

    # Malayalam
    if "പനി" in t:
        has_fever = True
    if "ചുമ" in t:
        has_cough = True
    if "തണുപ്പ്" in t:
        has_cold = True
    if ("വയറിളക്കം" in t) or ("ഛർദി" in t and "വയറിളക്കം" in t):
        has_loose_stools = True
    if ("വാന്തി" in t) or ("ഛര്‍ദി" in t):
        has_vomit = True
    if "വയറ്റുവേദന" in t:
        has_abd_pain = True
    if ("ശ്വാസംമുട്ടല്" in t) or ("ശ്വാസം മുട്ടല്" in t):
        has_breath = True
    if "വക്ഷോവേദന" in t:
        has_chest_pain = True
    if "കൊതുക്" in t:
        has_mosquito = True

    # Marathi / Konkani (Devanagari)
    if "ताप" in t:
        has_fever = True
    if "खोकला" in t:
        has_cough = True
    if "सर्दी" in t:
        has_cold = True
    if "जुलाब" in t:
        has_loose_stools = True
    if ("उलटी" in t) or ("ओकाऱ्या" in t):
        has_vomit = True
    if "पोटदुखी" in t or "पोट दुख" in t:
        has_abd_pain = True
    if ("श्वास" in t and "त्रास" in t) or ("श्वास घेण्यास त्रास" in t):
        has_breath = True
    if "छातीत दुख" in t or "छाती दुख" in t:
        has_chest_pain = True
    if "डास" in t:
        has_mosquito = True

    # Gujarati
    if "તાવ" in t:
        has_fever = True
    if "ખાંસી" in t or "ઉધરસ" in t:
        has_cough = True
    if "સર્દી" in t:
        has_cold = True
    if "અતિસાર" in t:
        has_loose_stools = True
    if "ઉલટી" in t:
        has_vomit = True
    if "પેટમાં દુખાવો" in t:
        has_abd_pain = True
    if "શ્વાસ લેવામાં તકલીફ" in t:
        has_breath = True
    if "છાતીમાં દુખાવો" in t:
        has_chest_pain = True
    if "મચ્છર" in t:
        has_mosquito = True

    # Punjabi (Gurmukhi)
    if "ਬੁਖਾਰ" in t:
        has_fever = True
    if "ਖੰਘ" in t:
        has_cough = True
    if "ਜ਼ੁਕਾਮ" in t:
        has_cold = True
    if "ਦਸਤ" in t:
        has_loose_stools = True
    if "ਉਲਟੀ" in t:
        has_vomit = True
    if "ਪੇਟ ਦਰਦ" in t:
        has_abd_pain = True
    if "ਸਾਹ ਫੁੱਲ" in t:
        has_breath = True
    if "ਛਾਤੀ ਦਰਦ" in t:
        has_chest_pain = True
    if "ਮੱਛਰ" in t:
        has_mosquito = True

    # Odia
    if "ଜ୍ୱର" in t:
        has_fever = True
    if "କାଶ" in t:
        has_cough = True
    if "ସର୍ଦି" in t:
        has_cold = True
    if "ଦସ୍ତ" in t:
        has_loose_stools = True
    if "ବାନ୍ତି" in t:
        has_vomit = True
    if "ପେଟ ଯନ୍ତ୍ରଣା" in t or "ପେଟ ବେଦନା" in t:
        has_abd_pain = True

    # Assamese
    if "জ্বৰ" in t:
        has_fever = True
    if "খোকলি" in t:
        has_cough = True
    if "ঠান্ডা" in t:
        has_cold = True
    if "ডায়েৰিয়া" in t or "ডায়রিয়া" in t:
        has_loose_stools = True
    if "বমি" in t:
        has_vomit = True
    if "পেট ব্যথা" in t:
        has_abd_pain = True

    # Urdu
    if "بخار" in t:
        has_fever = True
    if "کھانسی" in t:
        has_cough = True
    if "نزلہ" in t:
        has_cold = True
    if "دست" in t:
        has_loose_stools = True
    if "الٹی" in t:
        has_vomit = True
    if "پیٹ درد" in t:
        has_abd_pain = True
    if "سانس پھول" in t or "سانس لینے میں تکلیف" in t:
        has_breath = True
    if "سینے میں درد" in t:
        has_chest_pain = True
    if "مچھر" in t:
        has_mosquito = True

    # Nepali
    if "ज्वरो" in t:
        has_fever = True
    if " खोकी" in t or "खोकी" in t:
        has_cough = True
    if "रुघा" in t:
        has_cold = True
    if "झाडा" in t:
        has_loose_stools = True
    if "वान्ता" in t:
        has_vomit = True
    if "पेट दुख" in t or "पेट दुखाई" in t:
        has_abd_pain = True

    possible: List[str] = []

    # Respiratory / viral fever patterns
    if has_fever and (has_cough or has_cold or has_sore_throat):
        possible.append(
            "viral fever, common cold, or flu-like illnesses affecting the nose, throat, and lungs"
        )

    # Dengue-like pattern: fever + body pains + headache/behind eyes +/- rash
    if has_fever and has_body_pain and (has_headache or has_behind_eyes or has_rash):
        possible.append(
            "dengue or other viral infections that cause high fever with strong body pain and headache"
        )

    # Malaria-like pattern: fever + chills + mosquito/travel
    if has_fever and has_chills and (has_mosquito or has_travel):
        possible.append(
            "malaria or other mosquito-borne infections, especially if you are in or returned from a malaria-prone area"
        )

    # Gastroenteritis / food-related illness pattern
    if has_loose_stools or (has_vomit and has_abd_pain):
        possible.append(
            "gastroenteritis or food and water-related infections that cause loose stools, vomiting, and stomach pain"
        )

    # Urinary tract infection pattern
    if has_burning_urine or (has_urine_freq and (has_fever or "urine" in t)):
        possible.append(
            "urinary tract infection (UTI) or irritation of the urinary tract"
        )

    # Chest pain + breathlessness: serious cardio-respiratory problems
    if has_chest_pain and has_breath:
        possible.append(
            "serious heart or lung problems. This can be an emergency and needs immediate medical attention"
        )

    # Only upper-respiratory, no fever
    if not has_fever and (has_cough or has_cold or has_sore_throat):
        possible.append(
            "common cold, throat infection, or other upper-respiratory infections"
        )

    if not possible:
        return ""

    if len(possible) == 1:
        text = (
            f"From your description, your symptoms can sometimes be seen in conditions such as {possible[0]}. "
            "These are only possibilities; only a doctor and appropriate tests can confirm the exact cause."
        )
    else:
        joined = "; ".join(possible[:-1]) + f"; or {possible[-1]}"
        text = (
            f"From your description, your symptoms can be seen in conditions such as {joined}. "
            "These are only possibilities; only a doctor and appropriate tests can confirm whether any of these applies to you."
        )
    return text


def _build_possible_causes_text(top3_diseases: List[Tuple[str, float]]) -> str:
    """
    Turn top3_diseases [(key, score), ...] into a safe "possible causes" sentence.

    - Never says "this is your diagnosis".
    - Uses phrases like "can be seen in conditions such as...".
    - If scores look low/flat, emphasises uncertainty and seeing a doctor.
    """
    if not top3_diseases:
        return ""
    # Keep only disease keys; scores may be None/0 for some models
    names: List[str] = []
    scores: List[float] = []
    for key, score in top3_diseases[:3]:
        try:
            s = float(score)
        except Exception:
            s = 0.0
        scores.append(s)
        # Use human-friendly name from KB when available
        try:
            adv = retrieve_disease_advisory(key, lang="en")
            label = adv.get("name") or key
        except Exception:
            label = key
        if label not in names:
            names.append(label)
    if not names:
        return ""

    # Simple confidence heuristic
    top_score = max(scores) if scores else 0.0
    second_score = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0.0
    confident = top_score >= 0.7 and (top_score - second_score) >= 0.15

    if len(names) == 1:
        if confident:
            return (
                f"Your symptoms can sometimes be seen in illnesses such as {names[0]}. "
                "Only a doctor and appropriate tests can confirm this."
            )
        return (
            f"Your symptoms may fit different illnesses, for example {names[0]}, "
            "but it is not possible to be sure from this chat. A doctor should examine you."
        )

    # Multiple possible conditions
    if len(names) == 2:
        label = " or ".join(names)
    else:
        label = ", ".join(names[:-1]) + f", or {names[-1]}"

    if confident:
        return (
            f"Your symptoms can be seen in conditions such as {label}. "
            "These are only possibilities; a doctor can confirm the exact cause."
        )
    return (
        f"From your symptoms, conditions like {label} are possible, but it is not clear which one. "
        "You should visit a doctor for proper examination and tests."
    )


class DialogueState:
    """Per-session state: accumulated symptoms, last intent, turn count."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.symptoms: List[tuple] = []  # (code, phrase)
        self.last_intent: Optional[str] = None
        self.turn_count: int = 0
        self.last_diseases: List[tuple] = []  # (name, score)
        self.summary: Optional[str] = None

    def add_symptoms(self, new: List[tuple]) -> None:
        seen = {s[0] for s in self.symptoms}
        for code, phrase in new:
            if code not in seen:
                self.symptoms.append((code, phrase))
                seen.add(code)

    def all_symptom_codes(self) -> List[str]:
        return [s[0] for s in self.symptoms]


class DialogueManager:
    """Decide next action: follow-up, prevention, emergency. Ollama for phrasing only."""

    def __init__(self):
        self._sessions: Dict[str, DialogueState] = {}

    def get_or_create_state(self, session_id: str) -> DialogueState:
        if session_id not in self._sessions:
            self._sessions[session_id] = DialogueState(session_id)
        return self._sessions[session_id]

    def next_action(
        self,
        session_id: str,
        intent: str,
        symptoms: List[tuple],
        top3_diseases: List[tuple],
        is_emergency: bool,
        lang: str = "en",
        user_message: Optional[str] = None,
    ) -> str:
        """
        Decide next action and return action type: follow_up | prevention | vaccination | emergency | general.
        """
        state = self.get_or_create_state(session_id)
        state.turn_count += 1
        state.add_symptoms(symptoms)
        state.last_intent = intent
        state.last_diseases = top3_diseases

        if is_emergency:
            return "emergency"

        # Check for location queries
        if user_message:
            location_query_keywords = [
                "hospital", "doctor", "clinic", "medical", "healthcare",
                "near me", "nearby", "location", "where", "find", "near"
            ]
            message_lower = user_message.lower()
            if any(keyword in message_lower for keyword in location_query_keywords):
                return "location"

        if intent == "vaccination_schedule":
            return "vaccination"
        if intent == "prevention_guidance":
            return "prevention"
        if intent == "disease_information" or intent == "symptom_reporting":
            # If user is reporting symptoms (regardless of language), give personalised advice
            if symptoms:
                return "symptom_advice"
            if top3_diseases:
                return "prevention"
            # No symptoms this turn: treat as general medical question
            if not symptoms:
                return "general"
            if not state.symptoms and state.turn_count <= 2:
                return "follow_up"
            return "prevention"
        return "general"

    def build_response(
        self,
        session_id: str,
        action: str,
        intent: str,
        symptoms: List[tuple],
        top3_diseases: List[tuple],
        is_emergency: bool,
        lang: str = "en",
        use_ollama_phrasing: bool = True,
        user_message: Optional[str] = None,
        user_location: Optional[str] = None,
    ) -> str:
        """
        Build response: template + retrieved content. Optionally use Ollama for phrasing.
        For general medical questions, uses Ollama to answer.
        """
        state = self.get_or_create_state(session_id)
        parts: List[str] = []

        if is_emergency:
            parts.append(
                "Based on the symptoms you described (for example breathing difficulty or chest pain), "
                "you should seek immediate care at the nearest hospital or emergency facility."
            )
            if DISCLAIMER:
                parts.append(DISCLAIMER)
            return " ".join(parts)

        # Disease explanation mode: user asking "what is X / explain X", not mainly reporting symptoms
        if (
            intent == "disease_information"
            and top3_diseases
            and user_message
            and use_ollama_phrasing
        ):
            main_disease = top3_diseases[0][0]
            try:
                adv = retrieve_disease_advisory(main_disease, lang=lang)
                disease_name = adv.get("name") or main_disease
                prevent = "; ".join(adv.get("prevention", [])[:5])
                vaccines = "; ".join(adv.get("vaccines", [])[:3])
                updates = " ".join(
                    (u.get("summary") or "") for u in adv.get("updates", [])[:2]
                )
            except Exception:
                disease_name = main_disease
                prevent = ""
                vaccines = ""
                updates = ""

            factual_context_lines: List[str] = []
            factual_context_lines.append(f"Disease key: {main_disease}")
            factual_context_lines.append(f"Name: {disease_name}")
            if prevent:
                factual_context_lines.append(f"Prevention: {prevent}")
            if vaccines:
                factual_context_lines.append(f"Vaccines: {vaccines}")
            if updates:
                factual_context_lines.append(f"Updates: {updates}")
            factual_context = "\n".join(factual_context_lines)

            student_mode = _is_student_query(user_message)
            try:
                if student_mode:
                    prompt = (
                        "You are a clinical teaching assistant for medical and nursing students in India. "
                        "Using ONLY standard, widely accepted medical knowledge and the factual context below, "
                        "explain the requested disease in a structured way. "
                        "Do NOT give drug doses or detailed prescription regimens.\n\n"
                        f"{_language_instruction(lang)}\n\n"
                        "Factual context:\n"
                        f"{factual_context}\n\n"
                        "Write a concise but high-yield note with clear headings in this order:\n"
                        "1. Definition\n"
                        "2. Etiology / causes\n"
                        "3. Brief pathophysiology (high-level)\n"
                        "4. Clinical features (group common symptoms and signs)\n"
                        "5. Basic investigations (no hospital-specific logistics)\n"
                        "6. Principles of management (only general approach, no specific drug doses)\n"
                        "7. Important complications\n"
                        "8. Key exam points (short bullet list)\n\n"
                        "Disease/topic from the user's question:\n"
                        f"{user_message}\n"
                    )
                else:
                    prompt = (
                        "You are a medical information assistant for the public in India. "
                        "Explain the disease in simple language for a non-medical person. "
                        "Use the factual context below and standard medical knowledge. "
                        "Do NOT give a diagnosis for the specific user and do NOT prescribe exact medicines.\n\n"
                        f"{_language_instruction(lang)}\n\n"
                        "Factual context:\n"
                        f"{factual_context}\n\n"
                        "Write 1–3 short paragraphs that cover:\n"
                        "- What the disease is (simple definition)\n"
                        "- Main cause and how it usually spreads (if infectious)\n"
                        "- Common signs and symptoms\n"
                        "- When a person should see a doctor or go to hospital\n"
                        "- Simple prevention steps relevant to daily life\n\n"
                        "Disease/topic from the user's question:\n"
                        f"{user_message}\n"
                    )
                answer = call_ollama(prompt)
                if answer and _is_valid_answer(answer):
                    text = answer.strip()
                    if DISCLAIMER:
                        text = text + " " + DISCLAIMER
                    return text
            except Exception:
                # fall through to normal handling below
                pass

        # When user asks for advice or has symptoms
        if user_message and (symptoms or _is_advice_seeking(user_message)):
            # Build grounded factual context from ML + KB + alerts
            kb_lines: List[str] = []
            for disease_key, score in (top3_diseases or [])[:2]:
                adv = retrieve_disease_advisory(disease_key, lang=lang)
                prevent = "; ".join(adv.get("prevention", [])[:3])
                vaccines = "; ".join(adv.get("vaccines", [])[:2])
                parts_ctx = []
                if prevent:
                    parts_ctx.append(f"prevention: {prevent}")
                if vaccines:
                    parts_ctx.append(f"vaccines: {vaccines}")
                if parts_ctx:
                    kb_lines.append(f"{adv['name']} ({disease_key}) – " + "; ".join(parts_ctx))
            region = _normalize_region_from_location(user_location)
            alert_snippets = _get_region_alert_snippets(region, top3_diseases)
            if alert_snippets:
                kb_lines.append("Current public health alerts relevant to the user:")
                kb_lines.extend(alert_snippets)
            factual_context = "\n".join(kb_lines) if kb_lines else "No specific disease context available; use only generic safe advice."

            if use_ollama_phrasing:
                for attempt in range(2):
                    try:
                        symptom_phrase = (
                            " The user reports: " + ", ".join(p for _, p in symptoms) + "."
                            if symptoms
                            else ""
                        )
                        # Prefer rule-based explanation from user text (does not rely on ML scores)
                        possible_causes = _build_rule_based_symptom_explanation(user_message)
                        if attempt == 0:
                            prompt = (
                                "You are a medical information assistant for an Indian public-health chatbot. "
                                "Base your answer ONLY on the factual context, possible causes text, and general medical safety principles. "
                                "Do NOT give a confirmed diagnosis and do NOT prescribe medicines. "
                                "Always recommend seeing a doctor for concerning or persistent symptoms. "
                                f"{_language_instruction(lang)}\n\n"
                                "Factual context (from knowledge base, ensemble model, and health alerts):\n"
                                f"{factual_context}\n\n"
                                "Possible causes (not a diagnosis):\n"
                                f"{possible_causes or 'No clear disease suggestion; focus on safe general advice.'}\n\n"
                                "Now answer the user's health question in 3–6 sentences. "
                                "Structure your answer as: (1) possible causes in simple language (not a diagnosis), "
                                "(2) what the user can do at home now, (3) clear guidance on when to see a doctor or go to hospital. "
                                "Reply with ONLY the answer."
                                + symptom_phrase
                                + "\n\nUser question: "
                                + user_message
                            )
                        else:
                            prompt = (
                                "In 3–5 short sentences, give practical health advice for the following question. "
                                "Use ONLY this factual context and possible-causes text; do not state a confirmed diagnosis or prescribe medicines. "
                                f"{_language_instruction(lang)}\n\n"
                                "Factual context:\n"
                                f"{factual_context}\n\n"
                                "Possible causes (not a diagnosis):\n"
                                f"{possible_causes or 'No clear disease suggestion; focus on safe general advice.'}\n\n"
                                "User question: "
                                + user_message
                                + symptom_phrase
                            )
                        answer = call_ollama(prompt)
                        if answer and _is_valid_answer(answer):
                            return answer.strip() + " " + DISCLAIMER
                    except Exception:
                        break

            # Offline / fallback: rule-based possible causes + home care + doctor guidance
            parts_offline: List[str] = []
            maybe_causes = _build_rule_based_symptom_explanation(user_message)
            if maybe_causes:
                parts_offline.append(maybe_causes)

            msg_lower = (user_message or "").lower()
            _fever_keywords = [
                "fever", "bukhar", "bukar", "buukhar", "bhukar", "jwar", "jwara", "taap",
                "बुखार", "ज्वर", "ताप",   # Hindi/Marathi Devanagari
                "জ্বর",                    # Bengali/Assamese
                "జ్వరం",                   # Telugu
                "காய்ச்சல்",              # Tamil
                "ಜ್ವರ",                    # Kannada
                "പനി",                     # Malayalam
                "تاو", "بخار",             # Urdu
                "ਬੁਖਾਰ",                  # Punjabi
                "તાવ",                     # Gujarati
                "ज्वरो",                   # Nepali
                "ଜ୍ୱର",                   # Odia
            ]
            has_fever = any(kw in msg_lower for kw in _fever_keywords)

            # Basic home care
            if has_fever:
                parts_offline.append(
                    "For a mild fever, rest well, drink plenty of safe fluids, and avoid heavy physical activity. "
                    "You can monitor your temperature and note how many days the fever has lasted."
                )
            else:
                parts_offline.append(
                    "For many mild symptoms, resting, staying hydrated, and avoiding heavy physical work can help while you arrange to see a doctor."
                )

            # When to see a doctor / hospital
            parts_offline.append(
                "Please see a doctor urgently or visit a hospital if your symptoms are very severe, "
                "if you have trouble breathing, chest pain, confusion, seizures, bleeding, persistent vomiting, "
                "or if fever stays high or lasts more than 2–3 days."
            )

            text_offline = " ".join(parts_offline).strip()
            return (text_offline + " " + DISCLAIMER) if text_offline else (
                "For personalised advice, please visit a healthcare provider. " + DISCLAIMER
            )

        # ---------------------------------------------------------------
        # symptom_advice: personalised path for symptom_reporting in ANY language
        # ---------------------------------------------------------------
        if action == "symptom_advice":
            # Build factual context from KB + alerts
            kb_lines: List[str] = []
            for disease_key, score in (top3_diseases or [])[:2]:
                try:
                    adv = retrieve_disease_advisory(disease_key, lang=lang)
                    prevent = "; ".join(adv.get("prevention", [])[:3])
                    if prevent:
                        kb_lines.append(adv.get("name", disease_key) + ": " + prevent)
                except Exception:
                    pass
            region = _normalize_region_from_location(user_location)
            alert_snippets = _get_region_alert_snippets(region, top3_diseases)
            if alert_snippets:
                kb_lines.extend(alert_snippets[:2])
            factual_context = "\n".join(kb_lines) if kb_lines else "No specific disease context available."
            possible_causes = _build_rule_based_symptom_explanation(user_message)
            symptom_phrase = (
                " The user reports: " + ", ".join(p for _, p in symptoms) + "."
                if symptoms else ""
            )

            if use_ollama_phrasing:
                try:
                    lang_instr = _language_instruction(lang)
                    possible_str = possible_causes or "Unknown; give safe general advice."
                    prompt = (
                        "You are a helpful medical information assistant for a public-health chatbot in India. "
                        "The user has described their symptoms. Do NOT give a confirmed diagnosis. "
                        "Do NOT prescribe medicines by name. "
                        "Always advise seeing a doctor for persistent or severe symptoms.\n\n"
                        + lang_instr + "\n\n"
                        "Factual context:\n"
                        + factual_context + "\n\n"
                        "Possible causes (not a diagnosis):\n"
                        + possible_str + "\n\n"
                        "Answer in 3-5 sentences in the SAME LANGUAGE as the user message. "
                        "Cover: (1) what could cause these symptoms simply, "
                        "(2) what to do at home now, "
                        "(3) when to see a doctor urgently. "
                        "Reply ONLY with the answer, nothing else."
                        + symptom_phrase
                        + "\n\nUser message: "
                        + (user_message or "")
                    )
                    answer = call_ollama(prompt)
                    if answer and _is_valid_answer(answer):
                        return answer.strip() + " " + DISCLAIMER
                except Exception:
                    pass

            # Offline fallback: multilingual fever check
            parts_sa: List[str] = []
            maybe_causes_sa = _build_rule_based_symptom_explanation(user_message)
            if maybe_causes_sa:
                parts_sa.append(maybe_causes_sa)

            _fever_kw = [
                "fever", "bukhar", "bukar", "buukhar", "bhukar", "jwar", "taap",
                "\u092c\u0941\u0916\u093e\u0930", "\u091c\u094d\u0935\u0930",
                "\u099c\u09cd\u09ac\u09b0", "\u0c1c\u0c4d\u0c35\u0c30\u0c02",
                "\u0b95\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bb2\u0bcd",
                "\u0c9c\u0ccd\u0cb5\u0cb0", "\u0d2a\u0d28\u0d3f",
                "\u0628\u062e\u0627\u0631", "\u0a2c\u0a41\u0a16\u0a3c\u0a3e\u0a30",
                "\u0aa4\u0abe\u0ab5", "\u091c\u094d\u0935\u0930\u094b",
                "\u0b1c\u0b4d\u0b5f\u0b41\u0b31",
            ]
            msg_l = (user_message or "").lower()
            _has_fever_sa = any(kw in msg_l for kw in _fever_kw) or any(kw in (user_message or "") for kw in _fever_kw)

            if _has_fever_sa:
                parts_sa.append(
                    "For a mild fever, rest well, drink plenty of safe fluids, and avoid heavy physical activity. "
                    "Monitor your temperature and note how many days the fever has lasted."
                )
            else:
                parts_sa.append(
                    "For mild symptoms, rest, stay hydrated, and avoid heavy physical work while arranging to see a doctor."
                )
            parts_sa.append(
                "Please see a doctor urgently if symptoms are severe, you have trouble breathing, "
                "chest pain, confusion, seizures, persistent vomiting, or fever lasts more than 2-3 days."
            )
            return (" ".join(parts_sa) + " " + DISCLAIMER).strip()
        if action == "vaccination":
            # Retrieve vaccination info for top disease or generic
            if top3_diseases:
                adv = retrieve_disease_advisory(top3_diseases[0][0], lang=lang)
                vaccines = adv.get("vaccines", [])
                if vaccines:
                    parts.append("Vaccination guidance (education only): " + "; ".join(vaccines) + ".")
                else:
                    parts.append("For vaccination schedules, please check with your local health centre or MoHFW/WHO guidelines.")
            else:
                parts.append("You can get vaccination schedules from your nearest health centre or official MoHFW/WHO advisories.")
            # Attach region-specific alerts if relevant
            region = _normalize_region_from_location(user_location)
            alert_snippets = _get_region_alert_snippets(region, top3_diseases)
            if alert_snippets:
                parts.append("Current public health alerts for your region:")
                parts.extend(alert_snippets)
            parts.append(DISCLAIMER)
            return " ".join(parts)

        if action == "prevention" and top3_diseases:
            for disease_key, score in top3_diseases[:2]:
                adv = retrieve_disease_advisory(disease_key, lang=lang)
                parts.append(f"Prevention tips for {adv['name']} (education only): " + "; ".join(adv.get("prevention", [])[:3]) + ".")
                for u in adv.get("updates", [])[:1]:
                    parts.append(f"Latest ({u.get('source', '')}): {u.get('summary', '')}")
            if not parts:
                parts.append("General prevention: hand hygiene, safe water, and timely vaccination. Consult a healthcare provider for advice.")
            # Attach region-specific alerts if relevant
            region = _normalize_region_from_location(user_location)
            alert_snippets = _get_region_alert_snippets(region, top3_diseases)
            if alert_snippets:
                parts.append("Current public health alerts for your region:")
                parts.extend(alert_snippets)
            parts.append(DISCLAIMER)
            raw = " ".join(parts)
            if use_ollama_phrasing:
                try:
                    prompt = f"Rephrase the following health education message in simple, clear language. Do not add medical advice or diagnosis. Output only the rephrased text:\n\n{raw}"
                    phrased = call_ollama(prompt)
                    if phrased and "[Ollama" not in phrased:
                        return phrased.strip() + " " + DISCLAIMER
                except Exception:
                    pass
            return raw

        if action == "follow_up":
            parts.append("Could you describe any symptoms you are experiencing? (e.g. fever, cough, body pain) This helps us share relevant prevention information.")
            parts.append(DISCLAIMER)
            return " ".join(parts)

        # Location queries: find nearby hospitals
        if action == "location":
            location = None
            
            # Use provided coordinates if available
            if user_location:
                location = user_location
            elif user_message:
                # Try to extract location from message
                extracted = extract_location_from_message(user_message)
                if extracted and extracted != "current_location":
                    location = extracted
                elif extracted == "current_location":
                    # User said "near me" but no coordinates provided
                    parts.append(
                        "To find hospitals near you, please share your location or specify your city/area. "
                        "For example: 'hospitals in Bangalore' or 'find hospital near me' (with location enabled)."
                    )
                    parts.append(DISCLAIMER)
                    return " ".join(parts)
            
            # Default location if none provided
            if not location:
                location = "India"  # Default to India
            
            # Find hospitals
            hospitals = find_nearby_hospitals(location, radius_km=10, limit=5)
            response = format_hospital_response(hospitals, location)
            parts.append(response)
            parts.append(DISCLAIMER)
            return " ".join(parts)

        # General medical / health questions
        if action == "general" and user_message:
            if use_ollama_phrasing:
                # Ground the answer on KB + alerts when available
                kb_lines: List[str] = []
                for disease_key, score in (top3_diseases or [])[:2]:
                    adv = retrieve_disease_advisory(disease_key, lang=lang)
                    prevent = "; ".join(adv.get("prevention", [])[:3])
                    vaccines = "; ".join(adv.get("vaccines", [])[:2])
                    parts_ctx = []
                    if prevent:
                        parts_ctx.append(f"prevention: {prevent}")
                    if vaccines:
                        parts_ctx.append(f"vaccines: {vaccines}")
                    if parts_ctx:
                        kb_lines.append(f"{adv['name']} ({disease_key}) – " + "; ".join(parts_ctx))
                region = _normalize_region_from_location(user_location)
                alert_snippets = _get_region_alert_snippets(region, top3_diseases)
                if alert_snippets:
                    kb_lines.append("Current public health alerts relevant to the user:")
                    kb_lines.extend(alert_snippets)
                factual_context = "\n".join(kb_lines) if kb_lines else "No specific disease context available; use only generic safe advice."
                # Prefer rule-based explanation based on user text
                possible_causes = _build_rule_based_symptom_explanation(user_message)

                try:
                    prompt = (
                        "You are a medical information assistant for an Indian public-health chatbot. "
                        "Base your answer ONLY on the factual context, possible causes, and general medical safety principles. "
                        "Do NOT give a confirmed diagnosis or prescribe medicines. "
                        "Always recommend seeing a doctor for concerning or persistent symptoms. "
                        f"{_language_instruction(lang)}\n\n"
                        "Factual context (from knowledge base, ensemble model, and health alerts):\n"
                        f"{factual_context}\n\n"
                        "Possible causes (not a diagnosis):\n"
                        f"{possible_causes or 'No clear disease suggestion; focus on safe general advice.'}\n\n"
                        "Answer this health or medical question in 3–6 sentences. Be helpful and direct. "
                        "Explain possible causes in simple language, then what the user can do now, and when to see a doctor or hospital.\n\nQuestion: "
                        + user_message
                    )
                    answer = call_ollama(prompt)
                    if answer and _is_valid_answer(answer):
                        text = answer.strip()
                        return text + " " + DISCLAIMER
                except Exception:
                    pass

            # Offline: rule-based possible causes + simple prevention + doctor guidance
            parts_general: List[str] = []
            maybe_causes = _build_rule_based_symptom_explanation(user_message)
            if maybe_causes:
                parts_general.append(maybe_causes)

            if top3_diseases:
                try:
                    adv = retrieve_disease_advisory(top3_diseases[0][0], lang=lang)
                    tips = adv.get("prevention", [])[:3]
                except Exception:
                    tips = []
                if tips:
                    parts_general.append(
                        "General prevention tips that may help reduce infection risk include: "
                        + "; ".join(tips)
                        + "."
                    )

            parts_general.append(
                "For any worrying, persistent, or severe symptoms, please visit a qualified doctor or nearby hospital. "
                "They can examine you properly and order tests if needed."
            )

            text_general = " ".join(parts_general).strip()
            if text_general:
                return text_general + " " + DISCLAIMER

        # Final fallback: never return empty
        parts.append("For personalised advice, please visit a healthcare provider.")
        parts.append(DISCLAIMER)
        out = " ".join(parts)
        return out if out.strip() else "Please try asking in a different way, or consult a healthcare provider. " + DISCLAIMER