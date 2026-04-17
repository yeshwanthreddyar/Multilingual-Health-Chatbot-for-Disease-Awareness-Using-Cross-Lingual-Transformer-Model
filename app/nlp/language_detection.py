"""
Language Detection - Character n-gram / Unicode based.
Output: ISO 639-1 language code.
Supports 15+ Indian languages.
"""
from __future__ import annotations

import re
from typing import Optional

# ISO codes for required languages
SUPPORTED_LANGUAGES = {
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

# Unicode ranges for Indian scripts (simplified)
UNICODE_RANGES = {
    "hi": (0x0900, 0x097F),   # Devanagari (Hindi, Marathi, Nepali)
    "bn": (0x0980, 0x09FF),   # Bengali
    "te": (0x0C00, 0x0C7F),   # Telugu
    "ta": (0x0B80, 0x0BFF),   # Tamil
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "ml": (0x0D00, 0x0D7F),   # Malayalam
    "gu": (0x0A80, 0x0AFF),   # Gujarati
    "pa": (0x0A00, 0x0A7F),   # Gurmukhi (Punjabi)
    "or": (0x0B00, 0x0B7F),   # Odia
    "as": (0x0980, 0x09FF),   # Assamese (Bengali block)
    "ur": (0x0600, 0x06FF),   # Arabic (Urdu)
    "en": None,               # Latin
}


def _in_script(c: str, low: int, high: int) -> bool:
    if not c:
        return False
    cp = ord(c[0])
    return low <= cp <= high


def _dominant_script(text: str) -> Optional[str]:
    """Determine dominant script from Unicode."""
    counts: dict[str, int] = {}
    for c in text:
        if not c.strip():
            continue
        cp = ord(c)
        if 0x0900 <= cp <= 0x097F:
            counts["hi"] = counts.get("hi", 0) + 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["bn"] = counts.get("bn", 0) + 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["te"] = counts.get("te", 0) + 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["ta"] = counts.get("ta", 0) + 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["kn"] = counts.get("kn", 0) + 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["ml"] = counts.get("ml", 0) + 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["gu"] = counts.get("gu", 0) + 1
        elif 0x0A00 <= cp <= 0x0A7F:
            counts["pa"] = counts.get("pa", 0) + 1
        elif 0x0B00 <= cp <= 0x0B7F:
            counts["or"] = counts.get("or", 0) + 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["ur"] = counts.get("ur", 0) + 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["en"] = counts.get("en", 0) + 1
    if not counts:
        return "en"
    return max(counts, key=counts.get)


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    """Extract character n-grams."""
    text = re.sub(r"\s+", "", text.lower())
    return [text[i : i + n] for i in range(len(text) - n + 1)] if len(text) >= n else []


def detect_language(text: str) -> str:
    """
    Detect language using Unicode script + character n-gram heuristics.
    Returns ISO 639-1 code (e.g. 'hi', 'en').
    """
    if not text or not text.strip():
        return "en"

    script_lang = _dominant_script(text)
    if script_lang in ("hi", "bn"):
        # Devanagari and Bengali blocks cover multiple Indian languages (e.g., hi/mr/ne/kok and bn/as).
        # If langdetect is available, use it to disambiguate; otherwise fall back to the script default.
        try:
            import langdetect

            detected = langdetect.detect(text)
            if detected in SUPPORTED_LANGUAGES:
                return detected
        except Exception:
            pass
        return script_lang
    if script_lang:
        return script_lang

    # Fallback: langdetect if available
    try:
        import langdetect
        detected = langdetect.detect(text)
        if detected in SUPPORTED_LANGUAGES:
            return detected
        # Map to closest Indian language
        if detected in ("mr", "ne"):
            return detected
        return "en"
    except Exception:
        pass

    return "en"
