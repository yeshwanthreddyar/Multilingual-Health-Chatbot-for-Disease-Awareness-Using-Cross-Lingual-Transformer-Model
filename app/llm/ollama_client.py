"""
Ollama integration - response phrasing, summarization, and vision (prescription/report review).
No translation. No SDKs. No cloud calls.
"""
import base64
from typing import List, Optional

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_VISION_MODEL


def call_ollama(prompt: str, model: Optional[str] = None, timeout: int = 20) -> str:
    """
    Call local Ollama API for response phrasing / summarization.
    Uses a 20s timeout so the app can return a fallback quickly if Ollama is slow or down.
    """
    if model is None:
        model = OLLAMA_MODEL
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        out = (response.json().get("response") or "").strip()
        return out if out else "[Ollama returned empty]"
    except Exception as e:
        return f"[Ollama unavailable: {e}]"


def call_ollama_vision(
    prompt: str,
    image_b64_list: List[str],
    model: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """
    Call Ollama with image(s) for prescription/report review. Use a vision model (e.g. llava).
    image_b64_list: list of base64-encoded image strings (no data URL prefix).
    """
    if model is None:
        model = OLLAMA_VISION_MODEL or OLLAMA_MODEL
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "images": image_b64_list,
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"[Ollama vision unavailable: {e}]"
