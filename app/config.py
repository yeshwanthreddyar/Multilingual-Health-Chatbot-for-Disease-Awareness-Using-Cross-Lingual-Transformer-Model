"""
App config - safety disclaimer, evaluation targets, environment variables.
"""
import os
from typing import Optional

# Safety & ethics - previously appended to all responses.
# You requested to remove the trailing informational disclaimer sentences from outputs,
# so this is now an empty string.
DISLAIMER_TEXT_REMOVED = True
DISCLAIMER = ""

# Evaluation targets (from PDF)
EVAL_TARGET_ACCURACY = 0.80   # ≥ 80%
EVAL_TARGET_LATENCY_SEC = 2.0  # mean ≤ 2 seconds
SUPPORTED_LANG_CODES = [
    "en", "hi", "bn", "te", "ta", "kn", "ml", "mr", "gu", "pa", "or", "as", "ur", "ne", "kok",
]

# Environment variables
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "healthbot_verify")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")

SMS_ACCOUNT_SID = os.getenv("SMS_ACCOUNT_SID", "")
SMS_AUTH_TOKEN = os.getenv("SMS_AUTH_TOKEN", "")
SMS_FROM_NUMBER = os.getenv("SMS_FROM_NUMBER", "")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
# Vision model for prescription/report review (e.g. llava, llava:13b)
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Auth
AUTH_SECRET = os.getenv("AUTH_SECRET", "change_this_secret_in_production")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@healthbot.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
AUTH_ALGORITHM = "HS256"
AUTH_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# External health portals (WHO / MoHFW / ICMR or proxies)
MOHFW_API_URL: Optional[str] = os.getenv("MOHFW_API_URL") or None
WHO_FEED_URL: Optional[str] = os.getenv("WHO_FEED_URL") or None
ICMR_API_URL: Optional[str] = os.getenv("ICMR_API_URL") or None

# Toggle live portal integration (otherwise use mock-only)
HEALTH_PORTALS_ENABLED: bool = (os.getenv("HEALTH_PORTALS_ENABLED", "false").lower() == "true")
HEALTH_PORTAL_TIMEOUT_SEC: float = float(os.getenv("HEALTH_PORTAL_TIMEOUT_SEC", "5.0"))
