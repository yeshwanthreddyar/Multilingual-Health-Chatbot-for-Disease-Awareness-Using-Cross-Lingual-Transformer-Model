"""
REST API (FastAPI) - Chat, document review, alerts.
"""
from __future__ import annotations

import base64
from typing import Any, List, Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import DISCLAIMER
from app.integrations.government_mock import get_advisories, get_alerts, get_vaccines
from app.llm.ollama_client import call_ollama_vision
from app.dialog.manager import DialogueManager
from app.ml.pipeline import get_ml_pipeline
from app.nlp.pipeline import get_nlp_pipeline

router = APIRouter()
_dialog_manager = DialogueManager()

# Allowed image types for prescription/report upload
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _get_dialog_state(session_id: str):
    return _dialog_manager.get_or_create_state(session_id)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Session for multi-turn")
    location: Optional[str] = Field(None, description="User location: city name or 'lat,lon' coordinates")
    language_override: Optional[str] = Field(
        None,
        description="Force response in specific language (en, hi, bn, te, ta, kn, ml, mr, gu, pa, or, as, ur, ne, kok)"
    )

class ChatResponse(BaseModel):
    response: str
    lang: str
    intent: str
    session_id: str
    is_emergency: bool = False


def _get_user_id_from_token(authorization: Optional[str] = None) -> Optional[int]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        from app.api.auth import get_current_user
        user = get_current_user(authorization.split(" ", 1)[1])
        if not user:
            return None
        from app.db import SessionLocal
        from app.models.user import User
        db = SessionLocal()
        u = db.query(User).filter(User.email == user["email"]).first()
        db.close()
        return u.id if u else None
    except Exception:
        return None


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, authorization: Optional[str] = Header(None)) -> ChatResponse:
    """Process user message: NLP → ML → Dialogue → Response."""
    session_id = req.session_id or "default"
    nlp = get_nlp_pipeline()
    ml = get_ml_pipeline()

    nlp_out = nlp.process(req.message)
    ml_out = ml.run(
        req.message,
        nlp_out["tokens"],
        nlp_out["embedding"],
    )

    _VALID_LANG_CODES = {"en","hi","bn","te","ta","kn","ml","mr","gu","pa","or","as","ur","ne","kok"}
    effective_lang = (
        req.language_override
        if req.language_override and req.language_override in _VALID_LANG_CODES
        else nlp_out["lang"]
    )

    action = _dialog_manager.next_action(
        session_id,
        ml_out["intent"],
        ml_out["symptoms"],
        ml_out["top3_diseases"],
        ml_out["is_emergency"],
        lang=nlp_out["lang"],
        user_message=req.message,
    )

    response_text = _dialog_manager.build_response(
        session_id,
        action,
        ml_out["intent"],
        ml_out["symptoms"],
        ml_out["top3_diseases"],
        ml_out["is_emergency"],
        lang=nlp_out["lang"],
        use_ollama_phrasing=True,
        user_message=req.message,
        user_location=req.location,
    )
    if not (response_text and response_text.strip()):
        response_text = "Something went wrong. Please try again or rephrase your question. " + DISCLAIMER

    # Record case when user reports symptoms (for outbreak tracking)
    if (
        ml_out["intent"] in ("symptom_reporting", "disease_information")
        and ml_out.get("top3_diseases")
    ):
        try:
            from app.outbreak.case_records import add_case
            region = (req.location or "India").split(",")[0].strip() or "India"
            disease = ml_out["top3_diseases"][0][0]
            add_case(region=region, disease=disease, source="self_report")
        except Exception:
            pass

    # Log for admin and training
    try:
        from app.db import SessionLocal
        from app.models.log import Log
        health_intents = {"symptom_reporting", "disease_information", "prevention_guidance", "treatment_info", "diagnosis_info"}
        db = SessionLocal()
        db.add(Log(
            user_id=_get_user_id_from_token(authorization),
            session_id=session_id,
            message=req.message,
            response=response_text,
            intent=ml_out.get("intent"),
            is_health_related=ml_out.get("intent") in health_intents,
        ))
        db.commit()
        db.close()
    except Exception:
        pass

    return ChatResponse(
        response=response_text,
        lang=nlp_out["lang"],
        intent=ml_out["intent"],
        session_id=session_id,
        is_emergency=ml_out["is_emergency"],
    )


@router.post("/chat-offline", response_model=ChatResponse)
def chat_offline(req: ChatRequest, authorization: Optional[str] = Header(None)) -> ChatResponse:
    """Same as /chat but never uses Ollama or external APIs. Uses only trained models + knowledge base."""
    session_id = req.session_id or "default"
    nlp = get_nlp_pipeline()
    ml = get_ml_pipeline()

    nlp_out = nlp.process(req.message)
    ml_out = ml.run(
        req.message,
        nlp_out["tokens"],
        nlp_out["embedding"],
    )

    _VALID_LANG_CODES = {"en","hi","bn","te","ta","kn","ml","mr","gu","pa","or","as","ur","ne","kok"}
    effective_lang = (
        req.language_override
        if req.language_override and req.language_override in _VALID_LANG_CODES
        else nlp_out["lang"]
    )

    action = _dialog_manager.next_action(
        session_id,
        ml_out["intent"],
        ml_out["symptoms"],
        ml_out["top3_diseases"],
        ml_out["is_emergency"],
        lang=nlp_out["lang"],
        user_message=req.message,
    )

    response_text = _dialog_manager.build_response(
        session_id,
        action,
        ml_out["intent"],
        ml_out["symptoms"],
        ml_out["top3_diseases"],
        ml_out["is_emergency"],
        lang=nlp_out["lang"],
        use_ollama_phrasing=False,
        user_message=req.message,
        user_location=req.location,
    )
    if not (response_text and response_text.strip()):
        response_text = "Something went wrong. Please try again or rephrase your question. " + DISCLAIMER

    # Record case when user reports symptoms (for outbreak tracking)
    if (
        ml_out["intent"] in ("symptom_reporting", "disease_information")
        and ml_out.get("top3_diseases")
    ):
        try:
            from app.outbreak.case_records import add_case
            region = (req.location or "India").split(",")[0].strip() or "India"
            disease = ml_out["top3_diseases"][0][0]
            add_case(region=region, disease=disease, source="self_report")
        except Exception:
            pass

    # Log for admin and training
    try:
        from app.db import SessionLocal
        from app.models.log import Log
        health_intents = {"symptom_reporting", "disease_information", "prevention_guidance", "treatment_info", "diagnosis_info"}
        db = SessionLocal()
        db.add(Log(
            user_id=_get_user_id_from_token(authorization),
            session_id=session_id,
            message=req.message,
            response=response_text,
            intent=ml_out.get("intent"),
            is_health_related=ml_out.get("intent") in health_intents,
        ))
        db.commit()
        db.close()
    except Exception:
        pass

    return ChatResponse(
        response=response_text,
        lang=effective_lang,
        intent=ml_out["intent"],
        session_id=session_id,
        is_emergency=ml_out["is_emergency"],
    )


@router.get("/alerts")
def alerts(limit: int = 10) -> list:
    """Outbreak alerts (WHO/MoHFW style + HealthBot outbreak detection)."""
    from app.outbreak.detector import merge_with_existing_alerts
    existing = get_alerts(limit=limit * 2)
    return merge_with_existing_alerts(existing, limit=limit)


@router.get("/vaccines")
def vaccines(age_group: Optional[str] = None) -> list:
    """Mock: vaccination schedules."""
    return get_vaccines(age_group=age_group)


@router.get("/health-updates")
def health_updates(disease: Optional[str] = None, region: Optional[str] = None, limit: int = 10) -> list:
    """WHO and Indian health updates (from data/who_updates.json, data/india_health_updates.json)."""
    from app.data.load_health_updates import get_health_updates
    return get_health_updates(disease=disease, region=region, limit=limit)


@router.get("/advisories")
def advisories(category: Optional[str] = None, limit: int = 10) -> list:
    """Mock: public health advisories."""
    return get_advisories(category=category, limit=limit)


@router.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok", "service": "healthbot"}


# ----- Prescription / Report review -----

class ReviewResponse(BaseModel):
    review: str
    disclaimer: str = DISCLAIMER


@router.post("/review-document", response_model=ReviewResponse)
async def review_document(
    file: UploadFile = File(...),
) -> ReviewResponse:
    """
    Upload a prescription or medical report (image). Returns detailed review and suggestions from Ollama vision.
    """
    if (file.content_type or "").split(";")[0].strip().lower() not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only image files are supported (JPEG, PNG, WebP, GIF). For PDF, take a screenshot and upload.",
        )
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")
    b64 = base64.b64encode(contents).decode("utf-8")
    prompt = (
        "You are a medical assistant. Review this prescription or medical report image in detail. "
        "Provide: (1) Summary of what the document shows (medicines, dosages, diagnosis, lab values if any). "
        "(2) Suggestions: follow-up care, things to watch for, when to see a doctor, drug interactions or precautions if visible. "
        "(3) Any concerns or clarifications the patient should ask their doctor. "
        "Be clear and structured. Reply with only the review text."
    )
    review = call_ollama_vision(prompt, [b64])
    if "[Ollama" in review:
        review = "Vision model unavailable. Please ensure Ollama is running with a vision model (e.g. ollama pull llava). " + review
    return ReviewResponse(review=review.strip(), disclaimer=DISCLAIMER)


# ----- Location-based alerts -----

def get_alerts_for_region(region: Optional[str] = None, limit: int = 10) -> list:
    """Return alerts; if region given, filter to that region or 'National' or 'Multiple states'."""
    from app.integrations.alert_sender import region_matches
    from app.outbreak.detector import merge_with_existing_alerts
    existing = get_alerts(limit=limit * 2)
    all_alerts = merge_with_existing_alerts(existing, limit=limit * 2)
    if not region or not region.strip():
        return all_alerts[:limit]
    r = region.strip()
    out = [a for a in all_alerts if region_matches(a.get("region") or "", r)]
    return (out or all_alerts)[:limit]


class SubscribeAlertsRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=20)
    region: str = Field(..., min_length=1, max_length=200)
    channel: str = Field("whatsapp", description="whatsapp or sms")


@router.get("/alerts/by-location")
def alerts_by_location(region: Optional[str] = None, limit: int = 10) -> list:
    """Live disease/outbreak alerts for a region (real-time from source)."""
    return get_alerts_for_region(region, limit=limit)


@router.post("/alerts/subscribe")
def subscribe_alerts(req: SubscribeAlertsRequest) -> dict:
    """Subscribe to receive disease alerts via WhatsApp or SMS. Sends current alerts once (real-time)."""
    from app.integrations.alert_sender import send_alerts_to_subscribers
    from app.integrations.alert_subscriptions import add_subscription
    add_subscription(req.phone, req.region, req.channel.lower() or "whatsapp")
    sub = {"phone": req.phone, "region": req.region, "channel": req.channel.lower() or "whatsapp"}
    alerts = get_alerts_for_region(req.region, limit=10)
    result = send_alerts_to_subscribers([sub], alerts)
    return {
        "status": "ok",
        "message": f"Subscribed to {req.channel} alerts for region: {req.region}.",
        "alerts_sent": result.get("sent_whatsapp", 0) + result.get("sent_sms", 0),
    }


@router.get("/cases")
def list_cases(region: Optional[str] = None, disease: Optional[str] = None, days: Optional[int] = 7) -> list:
    """List disease cases (for outbreak tracking). Filter by region, disease, or last N days."""
    from app.outbreak.case_records import get_cases
    return get_cases(region=region, disease=disease, days=days)


@router.get("/alerts/subscriptions")
def list_alert_subscriptions() -> list:
    """List current alert subscriptions (for testing)."""
    from app.integrations.alert_subscriptions import get_subscriptions
    return get_subscriptions()


@router.post("/alerts/notify")
def notify_alert_subscribers() -> dict:
    """Send current disease alerts to all subscribers (real-time push). Called by background task or cron."""
    from app.integrations.alert_sender import send_alerts_to_subscribers
    from app.integrations.alert_subscriptions import get_subscriptions
    alerts = get_alerts(limit=20)
    result = send_alerts_to_subscribers(get_subscriptions(), alerts)
    return {"status": "ok", "result": result}
