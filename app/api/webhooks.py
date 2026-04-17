"""
WhatsApp and SMS webhooks - real-time chat and alerts.
Users can: send any message to get a health answer; send "alerts" or "alerts <region>" for live alerts; send "subscribe <region>" to receive alerts on this number.
"""
from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, Request, Response, Query
from pydantic import BaseModel, Field

from app.dialog.manager import DialogueManager
from app.integrations.alert_sender import (
    format_alert_message,
    region_matches,
    send_alerts_to_subscribers,
)
from app.integrations.alert_subscriptions import add_subscription, get_subscriptions
from app.integrations.government_mock import get_alerts
from app.ml.pipeline import get_ml_pipeline
from app.nlp.pipeline import get_nlp_pipeline

router = APIRouter()
_dialog_manager = DialogueManager()


def _process_message(text: str, session_id: str) -> str:
    """NLP → ML → Dialogue → response text."""
    nlp = get_nlp_pipeline()
    ml = get_ml_pipeline()
    nlp_out = nlp.process(text)
    ml_out = ml.run(text, nlp_out["tokens"], nlp_out["embedding"])
    action = _dialog_manager.next_action(
        session_id, ml_out["intent"], ml_out["symptoms"],
        ml_out["top3_diseases"], ml_out["is_emergency"], lang=nlp_out["lang"],
        user_message=text,
    )
    return _dialog_manager.build_response(
        session_id, action, ml_out["intent"], ml_out["symptoms"],
        ml_out["top3_diseases"], ml_out["is_emergency"], lang=nlp_out["lang"],
        use_ollama_phrasing=True,
        user_message=text,
        user_location=None,  # Can be extracted from WhatsApp/SMS location data if available
    )


def _get_alerts_for_region(region: Optional[str], limit: int = 5) -> list:
    """Alerts for region (or all if no region)."""
    all_alerts = get_alerts(limit=limit * 2)
    if not region or not region.strip():
        return all_alerts[:limit]
    r = region.strip()
    out = [a for a in all_alerts if region_matches(a.get("region") or "", r)]
    return (out or all_alerts)[:limit]


def _format_alerts_reply(alerts: list) -> str:
    if not alerts:
        return "No active health alerts for this area right now. You can ask me any health question."
    return "Live health alerts:\n\n" + "\n\n".join(format_alert_message(a) for a in alerts)


def _get_reply(text: str, session_id: str, phone: str, channel: str) -> str:
    """Real-time reply: handle alerts/subscribe commands or run chat."""
    t = (text or "").strip().lower()
    # "alerts" or "alerts Bengaluru" -> return current alerts for region
    if t.startswith("alerts"):
        region = text.strip()[6:].strip() or None
        alerts = _get_alerts_for_region(region)
        return _format_alerts_reply(alerts)
    # "subscribe" or "subscribe Bengaluru" -> add subscription and send current alerts
    if t.startswith("subscribe"):
        region = (text.strip()[9:].strip() or "National").strip()
        added = add_subscription(phone, region, channel)
        alerts = _get_alerts_for_region(region)
        sub_list = get_subscriptions()
        send_alerts_to_subscribers(
            [{"phone": phone, "region": region, "channel": channel}],
            alerts,
        )
        if added:
            return f"You're subscribed to health alerts for: {region}. Current alerts sent to you. Reply with 'alerts' or 'alerts <place>' anytime for live alerts."
        return f"Subscription updated to: {region}. Current alerts sent. Reply 'alerts' for live alerts."
    # "unsubscribe" -> remove
    if t.startswith("unsubscribe"):
        from app.integrations.alert_subscriptions import remove_subscription
        remove_subscription(phone, channel)
        return "You're unsubscribed from health alerts. Reply 'subscribe <region>' to get alerts again."
    # Otherwise: chat (medical Q&A)
    return _process_message(text, session_id)


# ----- WhatsApp Webhook -----

@router.get("/webhook/whatsapp")
def whatsapp_webhook_verify(
    request: Request,
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
) -> Response:
    """GET: WhatsApp verification (hub.mode=subscribe, hub.verify_token, hub.challenge)."""
    from app.config import WHATSAPP_VERIFY_TOKEN
    if hub_mode == "subscribe" and hub_verify_token == WHATSAPP_VERIFY_TOKEN and hub_challenge:
        return Response(content=hub_challenge, media_type="text/plain")
    return Response(status_code=403)


class WhatsAppIncoming(BaseModel):
    """Minimal incoming webhook payload (field names depend on provider)."""
    pass  # Parsed from request body in post


@router.post("/webhook/whatsapp")
async def whatsapp_webhook_post(request: Request) -> dict:
    """
    POST: Incoming WhatsApp message. Extract text and sender id, process, send reply via Meta API.
    Supports Meta Cloud API (WhatsApp Business Platform).
    """
    try:
        body = await request.json()
    except Exception:
        return {"status": "error", "message": "Invalid JSON"}

    from app.config import WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID
    import requests as req

    # Generic extraction (Meta Cloud API format)
    entries = body.get("entry", [])
    for entry in entries:
        changes = entry.get("changes", [])
        for change in changes:
            value = change.get("value", {})
            messages = value.get("messages", [])
            for msg in messages:
                text = (msg.get("text") or {}).get("body") or msg.get("body") or ""
                sender_id = msg.get("from") or value.get("contacts", [{}])[0].get("wa_id") or "unknown"
                if text:
                    reply_text = _get_reply(text, session_id=f"wa_{sender_id}", phone=sender_id, channel="whatsapp")
                    # Send reply via Meta Cloud API (real-time)
                    if WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID:
                        try:
                            send_url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
                            headers = {
                                "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                                "Content-Type": "application/json",
                            }
                            payload = {
                                "messaging_product": "whatsapp",
                                "to": sender_id,
                                "text": {"body": reply_text},
                            }
                            req.post(send_url, json=payload, headers=headers, timeout=10)
                        except Exception as e:
                            print(f"WhatsApp send error: {e}")
    return {"status": "ok"}


# ----- SMS Gateway -----

class SMSRequest(BaseModel):
    """Incoming SMS from gateway (e.g. Twilio-style). Accept JSON with From, Body."""
    From: str = Field(..., alias="From", description="Sender phone number")
    Body: str = Field(..., alias="Body", description="SMS body text")

    model_config = {"populate_by_name": True}


@router.post("/webhook/sms")
def sms_gateway(req: SMSRequest) -> dict:
    """
    SMS gateway: receive From + Body, return reply. Gateway sends reply as SMS (real-time).
    Commands: 'alerts', 'alerts <region>', 'subscribe <region>', 'unsubscribe', or any health question.
    """
    session_id = f"sms_{req.From}"
    phone = re.sub(r"^\+\s*", "", (req.From or "").replace(" ", ""))
    reply_text = _get_reply(req.Body or "", session_id, phone=phone, channel="sms")
    return {"reply": reply_text, "To": req.From}
