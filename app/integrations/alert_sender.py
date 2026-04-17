"""
Send disease alerts to subscribers via WhatsApp and SMS.
Uses Meta WhatsApp Cloud API and Twilio-style SMS.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import requests

from app.config import (
    SMS_ACCOUNT_SID,
    SMS_AUTH_TOKEN,
    SMS_FROM_NUMBER,
    WHATSAPP_ACCESS_TOKEN,
    WHATSAPP_PHONE_NUMBER_ID,
)


def _normalize_phone(phone: str) -> str:
    """Strip spaces and ensure country code for WhatsApp (no leading + in API)."""
    p = re.sub(r"\s+", "", phone)
    if p.startswith("+"):
        p = p[1:]
    return p


def send_whatsapp(to_phone: str, body: str) -> bool:
    """Send one WhatsApp text message via Meta Cloud API."""
    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        return False
    to = _normalize_phone(to_phone)
    try:
        url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        resp = requests.post(
            url,
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "text": {"body": body[:4096]},
            },
            headers={
                "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        return resp.status_code == 200
    except Exception:
        return False


def send_sms(to_phone: str, body: str) -> bool:
    """Send one SMS via Twilio API."""
    if not SMS_ACCOUNT_SID or not SMS_AUTH_TOKEN or not SMS_FROM_NUMBER:
        return False
    to = _normalize_phone(to_phone)
    if not to.startswith("+"):
        to = "+" + to
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{SMS_ACCOUNT_SID}/Messages.json"
        resp = requests.post(
            url,
            data={
                "To": to,
                "From": SMS_FROM_NUMBER,
                "Body": body[:1600],
            },
            auth=(SMS_ACCOUNT_SID, SMS_AUTH_TOKEN),
            timeout=15,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False


def format_alert_message(alert: Dict[str, Any]) -> str:
    """Single alert as short message for WhatsApp/SMS."""
    title = alert.get("title", "Health Alert")
    region = alert.get("region", "")
    summary = alert.get("summary", "")
    source = alert.get("source", "")
    return f"[HealthBot Alert] {title}\nRegion: {region}\n{summary}\nSource: {source}"


def region_matches(alert_region: str, user_region: str) -> bool:
    """True if alert applies to user's region (National/Multiple states always match)."""
    ar = (alert_region or "").strip().lower()
    ur = (user_region or "").strip().lower()
    if not ur:
        return True
    if ar in ("national", "multiple states"):
        return True
    return ur in ar or ar in ur


def send_alerts_to_subscribers(
    subscriptions: List[dict],
    alerts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    For each subscription, find alerts matching their region and send via WhatsApp or SMS.
    Returns counts: sent_whatsapp, sent_sms, skipped (no alerts), errors.
    """
    sent_whatsapp = 0
    sent_sms = 0
    skipped = 0
    errors = 0
    for sub in subscriptions:
        phone = sub.get("phone", "")
        region = sub.get("region", "")
        channel = (sub.get("channel") or "whatsapp").lower()
        matching = [a for a in alerts if region_matches(a.get("region", ""), region)]
        if not matching:
            skipped += 1
            continue
        body = "\n\n".join(format_alert_message(a) for a in matching[:5])
        if channel == "sms":
            ok = send_sms(phone, body)
            if ok:
                sent_sms += 1
            else:
                errors += 1
        else:
            ok = send_whatsapp(phone, body)
            if ok:
                sent_whatsapp += 1
            else:
                errors += 1
    return {"sent_whatsapp": sent_whatsapp, "sent_sms": sent_sms, "skipped": skipped, "errors": errors}
