"""
Shared in-memory store for alert subscriptions (WhatsApp/SMS).
Used by API routes and webhooks so users can subscribe via UI or by messaging "subscribe <region>".
"""
from __future__ import annotations

from typing import List


_subscriptions: List[dict] = []


def add_subscription(phone: str, region: str, channel: str = "whatsapp") -> bool:
    """Add or update subscription. Returns True if added/new."""
    channel = (channel or "whatsapp").lower()
    sub = {"phone": phone, "region": region.strip(), "channel": channel}
    for i, s in enumerate(_subscriptions):
        if s.get("phone") == phone and s.get("channel") == channel:
            _subscriptions[i] = sub
            return False
    _subscriptions.append(sub)
    return True


def get_subscriptions() -> List[dict]:
    """Return all subscriptions (for notify job and API)."""
    return list(_subscriptions)


def remove_subscription(phone: str, channel: str) -> bool:
    """Remove subscription by phone and channel. Returns True if removed."""
    global _subscriptions
    before = len(_subscriptions)
    _subscriptions = [s for s in _subscriptions if not (s.get("phone") == phone and (s.get("channel") or "whatsapp") == channel)]
    return len(_subscriptions) < before
