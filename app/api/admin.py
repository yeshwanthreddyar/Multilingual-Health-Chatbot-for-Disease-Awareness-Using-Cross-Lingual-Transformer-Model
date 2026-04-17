"""
Admin API - logs, users. Requires admin role.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.auth import get_current_user
from app.db import SessionLocal, init_db
from app.models.user import User
from app.models.log import Log

router = APIRouter()


def _require_admin(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = get_current_user(authorization.split(" ", 1)[1])
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")
    return user


@router.get("/admin/logs")
def list_logs(
    limit: int = 100,
    offset: int = 0,
    health_only: bool = False,
    admin: dict = Depends(_require_admin),
) -> list:
    """List chat logs. Admin only."""
    init_db()
    db = SessionLocal()
    q = db.query(Log).order_by(Log.created_at.desc())
    if health_only:
        q = q.filter(Log.is_health_related == True)
    rows = q.offset(offset).limit(limit).all()
    db.close()
    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "session_id": r.session_id,
            "message": r.message,
            "response": (r.response or "")[:200],
            "intent": r.intent,
            "is_health_related": r.is_health_related,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/admin/logs/stats")
def logs_stats(admin: dict = Depends(_require_admin)) -> dict:
    """Aggregate stats by intent. Admin only."""
    init_db()
    db = SessionLocal()
    rows = db.query(Log.intent, Log.is_health_related).all()
    db.close()
    intents = {}
    health_count = 0
    for intent, is_health in rows:
        intents[intent or "unknown"] = intents.get(intent or "unknown", 0) + 1
        if is_health:
            health_count += 1
    return {"by_intent": intents, "health_related_count": health_count, "total": len(rows)}


@router.post("/admin/retrain-intent")
def retrain_intent(admin: dict = Depends(_require_admin)) -> dict:
    """Run train_from_logs.py to retrain intent classifier. Admin only."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "train_from_logs.py"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        if result.returncode == 0:
            # Clear ML pipeline cache so next request loads new model
            import app.ml.pipeline as pl
            pl._ml_pipeline = None
        return {"status": "ok" if result.returncode == 0 else "error", "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/admin/users")
def list_users(admin: dict = Depends(_require_admin)) -> list:
    """List users. Admin only."""
    init_db()
    db = SessionLocal()
    rows = db.query(User).all()
    db.close()
    return [
        {"id": r.id, "email": r.email, "role": r.role, "created_at": str(r.created_at)}
        for r in rows
    ]
