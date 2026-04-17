"""
HealthBot - Multilingual Public Health Chatbot.
Default: Web UI + API (python main.py). Use --cli for terminal-only chat.
Real-time: chat, alerts, WhatsApp/SMS answers and alert push.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.api.webhooks import router as webhooks_router
from app.api.auth import router as auth_router
from app.api.admin import router as admin_router
from app.config import API_HOST, API_PORT

# Real-time alert push interval (seconds)
ALERT_PUSH_INTERVAL = 300  # 5 minutes


def _run_notify_subscribers() -> None:
    """Sync: send current alerts (incl. outbreak detection) to all WhatsApp/SMS subscribers."""
    try:
        from app.integrations.alert_sender import send_alerts_to_subscribers
        from app.integrations.alert_subscriptions import get_subscriptions
        from app.outbreak.detector import merge_with_existing_alerts
        from app.integrations.government_mock import get_alerts
        subs = get_subscriptions()
        if subs:
            existing = get_alerts(limit=20)
            alerts = merge_with_existing_alerts(existing, limit=20)
            send_alerts_to_subscribers(subs, alerts)
    except Exception as e:
        print(f"Alert push error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Init DB, create default admin if none, start alert push task."""
    from app.db import init_db, SessionLocal
    from app.models.user import User
    from app.api.auth import _hash_password, validate_password_policy
    from app.config import ADMIN_EMAIL, ADMIN_PASSWORD

    init_db()
    # Validate admin password early so startup fails with a clear message
    try:
        validate_password_policy(ADMIN_PASSWORD)
    except ValueError as exc:
        # Fail fast with a descriptive error instead of a low-level bcrypt/passlib trace
        raise RuntimeError(f"Invalid ADMIN_PASSWORD configuration: {exc}") from exc

    db = SessionLocal()
    if db.query(User).count() == 0:
        admin = User(email=ADMIN_EMAIL, password_hash=_hash_password(ADMIN_PASSWORD), role="admin")
        db.add(admin)
        db.commit()
        print(f"Created default admin: {ADMIN_EMAIL}")
    db.close()

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()

    async def run_periodic():
        await asyncio.sleep(30)  # first run after 30s so server is ready
        while not stop.is_set():
            loop.run_in_executor(None, _run_notify_subscribers)
            try:
                await asyncio.wait_for(stop.wait(), timeout=ALERT_PUSH_INTERVAL)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(run_periodic())
    yield
    stop.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="HealthBot API",
    description="Medical info, prescription review, real-time alerts via WhatsApp/SMS.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """
    Ensure clients always receive JSON (not plain-text "Internal Server Error")
    for unhandled exceptions.
    """
    try:
        detail = str(exc)[:500] if exc else "Internal Server Error"
    except Exception:
        detail = "Internal Server Error"
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error": detail})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(webhooks_router, prefix="/api", tags=["webhooks"])
app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(admin_router, prefix="/api", tags=["admin"])

# Web UI
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def root():
    """Serve the HealthBot web UI (chat, upload, alerts)."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"service": "HealthBot", "docs": "/docs", "health": "/api/health"}


def run_ollama_chat() -> None:
    """Ollama-first: terminal chat using NLP → ML → Dialogue → Ollama."""
    from run_ollama_chat import main as ollama_main
    ollama_main()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_ollama_chat()
    else:
        import uvicorn
        print(f"Starting HealthBot API on http://{API_HOST}:{API_PORT}")
        print(f"  Web UI:    http://localhost:{API_PORT}/")
        print(f"  API docs:  http://localhost:{API_PORT}/docs")
        print(f"  WhatsApp:  http://{API_HOST}:{API_PORT}/api/webhook/whatsapp")
        print(f"  SMS:       http://{API_HOST}:{API_PORT}/api/webhook/sms")
        uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
