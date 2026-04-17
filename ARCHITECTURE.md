## High-Level System Architecture

HealthBot is designed as a **FastAPI-based API/front-end** that can be decomposed into three
microservice-friendly domains:

- **API Gateway / Chat Service**
  - Entry point: `main.py` (`FastAPI` app).
  - Responsibilities:
    - HTTP API (`app/api/routes.py`, `app/api/auth.py`, `app/api/admin.py`, `app/api/webhooks.py`).
    - Web UI serving (`/` and `/static`).
    - Authentication and session management.
    - Periodic alert push to WhatsApp/SMS subscribers.
- **ML & Knowledge Service**
  - Core modules:
    - NLP: `app/nlp/language_detection.py`, `app/nlp/tokenization.py`, `app/nlp/embeddings.py`,
      `app/nlp/pipeline.py`.
    - ML: `app/ml/intent_classifier.py`, `app/ml/disease_classifier.py`,
      `app/ml/symptom_extractor.py`, `app/ml/pipeline.py`.
    - KB: `app/knowledge_base/graph.py`, `app/knowledge_base/retriever.py`.
    - Dialogue: `app/dialog/manager.py`.
  - Responsibilities:
    - Generate IndicBERT/mBERT embeddings (384‑dim).
    - Predict intents and diseases using ensemble models.
    - Look up structured medical knowledge and build grounded, multilingual responses.
- **Alert & External Data Service**
  - Core modules:
    - Outbreak detection: `app/outbreak/case_records.py`, `app/outbreak/detector.py`.
    - External portals: `app/integrations/health_portals.py`, `app/integrations/government_mock.py`.
    - Alert delivery: `app/integrations/alert_subscriptions.py`, `app/integrations/alert_sender.py`.
  - Responsibilities:
    - Pull and normalise government/WHO/ICMR (or proxy) health feeds.
    - Combine external alerts with locally detected outbreaks.
    - Push alerts to subscribed WhatsApp/SMS users.

In the current repository, these domains run **within a single FastAPI service**, but the module
boundaries and configuration are chosen so they can be separated into independent services later.

## Service Decomposition (Microservice-Ready)

An example decomposition:

- **`api-gateway` service**
  - Hosts `main:app` (FastAPI) and static web assets.
  - Public endpoints: `/api/chat`, `/api/chat-offline`, `/api/health`, `/api/alerts`,
    `/api/review-document`, etc.
  - Calls (when decomposed):
    - `ml-service` via REST/gRPC for ML+KB inference.
    - `alert-service` for subscription management and alert previews.
- **`ml-service`**
  - Exposes endpoints such as:
    - `POST /nlp/process` – language detection + tokens + embeddings.
    - `POST /ml/predict` – intents, symptoms, disease top‑k.
    - `POST /dialog/respond` – dialogue manager response generation (optionally with LLM).
  - Can be scaled independently (GPU/CPU) and colocated with heavy transformer models.
- **`alert-service`**
  - Periodically polls:
    - External feeds (`app/integrations/health_portals.py`).
    - Internal case records (`app/outbreak/case_records.py`).
  - Produces a consolidated alert stream; API gateway subscribes and pushes to end users.

In the monolithic Docker deployment, these three domains still live inside a single
FastAPI process. A future deployment can lift `ml-service` and `alert-service` into
separate containers by:

- Creating dedicated FastAPI apps exposing the `/nlp/*`, `/ml/*`, `/dialog/*`, and
  `/alerts/*` interfaces.
- Using the same Docker image but different entrypoints/commands per service (or
  separate images built from the same repository).

Interfaces between services can be expressed as JSON over HTTP (FastAPI in all services)
and later tightened to gRPC if needed.

## Containerisation

### Dockerfile (monolithic app)

The included `Dockerfile` builds a single container running the full HealthBot API:

- Based on `python:3.10-slim`.
- Installs `requirements.txt` and the ML/NLP dependencies.
- Exposes port `8000` and launches `uvicorn main:app`.
- Loads configuration from environment variables defined in `.env` or the orchestrator.

### docker-compose (development / monolithic production)

`docker-compose.yml` wires together:

- `api` – the HealthBot FastAPI + web front-end container.
- `ollama` – optional LLM runtime (`ollama/ollama` image) for response phrasing.

The compose file is designed so that:

- `api` reads the Ollama base URL from `OLLAMA_BASE_URL`.
- Auth/admin credentials and portal URLs come from `.env` with sensible defaults.
- External integrations (WhatsApp, SMS, MoHFW/WHO/ICMR feeds) are toggled purely through
  environment variables (`WHATSAPP_*`, `SMS_*`, `HEALTH_PORTALS_ENABLED`, etc.).

This layout provides a clear path to:

- Move `ml-service` into its own container by factoring `MLPipeline`/`DialogueManager`
  routes into a separate FastAPI app.
- Move `alert-service` into a lightweight polling/worker container that hits the
  same data stores (cases, subscriptions) but exposes only alert APIs or pushes via
  WhatsApp/SMS.
- Deploy all services to Kubernetes or a similar orchestrator using the same images and environment
  variables.

### Implementation Mapping (Paper → Code)

- **API and web front-end** → `main.py`, `app/api/*.py`, static assets under `app/static` (if present).
- **ML and knowledge microservice** (conceptual) → NLP/ML/KB/dialogue modules under `app/nlp`, `app/ml`,
  `app/knowledge_base`, `app/dialog`.
- **Outbreak and external portal integration** → `app/outbreak/*.py`,
  `app/integrations/government_mock.py`, `app/integrations/health_portals.py`.
- **Deployment & containerisation** → `Dockerfile`, `docker-compose.yml`, configuration via `app/config.py`.

