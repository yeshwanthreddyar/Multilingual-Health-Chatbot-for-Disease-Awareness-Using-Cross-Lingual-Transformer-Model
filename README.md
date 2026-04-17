# HealthBot – Multilingual Public Health Chatbot

Multilingual public-health chatbot for rural and semi-urban India. **Educational and advisory only; not a diagnostic tool.**

## 📱 Access Methods

| Method | Command | Setup |
|--------|---------|-------|
| **Terminal (CLI)** | `python main.py` | ✓ Ollama only |
| **WhatsApp** | `python main.py --api` | Meta Business API ([guide](#whatsapp--sms-setup)) |
| **SMS** | `python main.py --api` | Twilio/other ([guide](#whatsapp--sms-setup)) |
| **REST API** | `python main.py --api` | None – `/api/chat` endpoint |

## Architecture (matches PDF)

```
Users (Mobile / Web / WhatsApp / SMS / Kiosk)
        ↓
Multilingual Text Input (15+ Indian Languages)
        ↓
NLP Processing Layer (Language Detection → Tokenization → Embedding: IndicBERT / mBERT)
        ↓
Intent Recognition & Symptom Classification
        ↓
ML / DL (IndicBERT primary, mBERT fallback, Ensemble NB + RF + GB)
        ↓
Context-Aware Dialogue Manager (Ollama-assisted phrasing only)
        ↓
Medical Knowledge Base (Retrieval ONLY)
        ↓
WHO / MoHFW / ICMR APIs (Mocked) → Response Generation → Multilingual Output
```

## Project structure

```
healthbot/
├── app/
│   ├── api/          # REST, WhatsApp webhook, SMS gateway
│   ├── nlp/          # Language detection, tokenization, embeddings
│   ├── ml/           # Intent, symptom extraction, disease ensemble
│   ├── dialog/       # Dialogue manager (multi-turn, Ollama phrasing)
│   ├── knowledge_base/  # Disease graph, symptom–disease, prevention/vaccines
│   ├── integrations/   # Mock /alerts, /vaccines, /advisories
│   └── llm/          # ollama_client.py (no SDKs, no cloud)
├── data/
├── models/
├── evaluation/       # Accuracy, latency, language-wise report
├── main.py
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Ollama (required)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3
ollama serve
```

### 3. Run HealthBot

**Option A: Terminal chat (default)**
```bash
python main.py
```
Type in any of 15 Indian languages; get Ollama-powered responses.

**Option B: WhatsApp + SMS + REST API**
```bash
python main.py --api
```
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

For WhatsApp/SMS setup, see **[DEPLOYMENT.md](DEPLOYMENT.md)**

## WhatsApp & SMS Setup

**WhatsApp (Meta Business API):**
1. Get Phone Number ID and Access Token from [Meta for Developers](https://developers.facebook.com/)
2. Set webhook: `https://your-domain.com/api/webhook/whatsapp`
3. Configure `.env` (see `.env.example`)
4. Run: `python main.py --api`

**SMS (Twilio):**
1. Get Twilio phone number and credentials
2. Set webhook: `https://your-domain.com/api/webhook/sms`
3. Configure `.env`
4. Run: `python main.py --api`

**Full instructions:** See [DEPLOYMENT.md](DEPLOYMENT.md)

---

## API Endpoints (when running `--api`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat: `{"message": "...", "session_id": "..."}` |
| `/api/webhook/whatsapp` | GET/POST | WhatsApp webhook (Meta Cloud API) |
| `/api/webhook/sms` | POST | SMS gateway (Twilio/other) |
| `/api/alerts` | GET | Outbreak alerts (mock) |
| `/api/vaccines` | GET | Vaccination schedules |
| `/api/advisories` | GET | Public health advisories |
| `/api/health` | GET | Health check |

## Language support (15+ Indian languages)

- **Detection:** Unicode script + character n-grams; output ISO code.
- **Respond in same language as user.** IndicBERT primary, mBERT fallback. Ollama never translates.

Supported codes: `en`, `hi`, `bn`, `te`, `ta`, `kn`, `ml`, `mr`, `gu`, `pa`, `or`, `as`, `ur`, `ne`, `kok`.

## Safety & ethics

- No diagnosis; no medicine prescription.
- Emergency symptoms (e.g. breathing difficulty, chest pain) → hospital recommendation.
- All responses append: *"This chatbot provides health education and awareness only. It does not provide medical diagnosis or treatment."*

## Evaluation targets (from PDF)

- Accuracy ≥ 80%
- Mean latency ≤ 2 seconds
- Language-wise accuracy report
- See `evaluation/evaluate.py` for placeholders.

## Licence

Use for education and awareness only; align with local health regulations.
