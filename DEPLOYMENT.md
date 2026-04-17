# HealthBot Deployment Guide

## Overview

HealthBot can be accessed via:
1. **Terminal (CLI)** – Default, using Ollama
2. **WhatsApp** – Via Meta Business API webhooks
3. **SMS** – Via Twilio or other SMS gateway
4. **REST API** – For custom integrations

---

## Prerequisites

1. **Python 3.9+** and dependencies: `pip install -r requirements.txt`
2. **Ollama** running locally or remotely with `llama3` model
   ```bash
   ollama pull llama3
   ollama serve
   ```
3. **Publicly accessible server** (for WhatsApp/SMS webhooks)
   - Use ngrok for testing: `ngrok http 8000`
   - Or deploy to cloud (Azure, AWS, GCP, Heroku, etc.)

---

## 1. Terminal (CLI) – Default

**Run locally:**
```bash
python main.py
```

Chat in terminal. Type messages in any of 15 Indian languages.

---

## 2. WhatsApp Access (Meta Business API)

### Setup

1. **Create Meta Business App**
   - Go to [Meta for Developers](https://developers.facebook.com/)
   - Create app → Choose "Business" → Add "WhatsApp" product
   - Get: Phone Number ID, Access Token

2. **Configure webhook**
   - Webhook URL: `https://your-domain.com/api/webhook/whatsapp`
   - Verify Token: Set in `.env` as `WHATSAPP_VERIFY_TOKEN`
   - Subscribe to: `messages`

3. **Environment variables** (`.env`)
   ```bash
   WHATSAPP_VERIFY_TOKEN=your_verify_token
   WHATSAPP_ACCESS_TOKEN=your_meta_access_token
   WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
   ```

4. **Start API server**
   ```bash
   python main.py --api
   ```

5. **Test**
   - Send a message to your WhatsApp Business number
   - HealthBot responds via Ollama in the same language

### Testing with ngrok

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start HealthBot API
python main.py --api

# Terminal 3: Expose to internet
ngrok http 8000

# Use ngrok URL in Meta webhook settings: https://abc123.ngrok.io/api/webhook/whatsapp
```

---

## 3. SMS Access (Twilio or other gateway)

### Setup with Twilio

1. **Create Twilio account** → Get phone number
2. **Configure webhook**
   - Messaging webhook: `https://your-domain.com/api/webhook/sms`
   - Method: POST
   - Content-Type: JSON (or form-encoded, adjust code if needed)

3. **Environment variables** (`.env`)
   ```bash
   SMS_ACCOUNT_SID=your_twilio_account_sid
   SMS_AUTH_TOKEN=your_twilio_auth_token
   SMS_FROM_NUMBER=+1234567890
   ```

4. **Start API server**
   ```bash
   python main.py --api
   ```

5. **Test**
   - Send SMS to your Twilio number
   - HealthBot replies via Ollama

### Testing with ngrok

```bash
ngrok http 8000
# Use ngrok URL in Twilio: https://abc123.ngrok.io/api/webhook/sms
```

---

## 4. REST API (Custom integrations)

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat: `{"message": "...", "session_id": "..."}` |
| `/api/alerts` | GET | Outbreak alerts (mock) |
| `/api/vaccines` | GET | Vaccination schedules |
| `/api/advisories` | GET | Public health advisories |
| `/api/health` | GET | Health check |

**Example:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "मुझे बुखार है", "session_id": "user123"}'
```

---

## 5. Production Deployment

### Option A: Docker (recommended)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV OLLAMA_BASE_URL=http://ollama:11434
CMD ["python", "main.py", "--api"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    command: serve

  healthbot:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - ollama
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434

volumes:
  ollama_data:
```

**Run:**
```bash
docker-compose up -d
docker exec -it healthbot-ollama-1 ollama pull llama3
```

### Option B: Cloud deployment

**Azure App Service:**
```bash
az webapp up --name healthbot-india --runtime PYTHON:3.10
```

**AWS Elastic Beanstalk:**
```bash
eb init -p python-3.10 healthbot
eb create healthbot-env
```

**Google Cloud Run:**
```bash
gcloud run deploy healthbot --source . --allow-unauthenticated
```

**Note:** For cloud, either:
- Run Ollama in a separate VM and set `OLLAMA_BASE_URL` to its URL
- Or use cloud Ollama alternatives (if available)

---

## 6. Security & Compliance

- **HTTPS required** for WhatsApp/SMS webhooks
- **Verify tokens** for all webhooks
- **Rate limiting** recommended (e.g. with FastAPI middleware)
- **Data privacy**: No PHI stored; session state in memory only
- **HIPAA/GDPR**: If needed, add encryption + audit logs

---

## 7. Monitoring

- **Health check**: `GET /api/health`
- **Logs**: Check Ollama logs, API logs
- **Metrics**: Track latency (target ≤ 2s), accuracy (≥ 80%)

---

## 8. Multilingual Testing

Test with messages in all 15 languages:
- Hindi: "मुझे बुखार है"
- Bengali: "আমার জ্বর আছে"
- Telugu: "నాకు జ్వరం ఉంది"
- Tamil: "எனக்கு காய்ச்சல் இருக்கிறது"
- Kannada: "ನನಗೆ ಜ್ವರ ಬಂದಿದೆ"
- Malayalam: "എനിക്ക് പനി ഉണ്ട്"
- (etc.)

---

## Support

For issues or questions, refer to:
- README.md
- PDF architecture document
- Ollama docs: https://ollama.ai
- Meta WhatsApp API: https://developers.facebook.com/docs/whatsapp
- Twilio SMS: https://www.twilio.com/docs/sms
