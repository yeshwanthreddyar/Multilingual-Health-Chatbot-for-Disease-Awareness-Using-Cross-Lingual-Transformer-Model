# HealthBot – Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Install Ollama
```bash
# Windows/Mac/Linux: https://ollama.ai
# After install:
ollama pull llama3
ollama serve
```

### Step 2: Install Python dependencies
```bash
cd healthbot
pip install -r requirements.txt
```

### Step 3: Choose your interface

#### Option A: Terminal Chat (easiest)
```bash
python main.py
```
**You:** I have fever and cough  
**HealthBot:** Prevention tips for Influenza...

Type `quit` to exit.

#### Option B: WhatsApp + SMS + API
```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env with your credentials (see below)
# 3. Start server
python main.py --api

# 4. Test API
curl http://localhost:8000/api/health
```

Visit `http://localhost:8000/docs` for API documentation.

---

## 📱 WhatsApp Setup (5 minutes)

### Prerequisites
- Meta developer account (free): https://developers.facebook.com/
- WhatsApp Business app in Meta dashboard

### Steps
1. **Create Meta Business App**
   - Go to https://developers.facebook.com/apps/create/
   - Choose "Business" type
   - Add "WhatsApp" product

2. **Get credentials**
   - Phone Number ID (from WhatsApp settings)
   - Access Token (from app settings)

3. **Configure webhook**
   - For testing: Use ngrok
     ```bash
     ngrok http 8000
     ```
   - Webhook URL: `https://your-ngrok-url.ngrok.io/api/webhook/whatsapp`
   - Verify Token: (set same in `.env` as `WHATSAPP_VERIFY_TOKEN`)
   - Subscribe to: `messages`

4. **Update `.env`**
   ```bash
   WHATSAPP_VERIFY_TOKEN=your_verify_token_here
   WHATSAPP_ACCESS_TOKEN=your_meta_access_token
   WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
   ```

5. **Start server**
   ```bash
   python main.py --api
   ```

6. **Test**
   - Send a WhatsApp message to your business number
   - HealthBot replies in same language via Ollama!

---

## 📨 SMS Setup (Twilio – 5 minutes)

### Prerequisites
- Twilio account (free trial): https://www.twilio.com/try-twilio
- Twilio phone number

### Steps
1. **Get credentials**
   - Account SID (from console dashboard)
   - Auth Token (from console dashboard)
   - Phone number (from "Phone Numbers" section)

2. **Configure webhook**
   - In Twilio console → Phone Numbers → your number
   - Messaging webhook: `https://your-ngrok-url.ngrok.io/api/webhook/sms`
   - Method: HTTP POST

3. **Update `.env`**
   ```bash
   SMS_ACCOUNT_SID=your_twilio_sid
   SMS_AUTH_TOKEN=your_twilio_auth_token
   SMS_FROM_NUMBER=+1234567890
   ```

4. **Start server**
   ```bash
   python main.py --api
   ```

5. **Test**
   - Send SMS to your Twilio number
   - HealthBot replies via Ollama!

---

## 🧪 Test Everything

```bash
python test_system.py
```

Expected output:
```
[OK] All imports successful
[OK] NLP pipeline
[OK] ML pipeline
[OK] Dialogue manager
[OK] Knowledge base
[WARN] Ollama (check if running)
[OK] FastAPI app loaded

Results: 7/7 tests passed
[OK] All systems operational!
```

---

## 🌍 Test Multilingual

Try these in terminal chat (`python main.py`):

| Language | Text | Expected |
|----------|------|----------|
| English | I have fever | ✓ Prevention tips |
| Hindi | मुझे बुखार है | ✓ Hindi response |
| Telugu | నాకు జ్వరం ఉంది | ✓ Telugu response |
| Tamil | எனக்கு காய்ச்சல் | ✓ Tamil response |

---

## 🐳 Docker (Production)

```bash
# Build and run
docker-compose up -d

# Pull Ollama model
docker exec -it healthbot-ollama-1 ollama pull llama3

# Test
curl http://localhost:8000/api/health
```

WhatsApp/SMS webhooks work with Docker URL.

---

## 📚 Documentation

- **Full setup:** [README.md](README.md)
- **Deployment:** [DEPLOYMENT.md](DEPLOYMENT.md)
- **Summary:** [SUMMARY.md](SUMMARY.md)
- **API docs:** http://localhost:8000/docs (when running `--api`)

---

## 🆘 Troubleshooting

### Ollama not responding
```bash
# Check if running
curl http://localhost:11434/api/generate -d '{"model":"llama3","prompt":"hello"}'

# If not, start it
ollama serve
```

### WhatsApp webhook verification fails
- Check verify token matches in `.env` and Meta dashboard
- Check webhook URL is HTTPS (use ngrok for testing)
- Check server is running: `python main.py --api`

### SMS not working
- Check Twilio webhook URL is correct
- Check `.env` credentials
- Check Twilio console logs for errors

### Import errors
```bash
pip install -r requirements.txt
```

### "Negative values in data" error
- Already fixed with GaussianNB in latest code
- If still occurs, update: `pip install -U scikit-learn`

---

## ✅ Success Checklist

- [ ] Ollama installed and `llama3` model pulled
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Terminal chat works: `python main.py`
- [ ] System tests pass: `python test_system.py`
- [ ] (Optional) WhatsApp setup complete
- [ ] (Optional) SMS setup complete
- [ ] (Optional) Docker deployment

---

**Ready to deploy? See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup.**
