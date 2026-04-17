# HealthBot Features Summary

## 6. Offline Chat (No Ollama, No External APIs)

- **`run_offline_chat.py`** – Terminal chat using only trained models + knowledge base
- **`POST /api/chat-offline`** – Same as `/chat` but never calls Ollama
- Dialogue manager uses `use_ollama_phrasing=False` for template-based responses

**Usage:**
```bash
python run_offline_chat.py
```

---

## 5. Outbreak Tracking + Alerts

- **`app/outbreak/case_records.py`** – Stores cases in `data/cases.json`
- **`app/outbreak/detector.py`** – Detects outbreaks when cases exceed threshold (5+ in 7 days)
- Cases recorded when users report symptoms (with location)
- Alerts merged with WHO/MoHFW mock alerts and pushed to subscribers
- **`GET /api/cases`** – List cases (region, disease, days filters)

---

## 4. WHO + Indian Health Updates (File-Based)

- **`data/who_updates.json`** – WHO updates (edit manually)
- **`data/india_health_updates.json`** – MoHFW/ICMR updates (edit manually)
- **`app/data/load_health_updates.py`** – Loads and caches updates
- Updates included in prevention responses
- **`GET /api/health-updates`** – List updates (disease, region filters)

---

## 1. Login / Register + Admin

- **`app/db.py`** – SQLite database
- **`app/models/user.py`** – User model (email, password_hash, role)
- **`app/api/auth.py`** – Register, login, JWT
- Default admin: `admin@healthbot.local` / `admin123` (set via `ADMIN_EMAIL`, `ADMIN_PASSWORD`)
- Login page on `/` – must log in to use chat

---

## 2. Logging + Admin Page

- **`app/models/log.py`** – Log model (message, response, intent, is_health_related)
- All `/chat` and `/chat-offline` requests logged
- **`static/admin.html`** – Admin dashboard (admin only)
- **`GET /api/admin/logs`** – List logs
- **`GET /api/admin/logs/stats`** – Intent stats
- **`GET /api/admin/users`** – List users

---

## 3. Training from Logs

- **`train_from_logs.py`** – Trains intent classifier from health-related logs
- Merges with `data/processed/combined_dataset.json`
- Saves to `models/intent_classifier_trained_from_logs.pkl`
- **`POST /api/admin/retrain-intent`** – Admin button to run retrain
- ML pipeline loads trained model if it exists

**Usage:**
```bash
python train_from_logs.py
```

---

## Quick Start

1. Install: `pip install -r requirements.txt`
2. Run: `python main.py`
3. Login: `admin@healthbot.local` / `admin123`
4. Admin: Click "Admin" in header → view logs, retrain
5. Offline chat: `python run_offline_chat.py`

## Environment Variables (.env)

```
AUTH_SECRET=change_this_secret_in_production
ADMIN_EMAIL=admin@healthbot.local
ADMIN_PASSWORD=admin123
```
