# Step-by-Step: Run HealthBot with Ollama

This guide walks you through installing Ollama and running the HealthBot project so the chatbot uses a local LLM for responses.

---

## Step 1: Install Ollama

1. **Download Ollama for Windows**
   - Go to: **https://ollama.ai**
   - Click **Download** and choose **Windows**
   - Run the installer (`OllamaSetup.exe`)

2. **Finish installation**
   - Follow the installer (Next → Install → Finish)
   - Ollama usually starts in the background after install
   - You may see an Ollama icon in the system tray

3. **Check that Ollama is installed**
   - Open a **new** PowerShell or Command Prompt
   - Run:
     ```powershell
     ollama --version
     ```
   - You should see a version number (e.g. `ollama version is 0.x.x`)

---

## Step 2: Pull the Llama 3 model

HealthBot uses the **llama3** model by default. Download it once:

1. **Open PowerShell or Command Prompt**

2. **Pull the model**
   ```powershell
   ollama pull llama3
   ```
   - This downloads the model (a few GB). Wait until it finishes.
   - When done you’ll see something like: `success`

3. **Optional: try the model**
   ```powershell
   ollama run llama3 "Hello"
   ```
   - Type a message and press Enter. Type `/bye` to exit.
   - This confirms the model works.

---

## Step 3: Make sure Ollama is running

Ollama often runs as a background service after install. If you’re not sure:

1. **Check if it’s already running**
   - In a browser, open: **http://localhost:11434**
   - If you see something like `Ollama is running`, it’s fine.

2. **If it’s not running**
   - Open a **new** PowerShell/Command Prompt
   - Run:
     ```powershell
     ollama serve
     ```
   - Leave this window open while you use HealthBot.
   - Or: start Ollama from the Start Menu / system tray if you have a shortcut.

---

## Step 4: Install HealthBot dependencies (if not done)

In the project folder:

1. **Open PowerShell**
2. **Go to the project**
   ```powershell
   cd "c:\Users\Yeshwanth Reddy A R\Downloads\healthbot"
   ```
3. **Install Python dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

---

## Step 5: Run HealthBot with Ollama

You can use either **terminal chat** or the **API server**. Both use Ollama when it’s running.

### Option A: Terminal chat (simplest)

1. **Keep Ollama running** (Step 3).
2. **In a new PowerShell**, go to the project:
   ```powershell
   cd "c:\Users\Yeshwanth Reddy A R\Downloads\healthbot"
   ```
3. **Start the chat**
   ```powershell
   python main.py
   ```
   - You’ll see: `HealthBot (Ollama) – Education & awareness only...`
4. **Chat**
   - Type e.g. `I have fever and cough` and press Enter.
   - HealthBot will use Ollama to phrase the reply.
5. **Exit**
   - Type `quit` or `exit` and press Enter.

### Option B: API server (for browser / WhatsApp / SMS)

1. **Keep Ollama running** (Step 3).
2. **In a new PowerShell**, go to the project:
   ```powershell
   cd "c:\Users\Yeshwanth Reddy A R\Downloads\healthbot"
   ```
3. **Start the API**
   ```powershell
   python main.py --api
   ```
   - You should see: `Starting HealthBot API on 0.0.0.0:8000`
4. **Use the API**
   - Open in browser: **http://localhost:8000/docs**
   - Or health check: **http://localhost:8000/api/health**
5. **Stop the server**
   - In the same window, press **Ctrl+C**.

---

## Step 6: Verify Ollama is used

- **Terminal chat:** If Ollama is running and `llama3` is pulled, replies will be full, natural-language answers. If Ollama is down, you may see something like `[Ollama unavailable: ...]`.
- **Quick test:** Run `python test_system.py` from the project folder. It will report whether Ollama is reachable.

---

## Optional: Use another model or URL

By default HealthBot uses:

- **URL:** `http://localhost:11434`
- **Model:** `llama3`

To change them:

1. **Copy the env example**
   ```powershell
   copy .env.example .env
   ```
2. **Edit `.env`** and set for example:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```
   (Use any model you’ve pulled with `ollama pull <model>`.)

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| `ollama` not found | Restart the terminal after installing Ollama. If needed, add Ollama’s install folder to PATH. |
| `[Ollama unavailable: ...]` | Start Ollama: run `ollama serve` in a separate window, or start it from the tray. |
| Model not found | Run `ollama pull llama3` (or the model name you set in `.env`). |
| Slow first reply | First request after starting Ollama loads the model; later ones are faster. |
| Port 11434 in use | Another Ollama instance may be running. Close it or change `OLLAMA_BASE_URL` in `.env` if you use a different port. |

---

## Summary checklist

- [ ] Ollama installed (`ollama --version` works)
- [ ] Model pulled: `ollama pull llama3`
- [ ] Ollama running (http://localhost:11434 responds or `ollama serve` is running)
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Terminal chat: `python main.py` **or** API: `python main.py --api`
- [ ] Test: say "I have fever and cough" and get a natural-language reply

After this, HealthBot is running with Ollama.
