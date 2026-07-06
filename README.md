# ai-engineer-journey

## AI Core — Setup môi trường

### 1. Setup venv local

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install openai anthropic python-dotenv httpx
```

Dependencies are pinned in [requirements.txt](requirements.txt) (`pip install -r requirements.txt` to reproduce the same environment).

Copy [.env.example](.env.example) to `.env` and fill in your keys — `.env` is gitignored, never commit it:

```bash
cp .env.example .env
```

Verify the venv can reach the API without leaking the key:

```bash
.venv\Scripts\python.exe scripts\test_api_call.py   # Windows
.venv/bin/python scripts/test_api_call.py            # macOS/Linux
```

Expected output: `pong`.

### 2. Setup Google Colab: test API call bằng secrets (không lộ key)

Open [notebooks/colab_test_api_call.ipynb](notebooks/colab_test_api_call.ipynb) in Colab, then:

1. Click the **key icon** in the left sidebar (Secrets).
2. **Add new secret** → name `ANTHROPIC_API_KEY`, value = your key.
3. Toggle **Notebook access** on for this notebook.
4. Run all cells.

The key is loaded via `google.colab.userdata.get(...)` and never printed or hardcoded in the notebook — only the key length is logged as a sanity check.

Expected output: `pong`.
