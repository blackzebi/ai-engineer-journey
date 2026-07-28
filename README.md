# ai-engineer-journey

Hands-on projects from a 20-week transition from full-stack engineering into AI engineering.
Each week ships something that runs; the code here is the working record, not tutorials.

**Focus so far:** LLM APIs → structured output & tool use → embeddings & semantic search →
retrieval with pgvector → **RAG with grounded answers and citations**.

## Featured project — RAG-lite

**[`week4/ragLite`](week4/ragLite)** — Ask a question in Vietnamese about my own study notes and get
an answer that is grounded in those notes and **cited**, so every claim can be traced back to a file.

```
question ──embed──► top-3 chunks (pgvector) ──► grounded prompt ──► Claude ──► answer + citations
```

```
❓ asyncio trong Python dùng để làm gì?

💬 Asyncio dùng để viết code **không block khi chờ I/O** [1], cho phép chương trình xử lý
   nhiều tác vụ đồng thời mà không cần nhiều thread. [...]

Nguồn tham khảo:
  [1] Tuan-01_29Jun-05Jul_GD0-Khoi-dong/Async/README.md          (similarity 0.65)
  [2] Tuan-01_29Jun-05Jul_GD0-Khoi-dong/Async/README.md          (similarity 0.62)
  [3] Tuan-01_29Jun-05Jul_GD0-Khoi-dong/Async/async-advanced.md  (similarity 0.60)
```

The interesting part is the refusal behaviour: when the notes don't cover a question, the system
answers *"Tôi không tìm thấy thông tin này trong tài liệu."* instead of inventing something. That is
enforced by a grounded prompt contract and verified by a small eval — 5 cases, 3 answerable and
2 deliberately unanswerable, currently **5/5 PASS**.

Stack: PostgreSQL + pgvector (Docker) · sentence-transformers (multilingual MiniLM, 384d) ·
Anthropic Claude · psycopg 3. Details in the [project README](week4/ragLite/README.md).

## Projects by week

| Week | Project | What it covers |
|---|---|---|
| 1 | [`week1`](week1) | First Claude API calls, a minimal CLI chatbot, key handling via `.env` |
| 2 | [`week2`](week2) | [Summarization, classification, and a chatbot with conversation memory](week2/README.md) |
| 3 | [`week3`](week3) | Structured output and **tool use** — calculator / time / file tools, plus safe JSON parsing |
| 4 | [`week4/semanticSearch`](week4/semanticSearch/README.md) | Local semantic search: embeddings + cosine similarity written by hand, no vector DB |
| 4 | [`week4/pgvector`](week4/pgvector) | PostgreSQL + pgvector in Docker: vector columns, distance operators, top-k queries |
| 4 | [`week4/ragPipeline`](week4/ragPipeline) | Ingestion pipeline: chunking with overlap → embedding → storing vectors |
| 4 | [`week4/ragLite`](week4/ragLite/README.md) | **RAG-lite** — grounded, cited Q&A over the notes, with a refusal eval |

## Environment setup

### 1. Local virtual environment

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

### 2. Google Colab: test an API call using secrets (without exposing the key)

Open [notebooks/colab_test_api_call.ipynb](notebooks/colab_test_api_call.ipynb) in Colab, then:

1. Click the **key icon** in the left sidebar (Secrets).
2. **Add new secret** → name `ANTHROPIC_API_KEY`, value = your key.
3. Toggle **Notebook access** on for this notebook.
4. Run all cells.

The key is loaded via `google.colab.userdata.get(...)` and never printed or hardcoded in the notebook — only the key length is logged as a sanity check.

Expected output: `pong`.
