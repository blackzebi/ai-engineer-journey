# Semantic Search mini

A small semantic search that runs **locally and for free**: embed a corpus of text snippets (my
notes from weeks 1–3) with `sentence-transformers`, cache the vectors to JSON, then have
`search(query)` return the top-3 closest snippets by **hand-written cosine similarity**.
No API key, no vector database — a Python list is enough at this scale.

Learning goal: understand **retrieval** — the heart of RAG — before moving up to pgvector later in
week 4.

## Stack

- Python 3.10+
- [`sentence-transformers`](https://www.sbert.net/) — model `paraphrase-multilingual-MiniLM-L12-v2`
  (multilingual, good for Vietnamese, 384 dimensions, runs locally)
- Cosine similarity implemented by hand with `math` (no numpy)

## Run

```bash
pip install sentence-transformers

# Embed the corpus (run once, produces vectors.json). Downloads the model on first run.
python semantic_search.py build

# Query
python semantic_search.py search "con mèo là gì?"
```

Edit the corpus in `corpus.txt` (one snippet per line), then re-run `build`.
Changing `MODEL_NAME` also requires a rebuild — vectors from different models are not comparable.

## Layout

```
week4/semanticSearch/
├── semantic_search.py   # main code (build + search)
├── corpus.txt           # ~20 text snippets (one per line)
├── vectors.json         # produced by `build` (vector cache)
├── guide.md             # working notes (Vietnamese)
└── README.md
```

## Demo: semantic ≠ keyword

A good way to check ranking quality: the cosine score of #1 should be clearly ahead of #2.

**Sanity check** — `search "con mèo là gì?"`:

```
1. [0.6136] Con mèo không phải con chó nhưng cả hai đều là động vật sống trên Trái Đất.
2. [0.1389] System prompt định hướng cách model trả lời theo vai trò bạn đặt ra.
3. [0.0997] Trong Python bạn có thể dùng asyncio để chạy một hàm async.
```

#1 (0.61) is far ahead of #2 (0.14), so ranking works. This example still shares the literal words
"con mèo" with the target snippet, so it does not yet prove semantic ≠ keyword — the two examples
below do that.

**Queries sharing no words with the target snippet:**

### Example 1
- **Query:** `làm sao gọi mạng bất đồng bộ trong Python?` _(contains neither "httpx" nor "async")_
- **Top-3:**
  1. [0.5132] httpx là thư viện HTTP client hỗ trợ cả sync và async trong Python.
  2. [0.3649] Trong Python bạn có thể dùng asyncio để chạy một hàm async.
  3. [0.2756] REST API dùng các method GET/POST/PUT/DELETE trên tài nguyên qua URL.
- **Takeaway:** it surfaces the `httpx`/`async` snippets despite zero word overlap — matching on
  *meaning*, not *characters*.

### Example 2
- **Query:** `cách bắt AI trả về đúng định dạng máy đọc được` _(contains neither "JSON" nor "structured")_
- **Top-3:**
  1. [0.6287] Trong Anthropic có thể định nghĩa tool riêng để AI giải quyết việc bên ngoài như tính toán, hỏi ngày giờ.
  2. [0.4868] Structured output ép model trả JSON đúng schema để máy đọc được.
  3. [0.4659] System prompt định hướng cách model trả lời theo vai trò bạn đặt ra.
- **Takeaway:** the relevant structured-output snippet ranks highly, though the top hit shows ranking
  is not perfect at this corpus size.

## What I learned building this

**RAG starts with retrieval.** This whole project is the *retrieve* step of RAG in miniature:
`embed the corpus → embed the question → rank by vector distance → take top-k`. Swapping
"Python list + cosine" for a vector database (pgvector) is what turns it into real RAG.

**Embeddings turn text into vectors that encode meaning.** Snippets that mean similar things end up
with nearby vectors, so search matches *intent* rather than *characters* — it finds the right
snippet even when the query shares none of its keywords.

**Cosine similarity** measures the angle between two vectors: `dot(a,b) / (‖a‖·‖b‖)`, ranging over
`[-1, 1]`. It needs a **divide-by-zero guard** when a vector has norm 0. Scores are only meaningful
for *ranking within the same corpus and model* — they are not absolute, and not comparable across
models.

**Match the embedding model to the language of the data.** `all-MiniLM-L6-v2` is English-centric and
performed poorly on Vietnamese; switching to `paraphrase-multilingual-MiniLM-L12-v2` improved results
substantially with no code changes. Note that sentence-transformers has **no `input_type` parameter**
(unlike Voyage/OpenAI) — the same model embeds both corpus and query.

**Cache the vectors.** A snippet's embedding never changes, so `vectors.json` is written once and
reused by every later `search` — faster, and cheaper still if the embeddings came from a paid API.

**When you actually need a vector database.** At small scale (tens to hundreds of snippets), a list
plus a linear cosine scan is fine. Beyond that you need a purpose-built vector index
(pgvector / HNSW) instead of comparing against every element.

## Interview talking points

- "Keyword search matches *character strings*; semantic search matches *meaning* via vector distance."
- "Embed both the corpus and the query with the same model, rank by cosine similarity, take top-k."
- "The embedding model has to match the language of your data — I switched to a multilingual model for Vietnamese."
- "At small scale you don't need a vector DB; you need an index (pgvector/HNSW) once you scale up."
