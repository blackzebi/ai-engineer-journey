# Guide — Bảng chunks + INSERT + query top-k (T4 tuần 4, block AI core 8:30–11:00)

> Hôm qua (T3) đã dựng hạ tầng: container Postgres + extension `vector` + hiểu `vector(n)`.
> Hôm nay biến nó thành **vector store dùng được**: bảng `chunks` thật, INSERT embedding từ Python,
> và câu SQL top-k — chính là bước *retrieve* của RAG, nhưng chạy trong DB thay vì list Python.
>
> Code khung: [`vector_ops.py`](vector_ops.py) — điền 5 chỗ `# TODO`.
> Xong block này → sang [`../ragPipeline/guide.md`](../ragPipeline/guide.md) để ráp pipeline thật.

## Bức tranh hôm nay

```
Tuần 3 (list Python)                    Hôm nay (pgvector)
────────────────────                    ──────────────────
vectors.json                     -->    bảng chunks(id, source, content, embedding vector(384))
cosine_similarity() tự viết      -->    toán tử <=> trong SQL
sort() rồi lấy 3 đầu             -->    ORDER BY embedding <=> %s LIMIT 5
```

Cùng một thuật toán — chỉ đổi chỗ chạy. Hiểu điều này là hiểu "vì sao cần vector DB".

## Chuẩn bị (2 phút)

```bash
docker ps                        # rag-pg phải Up. Chưa Up: cd week4/pgvector && docker compose up -d
python pg_smoke_test.py          # vẫn in đủ 3 dòng ✅ -> hạ tầng còn nguyên
```

---

## Task 5 — Tạo bảng `chunks(id, source, content, embedding vector(384))` — vì sao 384

**Mục tiêu:** có bảng thật, giải thích được từng cột và con số 384.

- [X] Điền **TODO 1** (`create_chunks_table`): `CREATE EXTENSION IF NOT EXISTS vector` → (nếu `drop`)
      `DROP TABLE IF EXISTS chunks` → `CREATE TABLE IF NOT EXISTS chunks (...)` → `conn.commit()`.

**Vì sao mỗi cột:**

| Cột | Kiểu | Vì sao cần |
|---|---|---|
| `id` | `bigserial PRIMARY KEY` | khoá tự tăng, để trỏ tới chunk |
| `source` | `text NOT NULL` | **file gốc** của chunk. RAG bắt buộc trích nguồn — thiếu cột này thì tuần 5 không citation được |
| `content` | `text NOT NULL` | chữ gốc, sẽ nhét vào prompt LLM |
| `embedding` | `vector(384)` | vector nghĩa của `content` |

**Vì sao 384:** model tuần này `paraphrase-multilingual-MiniLM-L12-v2` xuất ra đúng **384 số**.
Cột `vector(n)` là **cố định cho cả bảng** → `n` phải khớp model. Đổi sang OpenAI
`text-embedding-3-small` (1536) là phải DROP + tạo lại bảng + embed lại toàn kho — không migrate mềm được.

**Xong khi:**

```bash
docker exec -it rag-pg psql -U postgres -d study_rag -c "\d chunks"
```

thấy đủ 4 cột, `embedding | vector(384)`.

---

## Task 6 — INSERT embedding từ Python + `<=>` vs `<->`

**Mục tiêu:** đưa vector từ Python vào DB, và **không bao giờ nhầm distance với similarity nữa**.

- [X] Điền **TODO 2** (`insert_chunk`): `INSERT ... VALUES (%s, %s, %s) RETURNING id` → `commit()`.
- [X] Điền **TODO 3** (`insert_chunks_batch`): `cur.executemany(...)` — dùng cho ingest.
- [X] Điền **TODO 4** (`compare_operators`): SELECT cùng lúc `<=>`, `1 - (<=>)`, `<->` rồi in ra so sánh.

### 🔑 Bẫy lớn nhất tuần này

> **pgvector trả về DISTANCE, không phải similarity. Số NHỎ = GẦN = liên quan hơn.**
> Ngược hoàn toàn với cosine similarity tuần 3 (số LỚN = giống hơn).
>
> ```
> cosine_similarity = 1 - (embedding <=> query)
> ```
>
> Nhầm chiều này thì top-k trả ra đúng những chunk **ít liên quan nhất** mà code vẫn chạy, không lỗi —
> loại bug im lặng, khó thấy. Nhớ bằng câu: *"`<=>` là khoảng cách, khoảng cách thì càng nhỏ càng gần."*

| Toán tử | Tên | Công thức | Khoảng giá trị |
|---|---|---|---|
| `<=>` | cosine distance | `1 - cos(a,b)` | `[0, 2]` — 0 = trùng hướng |
| `<->` | L2 (Euclidean) | `‖a - b‖` | `[0, ∞)` |
| `<#>` | negative inner product | `-(a·b)` | pgvector để dấu âm cho quy ước "nhỏ = gần" |

**Bẫy khác:**
- Dùng placeholder `%s`, **không** f-string SQL (SQL injection + lỗi quote vector).
- Sai số chiều → `expected 384 dimensions, not N`.
- **Quên `commit()`** → chạy xong, vào `psql` SELECT thấy 0 dòng, tưởng code sai. psycopg **không** autocommit.

**Quan sát để ghi notes:** thứ hạng theo `<=>` và theo `<->` có giống nhau không? Nếu vector được
chuẩn hoá (norm = 1) thì 2 thứ hạng **trùng**; sentence-transformers mặc định **không** normalize
(`encode(..., normalize_embeddings=True)` mới có) → có thể lệch. Câu này hay bị hỏi phỏng vấn.

**Xong khi:** in ra 1 dòng có đủ `cosine_distance`, `cosine_similarity`, `l2_distance` và bạn giải
thích được vì sao 2 số đầu cộng lại bằng 1.

---

## Task 7 — Query top-k: `ORDER BY embedding <=> %s LIMIT 5`

**Mục tiêu:** viết đúng câu SQL xương sống của RAG.

- [ ] Điền **TODO 5** (`search_top_k`):

```sql
SELECT id, source, content, 1 - (embedding <=> %s) AS similarity
FROM chunks
ORDER BY embedding <=> %s          -- distance tăng dần = gần nhất trước
LIMIT %s;
```

**Bẫy:**
- `ORDER BY similarity DESC` cũng ra **đúng kết quả**, nhưng sắp theo biểu thức `1 - ...` thì
  **index vector (HNSW/IVFFlat) không dùng được** → luôn `ORDER BY <toán tử khoảng cách>`, cột
  similarity chỉ để *hiển thị*.
- Chưa tạo index thì Postgres quét tuần tự. Vài nghìn dòng vẫn nhanh — đủ cho tuần này.
  Index là chuyện **tuần 6** (retrieval nâng cao). Đừng sa đà hôm nay.
- Query vector phải embed bằng **cùng model** với lúc ingest. Khác model = vector không so được.

- [ ] **Đối chiếu với cosine tự cài tuần 3:** lấy 1 câu hỏi, chạy `semantic_search.py search "..."`
      và `search_pg.py "..."` (block chiều) → so top-k. Kết quả nên **gần trùng**; lệch thì tìm lý do
      (chunk khác nhau? chưa normalize? nhầm chiều distance?). Ghi kết luận vào notes.

**Xong khi:** `python vector_ops.py` chạy hết, in top-5 với similarity giảm dần.

---

## Thứ tự gợi ý (~2h30)

1. Task 5 — tạo bảng (20') →
2. Task 6 — INSERT + so 3 toán tử (50', phần lớn là *hiểu* chứ không phải gõ) →
3. Task 7 — top-k (30') →
4. Còn dư: đọc lại `vectors.json` tuần 3 và tự hỏi "cột nào trong bảng chunks thay cho field nào trong JSON".

> Nếu cần thêm thời gian: lấy 30' của block **Ôn tập 16:30** (theo ghi chú tracker).

## Tự kiểm tra đã hiểu

- [ ] `<=>` trả về gì? Số nhỏ hay lớn thì liên quan hơn? Công thức đổi sang similarity?
- [ ] Vì sao `vector(384)` mà không phải số khác? Đổi model thì phải làm gì?
- [ ] Vì sao `ORDER BY embedding <=> %s` chứ không `ORDER BY similarity DESC`?
- [ ] Cột `source` để làm gì — bỏ đi thì tuần 5 hỏng chỗ nào?
- [ ] Vì sao phải `conn.commit()` sau INSERT?

**Talking points phỏng vấn:**
- "pgvector trả distance chứ không phải similarity; `similarity = 1 - (a <=> b)` — nhầm chiều là bug im lặng."
- "Sắp xếp phải theo toán tử khoảng cách thì index HNSW mới ăn được."
- "Số chiều cột `vector(n)` khoá chặt theo model embedding; đổi model là phải re-index toàn kho."
