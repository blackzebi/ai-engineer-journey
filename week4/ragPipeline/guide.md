# Guide — Pipeline ingest + search với pgvector (T4 tuần 4, block PROJECT 11:00–14:00)

> Block sáng đã có bảng `chunks` + câu SQL top-k. Chiều nay **ghép 3 mảnh rời rạc thành 1 pipeline**:
>
> ```
> [folder .md]  --chunker-->  [chunk]  --embed (tuần 3)-->  [vector]  --INSERT-->  pgvector
>                                                                                     │
>                     câu hỏi  --embed-->  [vector]  --ORDER BY <=>-->  top-5  ◄──────┘
> ```
>
> Đây chính là **nửa "R" của RAG**. Tuần 5 chỉ việc nhét top-5 vào prompt LLM là thành RAG đủ.

## Cấu trúc

```
week4/ragPipeline/
├── chunker.py     # đọc .md + cắt chunk        (3 TODO)
├── ingest.py      # chunk -> embed -> INSERT   (2 TODO)
├── search_pg.py   # câu hỏi -> top-5 + so sánh (3 TODO)
└── guide.md
```

Dùng lại `../pgvector/vector_ops.py` (block sáng) cho phần DB — **không viết lại SQL ở đây**.
Nếu import lỗi thì block sáng chưa xong, quay lại làm cho hết TODO đã.

---

## Bước 1 — `chunker.py`: folder .md → list chunk (~40')

**Mục tiêu:** biến kho notes thành các mẩu text vừa cỡ, kèm nguồn.

- [X] **TODO 1** `read_markdown_files`: `os.walk` đệ quy, lọc `.md`, đọc với `encoding="utf-8"`,
      lưu **đường dẫn tương đối** (sẽ thành cột `source`).
- [X] **TODO 2** `chunk_text`: cửa sổ trượt `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, bước nhảy = `size - overlap`.
- [X] **TODO 3** `chunk_folder`: ghép 2 hàm trên → `[(source, chunk), ...]`.
- [X] Chạy thử **trước khi đụng DB**:

  ```bash
  cd week4/ragPipeline
  python chunker.py "D:/Study/AI-Engineer-Study/python-knowledge"
  ```

**Vì sao chunk (câu trả lời phỏng vấn):** 1 vector cho cả file dài = trung bình cộng mọi chủ đề
trong file → search ra thứ chung chung. Chunk nhỏ = mỗi vector đại diện 1 ý → retrieval sắc nét.

**Vì sao overlap:** cắt cứng theo ký tự sẽ chặt ngang câu; chồng lấn 100 ký tự để ý bị cắt vẫn
xuất hiện trọn vẹn ở ít nhất 1 chunk.

**Bẫy:**
- `overlap >= chunk_size` → bước nhảy ≤ 0 → **lặp vô hạn**. Thêm guard `raise ValueError`.
- Thiếu `encoding="utf-8"` → tiếng Việt lỗi font / `UnicodeDecodeError` trên Windows.
- Chunk rỗng (file trống, toàn dòng trắng) → nhớ `.strip()` và bỏ qua.

**Xong khi:** in ra tổng số chunk hợp lý (kho ~64 file .md → thường vài trăm chunk) và 3 chunk mẫu
đọc được, có kèm tên file nguồn.

---

## Bước 2 — `ingest.py`: chunk → embed → INSERT (~50')

**Mục tiêu:** bảng `chunks` có N dòng dữ liệu **thật**, mỗi dòng 1 vector 384 chiều.

- [X] **TODO 1** `embed_texts`: `model.encode(texts, batch_size=64, show_progress_bar=True).tolist()`.
- [X] **TODO 2** `ingest`: chunk → embed → `create_chunks_table(drop=True)` → `insert_chunks_batch`.
- [X] Chạy:

  ```bash
  python ingest.py
  # hoặc chỉ folder khác: python ingest.py "D:/Study/AI-Engineer-Study/python-knowledge"
  ```

- [X] Kiểm tra bằng SQL:

  ```bash
  docker exec -it rag-pg psql -U postgres -d study_rag \
    -c "SELECT count(*) AS n_chunks, count(DISTINCT source) AS n_files FROM chunks;"
  ```

**Bẫy:**
- **Model phải trùng tuần 3** (`paraphrase-multilingual-MiniLM-L12-v2`). Khác model → vector khác
  không gian nghĩa → kết quả rác **mà không báo lỗi gì cả**.
- Chạy `ingest.py` 2 lần không xoá bảng → dữ liệu **trùng đôi**, top-5 toàn cặp giống nhau.
  Vì vậy mặc định `reset=True` (DROP rồi tạo lại).
- Embed từng chunk một sẽ rất chậm → luôn `encode()` theo **lô**.
- Lần đầu chạy sẽ tải model (~470MB) — bình thường, chờ.

**Xong khi:** `count(*)` > 0 và `count(DISTINCT source)` xấp xỉ số file .md trong kho.

---

## Bước 3 — `search_pg.py`: top-5 + so với bản thuần Python (~40')

**Mục tiêu:** hỏi được bằng tiếng Việt và trả về đúng chunk, **và giải thích được** khác gì tuần 3.

- [X] **TODO 1** `embed_query`: `get_model().encode(question).tolist()`.
- [X] **TODO 2** `search`: embed → `search_top_k(conn, vec, k=5)`.
- [X] Chạy vài câu hỏi thật (lấy từ notes tuần 1–3):

  ```bash
  python search_pg.py "làm sao gọi mạng bất đồng bộ trong Python?"
  python search_pg.py "structured output khác gì function calling?"
  python search_pg.py "vì sao phải chunk tài liệu trước khi embed?"
  ```

- [ ] **TODO 3** `compare_with_python`: chạy `--compare` để đối chiếu với `semanticSearch/semantic_search.py`.

  ```bash
  python search_pg.py "làm sao gọi mạng bất đồng bộ trong Python?" --compare
  ```

**Câu hỏi cần trả lời (ghi vào notes — đây mới là "sản phẩm" của buổi chiều):**

1. Top-5 hai bên có giống nhau không? Nếu khác — do **kho khác nhau** (corpus.txt ~20 dòng vs toàn
   bộ notes), do **chunk khác**, hay do **nhầm chiều distance/similarity**?
2. Điểm similarity 2 bên có cùng thang không? (Cả hai đều là cosine → phải cùng thang `[-1, 1]`.
   Nếu pgvector ra số lạ, khả năng cao là quên `1 - (<=>)`.)
3. Ở quy mô này (vài trăm chunk) bên nào **nhanh** hơn? Còn ở 100k chunk thì sao — vì sao?
4. pgvector cho thêm gì mà list Python không có? (persist qua lần chạy · SQL filter theo `source` ·
   index HNSW · nhiều process cùng đọc · backup)

**Bẫy:**
- Model ở `search_pg.py` phải **trùng** `ingest.py`. Đây là lỗi #1 khi kết quả "sai mà không lỗi".
- Câu hỏi tiếng Việt mà lỡ dùng model tiếng Anh → điểm thấp đều, thứ hạng lộn xộn.

**Xong khi:** ít nhất 3 câu hỏi trả về chunk **đúng chủ đề**, similarity của #1 tách biệt rõ so với #5,
và bạn viết được 4 câu trả lời trên vào notes.

---

## Bước 4 — Commit + push `week4/` lên GitHub (~15')

- [ ] Kiểm tra **không commit rác**: `.venv/`, `__pycache__/`, `.env` phải nằm trong `.gitignore`.

  ```bash
  cd D:\Projects\ai-engineer-journey
  git status                     # nhìn kỹ danh sách trước khi add
  ```

- [ ] Commit:

  ```bash
  git add week4/ requirements.txt .env.example
  git commit -m "week4(T4): bảng chunks + pipeline ingest/search với pgvector"
  git push
  ```

- [ ] Nếu có cài thêm package hôm nay:

  ```bash
  pip freeze | findstr /I "sentence-transformers psycopg" >> requirements.txt   # rồi dọn dòng trùng
  ```

**Bẫy:** `.env` chứa API key — lỡ push là phải **thu hồi key**, không phải chỉ xoá commit. Kiểm tra
`git status` kỹ trước khi `git add`.

**Xong khi:** GitHub thấy đủ `week4/pgvector/` + `week4/ragPipeline/`, **không** có `.env` và `__pycache__`.

---

## Tự kiểm tra cuối ngày

- [ ] Vẽ lại pipeline từ trí nhớ: folder → chunk → embed → INSERT → query → top-k.
- [ ] `CHUNK_SIZE` / `CHUNK_OVERLAP` ảnh hưởng chất lượng retrieval thế nào?
- [ ] Vì sao model ingest và model query bắt buộc giống nhau?
- [ ] pgvector hơn list Python ở điểm nào — và **khi nào list Python là đủ**?
- [ ] Nếu ngày mai muốn thêm 5 file notes mới mà không ingest lại toàn bộ — làm thế nào?
      (gợi ý: `DELETE FROM chunks WHERE source = %s` rồi insert lại phần đó)

**Talking points phỏng vấn:**
- "Ingest pipeline của tôi: đọc .md → chunk 800 ký tự overlap 100 → embed batch → bulk insert pgvector, giữ cột `source` để citation."
- "Chunk size là đánh đổi: nhỏ thì mất ngữ cảnh, lớn thì vector loãng nghĩa; overlap để câu không bị cắt cụt."
- "Tôi đã build cả bản list-Python lẫn bản pgvector và so — hiểu rõ vector DB thêm giá trị gì chứ không dùng theo phong trào."
