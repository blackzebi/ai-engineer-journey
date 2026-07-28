# Guide — Mini-project #5: RAG-lite hỏi-đáp notes (T5 tuần 4)

> Hôm nay ghép mảnh cuối: **retrieval (T4) + LLM = RAG đủ**. Không viết lại retrieval —
> chỉ thêm bước "nhét top-3 chunk vào prompt rồi gọi Claude".
>
> ```
> câu hỏi ──embed──► top-3 chunk (pgvector) ──► prompt(context+hỏi) ──► Claude ──► trả lời + nguồn
> ```

## Cấu trúc

```
week4/ragLite/
├── prompt_rag.py    # AI CORE: khuôn prompt grounding + trích nguồn   (3 TODO)
├── rag_qa.py        # PROJECT: pipeline end-to-end + eval             (5 TODO)
├── questions.json   # 5 câu test (3 có trong notes, 2 không)
├── guide.md
└── README.md        # mô tả portfolio + ví dụ hỏi-đáp (task 'Việc làm')
```

Phụ thuộc (phải xong trước): `../pgvector/vector_ops.py`, `../ragPipeline/search_pg.py`,
bảng `chunks` đã ingest (chạy `python ../ragPipeline/ingest.py` nếu chưa).

---

## Block 1 — `prompt_rag.py` (09:00–11:00, AI CORE)

**Mục tiêu buổi:** hiểu *vì sao* khuôn prompt giảm hallucination, rồi mới code.

- [X] **TODO 1** `format_context`: đánh số `[1] (nguồn: ... · similarity=...)` + nội dung chunk.
- [X] **TODO 2** `build_rag_prompt`: context TRƯỚC, câu hỏi SAU, kèm đường thoát "KHÔNG BIẾT".
- [X] **TODO 3** `format_citations`: danh sách nguồn hiện dưới câu trả lời.
- [X] Chạy thử KHÔNG cần DB: `python prompt_rag.py` (dùng chunk giả) — xem prompt ráp đúng hình chưa.

**Câu hỏi phải trả lời được sau block này** (viết vào T5/README):
1. Vì sao thêm câu "nếu không có thì nói KHÔNG BIẾT" lại giảm bịa? (gợi ý: đóng khung nguồn +
   mở đường thoát hợp lệ để model không bị ép trả lời bằng mọi giá).
2. Vì sao trích nguồn quan trọng với RAG production? (kiểm chứng được = tin được).

---

## Block 2 — `rag_qa.py` (11:00–14:00, PROJECT)

- [X] **TODO 1** `get_client`: đọc `ANTHROPIC_API_KEY`, báo lỗi rõ nếu thiếu.
- [X] **TODO 2** `retrieve`: gọi lại `embed_query` + `search_top_k(k=3)`. Bọc try/except DB.
- [X] **TODO 3** `generate`: `client.messages.create(system=SYSTEM_PROMPT, ...)`; chunks rỗng → `NO_ANSWER`.
- [X] **TODO 4** `answer`: ghép retrieve → generate → dict `{question, answer, sources}`.
- [X] **TODO 5** `run_eval`: đọc `questions.json` bằng `parse_json_safely` (utils tuần 3), tự chấm, in bảng markdown.
- [X] Chạy thật:

  ```bash
  cd week4/ragLite
  python rag_qa.py "asyncio trong Python dùng để làm gì?"
  python rag_qa.py --eval
  ```

**Tiêu chí "xong" của mini-project:**
- 3 câu CÓ trong notes → trả đúng + có trích nguồn (file + similarity).
- 2 câu KHÔNG có → trả đúng câu `Tôi không tìm thấy thông tin này trong tài liệu.`
- Dán bảng kết quả `--eval` vào `README.md` (đây là eval thô sơ đầu tiên → tuần 7 làm eval xịn).

---

## Block 3 — Việc làm (15:00–15:45)

- [X] Cập nhật `README.md` của repo gốc + `ragLite/README.md`: thêm mô tả RAG-lite, ảnh chụp/ví dụ 1 lượt hỏi-đáp.
- [ ] Commit + push:

  ```bash
  git add week4/ragLite
  git commit -m "week4: mini-project #5 RAG-lite (retrieve + generate + eval thô)"
  git push
  ```

## Ôn tập (15:45–17:00)

- [ ] Note **luồng RAG pipeline** vào `python-knowledge/Tuan-04_.../T5/README.md` (đã có bản nháp — bổ sung bằng lời của mình).
- [ ] Nhật ký tiếng Anh: giải thích embedding bằng 4 câu đơn giản (xem `T5/nhat-ky-tieng-anh.md`).

---

## Bẫy hay gặp hôm nay

| Triệu chứng | Nguyên nhân | Cách fix |
|---|---|---|
| Model vẫn bịa dù đã có context | Câu "đường thoát" bị bỏ, hoặc để trong `messages` thay vì `system` | Đưa luật vào `system=`, giữ nguyên câu KHÔNG BIẾT |
| `search_top_k` trả rỗng | Chưa ingest / sai `MODEL_NAME` giữa ingest và query | Chạy lại `ingest.py`; model embed 2 bên phải TRÙNG |
| Trả lời không khớp nguồn | k quá nhỏ hoặc chunk loãng | Thử k=5, xem lại `CHUNK_SIZE` ở chunker |
| `psycopg... connection refused` | Container Postgres chưa chạy | `docker compose up -d` trong `../pgvector` |
