# Guide — T2 tuần 5 (27/07): Ingest đa định dạng PDF / DOCX / TXT / MD

> **Mục tiêu ngày:** đưa Dự án 1 ra khỏi "sân nhà" notes `.md` — nạp được tài liệu thật
> nhiều định dạng vào pgvector, kèm metadata đủ để trích nguồn kiểu *"file X, trang 12"*.
>
> ```
> [.pdf/.docx/.txt/.md] ──loaders──► Block(text, page, heading) ──normalize──► chunk
>                                                                               │
>          pgvector(chunks + doc_type/title/page/heading/chunk_index) ◄──embed──┘
> ```

---

## 0. Đọc lại 4 tuần trước: cái gì dùng lại, cái gì phải sửa

Đây là bước đáng giá nhất trước khi gõ dòng code đầu tiên. Hôm nay **không phải viết
pipeline mới** — chỉ thay đúng một tầng.

### Dùng lại NGUYÊN SI (đừng viết lại)

| Từ đâu | Thứ dùng lại | Vì sao vẫn dùng được |
|---|---|---|
| `week4/ragPipeline/chunker.py` | `chunk_text()` | Cửa sổ trượt + overlap không quan tâm text đến từ đâu |
| `week4/pgvector/vector_ops.py` | `get_conn`, `to_pgvector`, `EMBED_DIM`, `search_top_k` | Tầng DB không đổi, chỉ thêm cột |
| `week4/ragPipeline/ingest.py` | *khuôn* embed theo lô + assert số chiều | Logic đúng, chỉ đổi nguồn dữ liệu vào |
| `week4/ragLite/*` | toàn bộ retrieve + prompt + eval | Hôm nay **không đụng** — nhưng T4 sẽ phải sửa `format_citations` để in trang |
| `week3/utils.py` | `parse_json_safely` | Chưa cần hôm nay |

### PHẢI sửa — và đây là 4 chỗ đau thật

**1. `chunk_folder()` tuần 4 làm mất metadata.**
Nó nối cả file thành một chuỗi rồi mới cắt. Với `.md` thì chấp nhận được vì `source` là
đủ. Với PDF 80 trang thì `source` chỉ nói được *"nằm đâu đó trong file này"* — vô dụng.
→ Không sửa `chunker.py`. Viết `blocks_to_chunks()` mới ở `ingest_docs.py`, gọi lại
`chunk_text()` cho **từng block**, mang page/heading đi kèm.

**2. `insert_chunks_batch()` chỉ có 3 cột.** → bản mở rộng 8 cột trong `db_docs.py`.

**3. `ingest(reset=True)` xoá sạch bảng mỗi lần chạy.**
Tuần 4 chính docstring đã ghi *"cách xịn hơn ở tuần sau"* — tuần sau là hôm nay.
Với vài trăm chunk `.md` thì embed lại toàn bộ mất 30 giây, không sao. Với 200 PDF thì
mỗi lần thêm 1 file phải embed lại tất cả. → bảng `documents` + `file_hash` + 3 nhánh
`skip / new / changed`.

**4. `chunk_text()` có guard chết.**
```python
if overlap >= chunk_size:      # nằm SAU vòng lặp for -> không bao giờ chạy tới
    raise ValueError(...)      # range(0, n, step) với step<=0 đã nổ ValueError trước rồi
```
→ Chuyển guard lên **đầu hàm**, trước `range()`. Sửa 2 dòng, commit riêng
(`fix(week4): move overlap guard before loop`). Việc nhỏ nhưng là thói quen tốt:
tuần này đọc lại code tuần trước và trả nợ ngay khi thấy.

### Chưa có, phải cài

```bash
pip install pypdf pdfplumber python-docx
pip freeze > requirements.txt          # nhớ commit cả file này
```

---

## Cấu trúc

```
week5/ingestDocs/
├── loaders.py        # AI CORE : PDF/DOCX/TXT/MD -> LoadedDoc[Block]      (3 TODO)
├── normalize.py      # AI CORE : làm sạch text PDF                        (4 TODO)
├── db_docs.py        # AI CORE : migration + chống trùng                  (4 TODO)
├── ingest_docs.py    # PROJECT : pipeline end-to-end                      (5 TODO)
└── guide.md
```

Phụ thuộc phải sẵn sàng trước: container `rag-pg` đang Up
(`cd week4/pgvector && docker compose up -d`).

---

## Block 1 — AI CORE (09:00 – 11:00)

**Thứ tự làm — bám đúng thứ tự này, mỗi bước chạy được rồi mới sang bước sau:**

### 09:00–09:15 · Chuẩn bị bộ tài liệu test
- [X] Tạo `D:/Study/tai-lieu-test/` với **5 file thật**: 2 PDF (1 file chữ bình thường,
      1 file "khó" — nhiều cột / nhiều bảng), 1 DOCX có heading, 1 TXT, 1 MD.
- [X] Nếu có PDF scan thì để luôn vào — nó sẽ lộ ra bug im lặng thú vị nhất hôm nay.

> Đừng dùng file mẫu tải trên mạng cho đẹp. File thật bẩn ở chỗ nào thì phải thấy hôm nay,
> không phải lúc demo.

### 09:15–10:00 · `loaders.py` (TODO 1–3)
- [X] **TODO 1** `load_pdf` — 1 Block / 1 trang, `page` là số trang **thật**.
- [X] **TODO 2** `load_docx` — gom paragraph theo Heading; nhớ flush buffer lần cuối.
- [X] **TODO 3** `load_text` — `.md` tách theo `#`, `.txt` một block; fallback encoding.
- [X] Chạy `python loaders.py "<từng file>"` — **cả 5 file**, nhìn số ký tự và 200 ký tự đầu.

**So pypdf vs pdfplumber (task trong plan — làm thật, đừng đọc lý thuyết rồi kết luận):**
cùng file, cùng **trang 2** (trang 1 hay là bìa, không nói lên gì), in 300 ký tự đầu của
mỗi bên cạnh nhau. Ghi vào notes: file nào, bên nào ít vỡ hơn, vỡ kiểu gì.

### 10:00–10:45 · `normalize.py` (TODO 1–4) — ⭐ phần khó thật
- [X] Chạy `python normalize.py` trước tiên — dữ liệu giả mô phỏng đủ 4 loại rác,
      không cần PDF, không cần DB. Sửa cho tới khi đúng kỳ vọng ghi cuối file.
- [X] **TODO 1** `find_repeated_lines` — thống kê trên chính tài liệu, không hard-code blacklist.
- [X] **TODO 2** `is_page_number` — vài regex đọc được, đừng gom thành một regex thần thánh.
- [X] **TODO 3** `join_broken_lines` — **viết luật ra giấy trước khi code**. Chỗ này code mò
      là mất 40 phút không lối thoát.
- [X] **TODO 4** `collapse_whitespace` — nhớ giữ lại ngắt đoạn.
- [X] Nối vào `load_pdf`, chạy lại trên PDF thật, **so trước/sau bằng mắt**.

### 10:45–11:00 · `db_docs.py` (TODO 1–2, phần schema)
- [X] **TODO 1** `migrate_chunks_table` — `ADD COLUMN IF NOT EXISTS` + unique index.
- [X] **TODO 2** `create_documents_table`.
- [X] `python db_docs.py` → in ra schema có đủ 5 cột mới. Chạy **2 lần** để chắc idempotent.
- [ ] 🗣 **Tiếng Anh (10:45–11:00)**: ghi đúng 5 từ **lấy từ docs pypdf/python-docx vừa đọc** —
      gợi ý sẵn có trong bài: *extract, layout, corrupted, whitespace, idempotent, deduplicate,
      metadata, fallback*. Chọn 5 từ **mình vừa thật sự gặp**, không chép list này.

**Câu hỏi phải trả lời được sau block này** (viết vào `python-knowledge/Tuan-05/T2/README.md`):

1. Vì sao PDF cần normalize mà `.md` thì không? (gợi ý: PDF mô tả *cách vẽ chữ lên giấy*,
   không phải cấu trúc văn bản)
2. Header lặp lại 40 lần ảnh hưởng retrieval thế nào — và vì sao **rác lặp lại** nguy hiểm
   hơn rác ngẫu nhiên?
3. Vì sao trích nguồn kèm `page` thuyết phục hơn hẳn kèm khoảng ký tự?
4. `ADD COLUMN IF NOT EXISTS` — "idempotent" nghĩa là gì và vì sao migration bắt buộc phải vậy?

---

## Block 2 — PROJECT (11:00 – 14:00, nghỉ trưa 12:00–13:00)

### 11:00–12:00 · `db_docs.py` (TODO 3–4) + `ingest_docs.py` (TODO 1–3)
- [X] **TODO 3** `should_ingest` / `delete_document` — 3 nhánh `skip / new / changed`.
- [X] **TODO 4** `insert_chunks_batch` (8 cột) + `record_document` (upsert `ON CONFLICT`).
- [X] **TODO 1** `collect_files` — nhớ bỏ file tạm `~$*.docx` của Word.
- [X] **TODO 2** `blocks_to_chunks` — ⭐ mảnh quan trọng nhất buổi chiều.
- [X] **TODO 3** `embed_texts` — copy khuôn tuần 4, giữ nguyên assert số chiều.

### 13:00–13:40 · `ingest_docs.py` (TODO 4–5) + chạy thật
- [X] **TODO 4** `ingest_one` — try/except quanh đọc file, 1 PDF hỏng không giết cả lượt chạy.
- [X] **TODO 5** `ingest_folder` — thống kê + in tiến độ từng file.
- [X] Chạy: `python ingest_docs.py "D:/Study/tai-lieu-test"`

### 13:40–14:00 · Nghiệm thu — 4 phép thử, làm đủ cả 4
```bash
# 1. Chống trùng: chạy lại ĐÚNG lệnh trên
python ingest_docs.py "D:/Study/tai-lieu-test"
#    → skip=5, new=0, và tổng chunk trong DB KHÔNG đổi

# 2. Phân bố theo định dạng — loader nào đọc hụt sẽ lộ ra ở đây
docker exec -it rag-pg psql -U postgres -d study_rag \
  -c "SELECT doc_type, COUNT(*) n_chunks, COUNT(DISTINCT source) n_files
      FROM chunks GROUP BY doc_type ORDER BY n_chunks DESC;"

# 3. Metadata có thật không (không phải NULL hết)
docker exec -it rag-pg psql -U postgres -d study_rag \
  -c "SELECT source, page, heading, left(content,60) FROM chunks
      WHERE doc_type='pdf' ORDER BY random() LIMIT 5;"

# 4. Sửa 1 file test rồi ingest lại → phải ra changed=1, và số chunk của file đó
#    thay đổi chứ KHÔNG cộng dồn
```

- [X] Commit & push:
```bash
git add week5/ingestDocs requirements.txt
git commit -m "week5: ingest đa định dạng (pdf/docx/txt/md) + metadata trang + chống trùng"
git push
```

**Tiêu chí "xong" của hôm nay:**

- 5 file, 4 định dạng → vào DB đủ, `SELECT doc_type, COUNT(*)` khớp với số file có thật.
- Chunk từ PDF có `page` **không NULL**; chunk từ DOCX/MD có `heading` không NULL.
- Chạy lại lần 2 → `skip` toàn bộ, số chunk không đổi.
- PDF đọc ra 0 ký tự thì **có cảnh báo rõ ràng**, không im lặng trôi qua.

---

## Nhìn trước: hôm nay ăn khớp với phần còn lại của tuần thế nào

| Ngày | Việc | Phụ thuộc gì từ hôm nay |
|---|---|---|
| T3 29/07 | Index HNSW + tuning retrieval | Cần dữ liệu đủ lớn & đủ sạch — nếu chunk còn rác thì đo tuning ra số vô nghĩa |
| T4 30/07 | Đóng gói CLI + robustness | `ingest_one` đã có try/except từ hôm nay là nền sẵn |
| T5 31/07 | **Dự án 1 end-to-end** | Trích nguồn "file X, trang 12" — chính là cột `page` hôm nay |

> Nói cách khác: hôm nay không ra được demo đẹp nào cả. Nhưng nếu metadata hôm nay sai thì
> T5 không có gì để khoe. Đây là ngày làm móng.

---

## Việc ngoài code trong plan hôm nay — đừng để trôi

- ⬅️ **(Nợ 3 tuần)** 14:00 — ghi âm 2–3 phút mô tả 5 mini-project bằng tiếng Anh.
  Ghi âm xong là ĐẠT. Món nợ lâu nhất lộ trình; hôm nay có 5 project để kể rồi, không còn cớ.
- ⬅️ **BHTN** 15:00 — mốc "đủ hồ sơ cuối tháng 7" rơi đúng tuần này. Nếu thiếu: hỏi rõ
  thiếu giấy gì, khi nào có.
- ⬅️ **(Nợ tuần 4, ưu tiên số 1)** 15:45 — viết bù note T4 pgvector vào
  `Tuan-04/T4/README.md`: bảng chunks, `<=>` vs similarity, pipeline ingest, bẫy đã gặp.
- [ ] LinkedIn: 1 post ngắn về RAG-lite.
- [ ] Note T2 tuần 5 + tick bảng lịch tuần.
