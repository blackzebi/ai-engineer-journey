# Guide — T3 tuần 5 (28/07): Index HNSW + đo chất lượng retrieval

> **Mục tiêu ngày:** làm retrieval NHANH hơn (index) và ĐO được nó tốt tới đâu, thay vì đoán.
> Ngày kỹ thuật sâu nhất tuần. Kết thúc ngày phải có **4 con số thật** trong note, không có câu nào là "cảm giác nhanh hơn".
>
> ```
>                    ┌─ AI CORE ────────────────────────────────────┐
>  chunks (đã ingest)│  EXPLAIN ANALYZE  ->  Seq Scan, T0 ms        │
>       T2 tuần 5    │  CREATE INDEX hnsw (embedding vector_cosine_ops)
>                    │  EXPLAIN ANALYZE  ->  Index Scan, T1 ms      │
>                    │  ef_search 20/40/100 -> T1 đổi               │
>                    └──────────────┬───────────────────────────────┘
>                                   │
>                    ┌─ nợ tuần 4 ──┴───────────────────────────────┐
>                    │  pgvector top-k   vs   cosine tay (Python)   │
>                    │  CÙNG kho, CÙNG query -> overlap@k           │
>                    └──────────────┬───────────────────────────────┘
>                                   │
>                    ┌─ PROJECT ────┴───────────────────────────────┐
>                    │  chunking 500/80 vs 1000/150 (trong RAM)     │
>                    │  WHERE doc_type -> index còn dùng không?     │
>                    │  k=3 vs k=5 -> phủ rộng hay loãng?           │
>                    └──────────────────────────────────────────────┘
> ```

---

## 0. Đọc lại tuần trước: cái gì dùng lại, cái gì phải sửa

### Dùng lại NGUYÊN SI (đừng viết lại)

| Từ đâu | Thứ dùng lại | Vì sao vẫn dùng được |
|---|---|---|
| `week4/pgvector/vector_ops.py` | `get_conn`, `to_pgvector`, `EMBED_DIM`, `search_top_k` | `search_top_k` viết `ORDER BY embedding <=> %s` — đúng dạng index dùng được. Hôm nay chỉ thêm index, **SQL không sửa một chữ** |
| `week4/ragPipeline/chunker.py` | `chunk_text(text, chunk_size, overlap)` | đã nhận tham số sẵn → truyền 2 cấu hình vào được |
| `week4/semanticSearch/semantic_search.py` | công thức `cosine_similarity` | dùng làm "chân lý" exact để đối chiếu HNSW |
| `week4/ragLite/questions.json` | 5 câu hỏi mẫu (3 answerable + 2 bẫy) | dùng đúng bộ này thì so được kết quả tuần 4 ↔ tuần 5 |
| `week5/ingestDocs/loaders.py` | `load_any`, `LoadedDoc`, `Block` | chunking so sánh cần đọc tài liệu theo block, không đọc lại từ đầu |
| `week5/ingestDocs/db_docs.py` | `stats_by_doc_type` | biết phân bố doc_type trước khi lọc metadata |

### PHẢI sửa — 3 chỗ đau thật

**1. `blocks_to_chunks(doc)` không nhận cấu hình chunking** — chặn hẳn task compare_chunking

`week5/ingestDocs/ingest_docs.py:92` gọi `chunk_text(block.text)` với size/overlap mặc định (800/100), không có đường truyền tham số vào. Không sửa thì `compare_chunking.py` sẽ ra **số chunk BẰNG NHAU cho cả 2 cấu hình** — và đây là bug im lặng: bảng in ra rất thuyết phục, chỉ có điều nó đang so 800/100 với 800/100.

Sửa (thêm tham số **có mặc định** → mọi chỗ gọi cũ không phải đổi):

```python
def blocks_to_chunks(
    doc: LoadedDoc,
    chunk_size: int = CHUNK_SIZE,        # import thêm CHUNK_SIZE, CHUNK_OVERLAP từ chunker
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, int | None, str | None, int]]:
    ...
        for piece in chunk_text(block.text, chunk_size=chunk_size, overlap=overlap):
```

**2. `chunk_text` — guard chống lặp vô hạn đặt SAI CHỖ, không bao giờ chạy**

`week4/ragPipeline/chunker.py`:

```python
step = chunk_size - overlap
for start in range(0, len(text), step):     # <- nổ / trả rỗng NGAY tại đây
    ...
    if overlap >= chunk_size:               # <- guard nằm TRONG loop, tới đây là muộn rồi
        raise ValueError("error infinity loop")
```

Đã kiểm chứng bằng Python:
- `overlap == chunk_size` → `step = 0` → `range(0, n, 0)` raise `ValueError: range() arg 3 must not be zero` — thông báo lỗi của Python, không phải của mình, khó truy ngược.
- `overlap > chunk_size` → `step < 0` → `range` **rỗng** → `chunk_text` trả `[]`, **không lỗi gì cả**. Cả tài liệu biến mất im lặng.

Hôm nay dùng 500/80 và 1000/150 nên chưa dính, nhưng compare_chunking là chỗ đầu tiên trong toàn repo truyền tham số tuỳ ý vào — sớm muộn sẽ thử 500/500. Sửa: dời guard lên **trước** `step = ...`. Commit riêng, 2 dòng.

**3. `so_sanh_pdf.py` — tên hàm/biến bằng tiếng Việt, và tên file cũng vậy**

`week5/ingestDocs/so_sanh_pdf.py` có `doc_bang_pypdf`, `doc_bang_pdfplumber`, `in_ket_qua(ten, ...)`. Toàn bộ code còn lại của repo đang dùng identifier tiếng Anh. Cần rename → `compare_pdf.py` với `read_with_pypdf` / `read_with_pdfplumber` / `print_result(label, ...)`. **Commit riêng, chỉ rename, không đổi logic** — trộn rename vào commit tính năng là làm diff không đọc được.

> Quy ước từ nay: identifier (tên file, hàm, biến, hằng, khoá dict, tên bảng/cột) **100% tiếng Anh**. Comment và docstring vẫn tiếng Việt.

### Chưa có, phải cài

Không cần cài gì mới. `psycopg`, `sentence-transformers`, `python-dotenv` đã có từ tuần 4.
Container phải Up: `docker ps` → thấy `rag-pg`. Chưa Up: `cd week4/pgvector && docker compose up -d`.

---

## Cấu trúc

```
week5/indexTuning/
├── guide.md                 (file này)
├── hnsw_index.py            4 TODO   ← AI CORE
├── compare_topk.py          4 TODO   ← nợ T4 tuần 4
├── compare_chunking.py      4 TODO   ← PROJECT
└── filter_metadata.py       3 TODO   ← PROJECT
                            ── 15 TODO
```

Mọi file đều có cờ `--dry` (trừ `filter_metadata.py`): chạy self-check bằng dữ liệu giả, **không cần DB, không cần model**. Sửa–chạy trong 2 giây. Dùng `--dry` cho tới khi pass rồi mới đụng DB.

---

## Block 1 — AI CORE (09:00 – 11:00)

### 09:00–09:20 · Đọc trước khi gõ

Đọc mục *Indexing* trong docs pgvector (HNSW + IVFFlat). Đọc để trả lời được 1 câu: **vì sao index vector là "approximate" mà index B-tree thì không?** Chưa trả lời được thì đọc tiếp, đừng gõ code.

### 09:20–10:00 · `hnsw_index.py` — 4 TODO

- [X] **TODO 2** `plan_uses_index` — đọc plan text, xác định có Index Scan không. *(làm TRƯỚC: `--dry` test được ngay, không cần DB)*
- [X] **TODO 1** `explain_analyze` — chạy `EXPLAIN (ANALYZE, BUFFERS)`, móc `Execution Time` bằng regex
- [X] **TODO 3** `create_hnsw_index` — `USING hnsw (embedding vector_cosine_ops)` + đo build time
- [X] **TODO 4** `benchmark_query` — `SET hnsw.ef_search`, warm-up rồi mới đo

```bash
python hnsw_index.py --dry      # phải in "4/4 assert pass"
python hnsw_index.py            # đo thật
```

**Câu hỏi phải trả lời được sau block này** (viết vào `python-knowledge/Tuan-05_27Jul-02Aug_GD1-Nen-tang/T3/README.md`):

1. HNSW đánh đổi cái gì lấy cái gì? Nói bằng con số recall/tốc độ, không nói "nhanh hơn".
2. 3 lý do khiến Postgres **bỏ qua** index vector dù index tồn tại?
3. `m`, `ef_construction`, `ef_search` — cái nào đổi được sau khi build, cái nào không, vì sao?
4. Vì sao IVFFlat tạo trên bảng rỗng là sai, còn HNSW thì không?
5. Số thật: no-index ___ ms · ef_search=20 ___ ms · 40 ___ ms · 100 ___ ms · build ___ s · bảng ___ dòng.

### 10:00–10:45 · `compare_topk.py` — trả nợ T4 tuần 4, 4 TODO

- [X] **TODO 1** `parse_pgvector` — psycopg trả cột `embedding` là **CHUỖI**, không phải list *(`--dry` test được)*
- [X] **TODO 3** `topk_python` — cosine exact, sort **giảm** dần *(`--dry` test được)*
- [X] **TODO 2** `fetch_all_rows` — kéo hết embedding về RAM, cảnh báo khi chạm LIMIT
- [X] **TODO 4** `compare_rankings` — overlap@k, missing/extra ids, `max_score_gap`

```bash
python compare_topk.py --dry
python compare_topk.py "Embedding là gì và dùng để làm gì trong semantic search?"
```

**Câu hỏi phải trả lời được:**

1. Vì sao `compare_with_python()` của tuần 4 **không** trả được món nợ này? (gợi ý: nó so 2 kho khác nhau)
2. 2 bên khớp/lệch bao nhiêu (`overlap@5 = ?/5`), và **vì sao** — chọn 1 trong 3 nguyên nhân ở docstring, đừng đoán.
3. `<=>` trả distance còn Python trả similarity — quy đổi thế nào, sort ngược chiều nhau ra sao?
4. Ở 5.000 chunk bên nào nhanh hơn? Ở 5.000.000 chunk thì sao? Vì sao đổi chiều?

### 10:45–11:00 · Tiếng Anh — 5 từ lấy từ docs vừa đọc

`index` · `recall` · `approximate` · `trade-off` · `to scan`
Mỗi từ 1 câu **tự viết** về đúng việc vừa làm sáng nay. Không copy ví dụ trong từ điển.

---

## Block 2 — PROJECT (11:00 – 14:00, nghỉ trưa 12:00–13:00)

### 11:00–11:15 · Sửa 2 chỗ ở mục 0 trước

`blocks_to_chunks` thêm tham số (chỗ đau **1**) và dời guard trong `chunk_text` (chỗ đau **2**). 15 phút, làm xong mới chạy được task tiếp theo. Bỏ qua bước này thì compare_chunking sẽ ra số liệu **đẹp và sai**.

### 11:15–12:00 · `compare_chunking.py` — 4 TODO

- [X] **TODO 2** `chunk_stats` — n_chunks / avg / min / max / n_tiny / total_chars *(`--dry` test được)*
- [X] **TODO 1** `chunk_with_config` — gọi `blocks_to_chunks` với 2 cấu hình
- [X] **TODO 3** `rank_questions` — embed 1 lần cho cả kho, rồi top-3 cho 5 câu hỏi
- [X] **TODO 4** `print_comparison` — 2 cột cạnh nhau + cảnh báo `total_chars` lệch

```bash
python compare_chunking.py --dry
python compare_chunking.py "D:/Study/tai-lieu-test"
```

**Câu hỏi phải trả lời được:**

1. size gấp đôi thì số chunk có giảm một nửa? Tỷ lệ thật là bao nhiêu, **vì sao không phải 2:1**?
2. `total_chars` của 2 cấu hình chênh mấy %? Chênh nhiều nghĩa là gì?
3. Với 5 câu hỏi, top-1 bên nào **trả lời được** câu hỏi tốt hơn? Vì sao không so trực tiếp điểm similarity giữa 2 cấu hình được?
4. Chốt 1 cấu hình cho Dự án 1 (T5) — cấu hình nào, lý do bằng số.

### 13:00–13:40 · `filter_metadata.py` — 3 TODO

- [X] **TODO 1** `search_filtered` — `WHERE doc_type = %s` + thứ tự `%s` (bẫy chính)
- [X] **TODO 2** `inspect_filter_plan` — index còn dùng không: pre-filter hay post-filter?
- [X] **TODO 3** `compare_k_values` — k=3 vs k=5: `new_sources` (phủ rộng) vs `tail_gap` (loãng)

```bash
python filter_metadata.py "Vì sao phải chunk tài liệu trước khi embed?"
python filter_metadata.py "Vì sao phải chunk tài liệu trước khi embed?" --doc-type pdf
```

**Câu hỏi phải trả lời được:**

1. Câu có `WHERE doc_type` là pre-filter hay post-filter? Dựa vào **dòng nào** trong plan mà kết luận?
2. Post-filter có thể trả về **ít hơn k** dòng — vì sao? Đã quan sát được chưa?
3. `Rows Removed by Filter: N` — N là bao nhiêu, con số đó nói gì?
4. k=3 vs k=5: phủ rộng hơn hay loãng hơn? Trả lời bằng `new_sources` + `tail_gap`, chấm trên **≥3 câu hỏi**, không phải 1.

### 13:40–14:00 · Nghiệm thu — 5 phép thử, làm đủ cả 5

```bash
# 1. Ba self-check không cần hạ tầng
python hnsw_index.py --dry && python compare_topk.py --dry && python compare_chunking.py --dry
#    kỳ vọng: 3 dòng "pass", không exception nào

# 2. Index tồn tại và ĐÚNG opclass
docker exec -it rag-pg psql -U postgres -d study_rag -c "\d chunks"
#    kỳ vọng: có dòng chứa "hnsw" VÀ "vector_cosine_ops". Thiếu vector_cosine_ops = sai, làm lại TODO 3

# 3. Query thật dùng index
python hnsw_index.py
#    kỳ vọng: dòng "no index" -> [SEQ SCAN]; các dòng ef_search -> [index]; ms tăng dần theo ef_search
#    (bảng < 1000 dòng thì SEQ SCAN vẫn là bình thường — ghi số dòng vào note)

# 4. pgvector vs Python trên cùng kho
python compare_topk.py "Embedding là gì và dùng để làm gì trong semantic search?"
#    kỳ vọng: in overlap@5, same_order, max_score_gap; similarity 2 bên đều giảm dần, đều trong [-1, 1]

# 5. Chunking + filter
python compare_chunking.py "D:/Study/tai-lieu-test"
python filter_metadata.py "Vì sao phải chunk tài liệu trước khi embed?" --doc-type pdf
#    kỳ vọng: n_chunks(A) > n_chunks(B), không cảnh báo total_chars; mọi dòng trả về đúng doc_type='pdf'
```

**Tiêu chí "xong" của hôm nay:**

1. 15/15 TODO không còn `NotImplementedError`.
2. `python-knowledge/.../T3/README.md` có **bảng số đo thật**: 4 mốc ms + build time + số dòng bảng.
3. Trả lời được (bằng chữ, trong note) 3 lý do index vector bị bỏ qua.
4. Kết luận `overlap@5` giữa pgvector và Python exact, kèm nguyên nhân — chọn từ 3 nguyên nhân, không đoán.
5. Chốt **1** cấu hình chunking cho Dự án 1, có lý do bằng số.
6. Nhận xét k=3 vs k=5 dựa trên `new_sources` + `tail_gap` trên ≥3 câu hỏi.
7. 3 commit **riêng**: (a) `fix: dời guard overlap>=chunk_size ra trước vòng lặp`, (b) `refactor: rename so_sanh_pdf.py -> compare_pdf.py, identifier sang tiếng Anh`, (c) `feat(week5): index HNSW + đo retrieval`.

---

## Nhìn trước: hôm nay ăn khớp với phần còn lại của tuần thế nào

| Ngày | Việc | Phụ thuộc gì từ hôm nay |
|---|---|---|
| T4 29/07 | Đóng gói CLI + robustness | CLI cần tham số `--k` và `--doc-type` — chính 2 núm xoay đã đo hôm nay. Không đo trước thì mặc định là số bốc từ trên trời |
| T5 30/07 | **Dự án 1: RAG end-to-end** | dùng cấu hình chunking đã chốt + `k` đã chọn + index đã tạo. Đây là ngày "thu hoạch", không phải ngày thử nghiệm |
| T6 31/07 | Ôn tập + interview section E | 4 con số đo được hôm nay là câu trả lời cho câu hỏi "bạn từng tối ưu retrieval thế nào?" — có số liệu tự đo là điểm cộng lớn |
| Tuần 6 | Hybrid retrieval + reranking | `compare_topk.py` chính là bộ đo để chứng minh reranking có cải thiện thật. Tuần 7 nâng nó thành golden dataset + LLM-as-judge |

---

## Việc ngoài code trong plan hôm nay — đừng để trôi

- **14:00–15:00 · Tiếng Anh:** đọc to 1 mục docs pgvector (indexing/operators) 15 phút — chú ý trọng âm: *dimension · similarity · persistence · approximate*. Rồi nhật ký 3–4 câu tiếng Anh: hôm nay **đo** được gì sau khi thêm index (dùng số, không dùng "faster").
- **15:00–15:45 · Việc làm & nền tảng:**
  - Chuẩn bị **mock interview 17:00 hôm nay** — đọc lướt E22–E27 trong `docs/interview-question-bank.md`. Lịch cố định Thứ 3 17:00.
  - ⬅️ **Nợ hỏi lại từ Buổi 2:** đọc lại A6 (React.memo/useCallback) và C3 (N+1) — 15 phút, chỉ đọc để nhớ. Fullstack phải giữ song song với AI.
- **15:45–17:00 · Ôn tập & nhật ký:** note index HNSW/IVFFlat + số liệu vào `python-knowledge/Tuan-05_27Jul-02Aug_GD1-Nen-tang/T3/` — **kèm cả câu `EXPLAIN ANALYZE` và kết quả trước/sau**. Tick bảng lịch tuần + điền xlsx hôm nay (giờ thực tế, năng lượng, điều tốt nhất).

---

## Thứ tự đề nghị nếu bị hụt giờ

Cắt theo đúng thứ tự này, đừng cắt ngẫu nhiên:

1. **Không cắt:** `hnsw_index.py` (đây là mục tiêu chính của ngày) và 2 chỗ sửa ở mục 0.
2. **Cắt được:** `filter_metadata.py` TODO 2 (`inspect_filter_plan`) — quan sát plan hay nhưng T5 không cần nó để chạy.
3. **Cắt được:** `compare_topk.py` — nợ tuần 4, dời thêm 1 ngày cũng không chặn Dự án 1. Nhưng đừng dời quá T6, nợ 2 tuần là nợ chết.
4. **Đừng cắt:** phần chốt cấu hình chunking — T5 cần con số này để bắt đầu.
