# Guide — T4 tuần 5 (29/07): Đóng gói — CLI, streaming, xử lý lỗi

> **Mục tiêu ngày:** Biến pipeline thành thứ NGƯỜI KHÁC clone về chạy được — CLI, streaming,
> lỗi được xử lý tử tế, key không lộ. Hôm nay không thêm tính năng AI nào mới; hôm nay là ngày
> làm cho thứ đã có trở nên **dùng được bởi người không phải mình**.
>
> ```
>   argv ──parse_args──► question, k, doc_type
>                             │
>              validate_question          ← chặn câu rỗng TRƯỚC khi tốn tiền
>                             │
>              retriever.retrieve ──► chunk (đã lọc ngưỡng similarity)
>                             │              └─ rỗng ⇒ in "không tìm thấy", KHÔNG gọi LLM
>                             │
>              stream_and_collect ──► chữ hiện DẦN + usage(token)
>                             │
>              format_sources · estimate_cost · log_query(JSONL)
>
>   Mọi lỗi ──AskError──► main() ──► thông điệp gọn + exit code
>                            ▲
>                     CHỈ chỗ này được exit
> ```

---

## 0. Đọc lại tuần trước: cái gì dùng lại, cái gì phải sửa

### Dùng lại NGUYÊN SI (đừng viết lại)

| Từ đâu | Thứ dùng lại | Vì sao vẫn dùng được |
|---|---|---|
| `week4/ragLite/prompt_rag.py` | `SYSTEM_PROMPT`, `build_rag_prompt` | Grounding prompt vẫn đúng nguyên. Mẹo: nhét `title · trang 12` vào chỗ `source` qua `as_prompt_chunk()` là model trích được tới trang mà **không sửa một dòng nào** của tuần 4 |
| `week4/pgvector/vector_ops.py` | `get_conn`, `to_pgvector` | Connection + literal vector không đổi |
| `week4/ragPipeline/search_pg.py` | `embed_query` | Cùng model, cùng 384 chiều |
| `week5/ingestDocs/db_docs.py` | schema `title/page/heading/doc_type` | T2 đã nhét metadata vào DB, tới hôm nay mới thực sự **dùng** nó |
| `week5/indexTuning/hnsw_index.py` | index HNSW đã tạo | Query hôm nay giữ `ORDER BY embedding <=> %s` nên vẫn ăn index |

### PHẢI sửa — 5 chỗ đau thật

**1. `requirements.txt` đang là UTF-16 → người clone về `pip install -r` sẽ lỗi.**
Mở file bằng `cat` thấy chữ giãn cách kiểu `a n n o t a t e d`: đó là byte NUL xen giữa,
dấu vết của `pip freeze > requirements.txt` trong PowerShell (toán tử `>` của PowerShell
ghi UTF-16LE). Đây **đúng là loại lỗi mà mục tiêu hôm nay nhắm tới** — repo trông đầy đủ
nhưng bước đầu tiên của người lạ đã hỏng.

```powershell
pip freeze | Out-File -Encoding utf8 requirements.txt
# rồi kiểm chứng: file phải đọc bình thường, không có khoảng trắng xen giữa từng chữ cái
```
Nhân tiện bổ sung `streamlit` (nếu làm task tuỳ chọn) và ghi rõ trong README lệnh cài.

**2. `.env.example` lệch với thực tế.** Đang có `OPENAI_API_KEY` mà `.env` thật không dùng;
đang **thiếu `DOCS_DIR`** trong khi `ingest_docs.py` đọc `os.getenv("DOCS_DIR")` với mặc định
hardcode `D:\Study\tai-lieu-test` — người khác clone về chắc chắn không có đường dẫn đó.
Sửa `.env.example` thành đúng 3 biến đang dùng thật (`ANTHROPIC_API_KEY`, `DATABASE_URL`,
`DOCS_DIR`), giữ placeholder, **không có key thật**.

**3. `.gitignore` thiếu `logs/`.** Hôm nay bắt đầu ghi `askCli/logs/queries.jsonl` chứa câu
hỏi thật. Thêm `logs/` **trước** khi chạy `ask.py` lần đầu — commit nhầm rồi thì xoá khỏi
lịch sử git rất phiền. Tiện thể thêm `.vscode/` (đang có trong repo, chưa ignore).

**4. `rag_qa.py::retrieve` gọi `raise SystemExit` từ trong hàm thư viện.** Hàm sâu tự quyết
định giết cả tiến trình. Hậu quả: dùng lại nó trong Streamlit/FastAPI là sập server, và không
viết được test "DB hỏng thì báo đúng lỗi". Hôm nay `retriever.retrieve` làm đúng: **raise
`AskError`, chỉ `main()` mới exit**. Không cần sửa file tuần 4 — nó là bản ghi lịch sử học tập;
ghi vào note lý do bản mới khác.

**5. Identifier tiếng Việt còn sót — commit RIÊNG, chỉ rename, không đổi logic.**
Trộn rename vào commit tính năng làm diff không đọc nổi khi review.

| File | ❌ Hiện tại | ✅ Đổi thành |
|---|---|---|
| `week5/ingestDocs/normalize.py` | `def nen_noi(a, b)` | `def should_join(a, b)` |
| `week5/ingestDocs/normalize.py` | `nguong = len(pages) * ratio` | `threshold = len(pages) * ratio` |
| `week5/ingestDocs/compare_pdf.py` | `n_dong = len(...)` | `line_count = len(...)` |
| `week5/ingestDocs/compare_pdf.py` | docstring + `Dùng: python so_sanh_pdf.py` | đổi thành `compare_pdf.py` (file đã rename, nội dung quên sửa) |

Dọn kèm: `week5/ingestDocs/__pycache__/so_sanh_pdf.cpython-310.pyc` và
`week4/__pycache__/` (mồ côi, không module nào ở đó) — `.pyc` cũ của file đã đổi tên chỉ gây
nhiễu khi grep.

### Dedup với T3 — làm SAU khi `ask.py` chạy được

T3 đã xong, `filter_metadata.search_filtered` chạy được rồi. Nhưng `retriever.search_chunks`
hôm nay **vẫn viết câu SQL riêng**, có lý do: `search_filtered` trả **6 cột**, thiếu `title`
và `heading`. Thiếu title thì trích nguồn chỉ hiện đường dẫn file thô; thiếu heading thì
docx/md/txt (page luôn NULL) mất sạch địa chỉ để chỉ — mà trích nguồn "Báo cáo Q3 · trang 12"
chính là thứ hôm nay phải làm ra.

Hệ quả là **hai câu SQL gần giống hệt nhau đang nằm ở hai file**. Sau vài tuần chúng sẽ phân
kỳ: sửa cột ở file này quên file kia, và kiểu bug đó không báo lỗi. Dọn bằng **một commit
riêng**, làm SAU khi `ask.py` chạy được — đừng refactor giữa lúc đang làm tính năng, hỏng thì
không biết hỏng vì đâu.

Hướng dọn: giữ `retriever.search_chunks` (8 cột) làm bản gốc **duy nhất**, cho
`search_filtered` gọi lại nó rồi cắt bớt cột — giữ nguyên chữ ký cũ để `compare_k_values`
và `inspect_filter_plan` của T3 không phải sửa một dòng nào:

```python
# week5/indexTuning/filter_metadata.py
sys.path.append(os.path.join(HERE, "..", "askCli"))
from retriever import search_chunks

def search_filtered(conn, query_vector, doc_type=None, k=5):
    """Giữ nguyên 6 cột cũ; SQL thật nằm ở retriever.search_chunks."""
    chunks = search_chunks(conn, query_vector, k=k, doc_type=doc_type)
    return [(c.id, c.source, c.doc_type, c.content, c.page, c.similarity) for c in chunks]
```

Nghiệm thu refactor: chạy lại `python filter_metadata.py --doc-type pdf`, **số liệu phải
không đổi**. Đổi số nghĩa là đã sửa nhầm logic chứ không phải dọn code.

> **Tận dụng ngay hôm nay:** `inspect_filter_plan` của T3 đã đo được post-filter thật.
> Chạy `python ../indexTuning/filter_metadata.py --doc-type pdf` một lần, dán
> `filter_position` + `execution_ms` của hai câu (có WHERE / không WHERE) vào note T4.
> Đó là câu trả lời **có số liệu** cho *"vì sao `--doc-type pdf` đôi khi trả về ít hơn `--k`"*
> — hỏi đúng câu này là chuyện rất hay gặp khi phỏng vấn về vector search.

### Chưa có, phải cài

Không cần cài gì mới cho phần bắt buộc (`anthropic`, `psycopg`, `python-dotenv` đã có).
Chỉ khi làm task tuỳ chọn:

```bash
pip install streamlit
```

---

## Cấu trúc

```
week5/askCli/
├── guide.md               ← file này
├── errors.py              4 TODO   ← chạy được ngay, không cần DB/API   ⭐ LÀM TRƯỚC
├── retriever.py           4 TODO   ← self-test bằng chunk giả
├── stream_answer.py       3 TODO   ← self-test bằng client giả (không tốn tiền)
├── ask.py                 2 TODO   ← ráp lại; run_once + main viết sẵn
└── logs/queries.jsonl              ← tự sinh khi chạy (nhớ .gitignore)
```

**Tổng 13 TODO.** Thứ tự trên đã xếp theo nguyên tắc: file test được **không cần hạ tầng**
làm trước. `errors.py` sửa–chạy mất 1 giây; `ask.py` cần cả Docker lẫn API key. Vòng lặp ngắn
trước, dài sau.

---

## Block 1 — AI CORE (09:00 – 11:00)

### 09:00–09:20 · Đọc, chưa gõ

Đọc PHẦN 1 của cả 3 hàm trong `errors.py` và docstring đầu `stream_answer.py`. Hai câu hỏi
của block sáng nằm hết ở đó — hiểu rồi mới gõ thì gõ rất nhanh.

### 09:20–10:10 · `errors.py` — danh mục lỗi

- [X] **TODO 1** `classify_exception` — exception của thư viện → `ErrorKind` của mình
- [X] **TODO 2** `format_user_error` — điền đủ **8 loại lỗi**, mỗi loại 1 câu "phải làm gì"
- [X] **TODO 3** `validate_question` — chặn câu rỗng/quá ngắn/quá dài
- [X] **TODO 4** `retry_with_backoff` — chỉ retry lỗi TẠM THỜI, giãn cách gấp đôi

TODO 2 chính là task *"liệt kê lỗi PHẢI xử lý"* của xlsx — bảng `MESSAGES` **là** bản liệt kê
đó, viết dưới dạng code chạy được thay vì gạch đầu dòng trong note.

```bash
python errors.py      # phải chạy sạch, không cần DB/API/model
```

### 10:10–10:45 · `stream_answer.py` — streaming (TODO 1 + 2)

- [X] **TODO 1** `stream_and_collect` — vừa in dần vừa gom text, lấy `usage`
- [X] **TODO 2** `estimate_cost` — quy token ra USD, tách input/output

```bash
python stream_answer.py    # client GIẢ — không cần API key, không tốn một đồng nào
```

**Câu hỏi phải trả lời được sau block này** (viết vào `python-knowledge/Tuan-05/T4/README.md`):

1. Streaming làm **tổng thời gian** ngắn lại hay không? Vậy thứ thực sự thay đổi là gì?
2. Quên `flush=True` thì hỏng ở đâu, và vì sao code vẫn **trông** đúng?
3. Liệt kê 8 loại lỗi của app RAG, mỗi loại 1 dòng: *khi nào xảy ra → app làm gì → exit code*.
4. Vì sao `DB_UNAVAILABLE` **không** nằm trong `RETRYABLE_KINDS` còn `API_RATE_LIMIT` thì có?
5. Trong RAG, input token hay output token thường tốn tiền hơn? Tăng `--k` từ 3 lên 10 thì
   chi phí mỗi câu hỏi đổi thế nào?

### 10:45–11:00 · Tiếng Anh (nợ tuần 4 — ⬅️ đừng bỏ)

5 từ + luyện nói *"What is RAG?"* bằng 4–5 câu đơn giản.
Chất liệu: `python-knowledge/Tuan-04/T5/nhat-ky-tieng-anh.md`.
Gợi ý 5 từ lấy thẳng từ code hôm nay: *streaming · retry · backoff · threshold · idempotent*.

---

## Block 2 — PROJECT (11:00 – 14:00, nghỉ trưa 12:00–13:00)

### 11:00–11:15 · Config sạch — làm TRƯỚC khi chạy `ask.py` lần đầu

Thứ tự này quan trọng: thêm `logs/` vào `.gitignore` **trước** khi sinh ra file log đầu tiên.

- [X] `.gitignore`: thêm `logs/` và `.vscode/`
- [X] `.env.example`: bỏ `OPENAI_API_KEY`, thêm `DOCS_DIR`, kiểm lại **không có key thật**
- [X] `requirements.txt`: ghi lại bằng UTF-8 (xem mục PHẢI sửa #1)
- [X] `git status` — xác nhận `.env` **không** xuất hiện trong danh sách

### 11:15–12:00 · `retriever.py` — metadata + ngưỡng similarity

- [X] **TODO 1** `search_chunks` — query 8 cột, WHERE `doc_type` tuỳ chọn
- [X] **TODO 2** `apply_similarity_threshold` — lớp chống bịa thứ hai
- [X] **TODO 3** `format_sources` — khối trích nguồn, xử lý `page = None`
- [X] **TODO 4** `retrieve` — ráp lại, ném `AskError` thay vì `SystemExit`

```bash
python retriever.py                       # chunk giả, chạy tức thì
python retriever.py "embedding là gì?"    # thật (cần Docker Up + đã ingest)
```

### 13:00–13:30 · `ask.py` + `stream_answer.py` TODO 3

- [X] **TODO 3** (`stream_answer.py`) `log_query` — ghi JSONL
- [X] **TODO 1** (`ask.py`) `parse_args` — argparse, có `-h` và `choices`
- [X] **TODO 2** (`ask.py`) `check_env` — liệt kê **hết** biến thiếu trong một lần

### 13:30–13:45 · Tuỳ chọn — Streamlit

**Chỉ làm nếu CLI đã xong và chạy mượt.** Đừng đánh đổi CLI ổn định lấy UI đẹp: người xem
portfolio đọc `ask.py` và README, không ai chạy thử UI của bạn. Nếu làm: 1 file
`app_streamlit.py`, gọi thẳng `retrieve` + `stream_and_collect`, không copy logic sang.

### 13:45–14:00 · Nghiệm thu — 6 phép thử, làm đủ cả 6

```bash
cd week5/askCli

python ask.py -h                        # → bảng hướng dẫn có ví dụ
python ask.py --check                   # → "Cấu hình đủ"
python ask.py ""                        # → 1 dòng gọn, KHÔNG có chữ "Traceback"
python ask.py "embedding là gì?"        # → chữ hiện DẦN, nguồn [1][2], dòng token/chi phí
python ask.py "kết luận báo cáo?" --k 5 --doc-type pdf   # → mọi nguồn đều là .pdf
python ask.py "giá vàng hôm nay bao nhiêu?"              # → "không tìm thấy", 0 token

docker compose stop                     # (trong week4/pgvector)
python ask.py "embedding là gì?"        # → gợi ý "docker compose up -d", KHÔNG traceback
docker compose start
```

**Tiêu chí "xong" của hôm nay:**

- [X] Không có tình huống nào in ra `Traceback` cho người dùng, trừ nhánh `UNKNOWN` (cố ý)
- [X] `python ask.py -h` đủ để người lạ biết cách dùng, không phải đọc code
- [X] `git status` sạch — `.env` và `logs/` không lọt vào
- [X] `logs/queries.jsonl` có dòng cho **cả** câu trả lời được lẫn câu "không tìm thấy"
- [X] Nói được một câu: *"mỗi câu hỏi tốn khoảng ___ USD, trong đó ___% là input token"*
- [X] Commit riêng phần rename identifier, không trộn vào commit tính năng

---

## Nhìn trước: hôm nay ăn khớp với phần còn lại của tuần thế nào

| Ngày | Việc | Phụ thuộc gì từ hôm nay |
|---|---|---|
| T5 30/07 | **Dự án 1 chạy end-to-end** | `ask.py` là mặt tiền để demo. Không có nó thì T5 chỉ có script rời |
| T5 30/07 | README + pitch dự án | Số liệu token/chi phí trong `logs/queries.jsonl` |
| T6 31/07 | Ôn tập + phỏng vấn section E | Câu chuyện "xử lý lỗi + đo chi phí" là chất liệu trả lời trực tiếp |
| Tuần 7 | Golden dataset + LLM-as-judge | `logs/queries.jsonl` **chính là** dataset thô đầu tiên — hôm nay ghi để tuần 7 có cái mà đo |
| Tuần 9 | FastAPI + Docker | `retrieve` / `stream_and_collect` raise thay vì exit ⇒ bọc thành API không phải viết lại |

---

## Việc ngoài code trong plan hôm nay — đừng để trôi

- **10:45–11:00 · Tiếng Anh** — 5 từ + ⬅️ *(nợ tuần 4)* luyện nói "What is RAG?" 4–5 câu đơn giản.
  Chất liệu có sẵn ở `python-knowledge/Tuan-04/T5/nhat-ky-tieng-anh.md`.
- **14:00–15:00 · Tiếng Anh** — podcast/video kỹ thuật 15–20ph + 5 từ. Không nghe được thì
  lấy từ trong docs Anthropic streaming / Streamlit — miễn là có đủ 5 từ ghi vào file.
  Nhật ký tiếng Anh 3–4 câu về việc đóng gói project.
- **15:00–15:45 · Việc làm** — nháp *project pitch* 5 câu cho Dự án 1: vấn đề → cách làm →
  kết quả **đo được** (dùng chính con số token/chi phí sinh ra hôm nay). Dùng lại cho CV,
  LinkedIn và câu G8 *"what have you built outside work"*. Theo dõi BHTN + duy trì LinkedIn.
- **15:45–17:00 · Ôn tập** — note streaming + danh mục lỗi vào `python-knowledge/Tuan-05/T4/`,
  ghi cả **hành vi mong muốn** cho từng loại lỗi. Tick bảng lịch tuần + điền xlsx hôm nay.
