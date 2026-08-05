<!--
KHUNG README PORTFOLIO — Dự án 1. Điền vào block 13:40–14:00.

LUẬT VIẾT (đọc trước khi gõ):
  · Viết bằng TIẾNG ANH — đây là trang người tuyển dụng đọc. Comment/ghi chú riêng thì tiếng Việt.
  · Mọi CON SỐ phải lấy từ eval_results.md, không gõ ước lượng.
  · Mục "Why these parameters" là mục quan trọng nhất. Mỗi tham số = 1 câu lý do + 1 SỐ ĐO.
    "Vì thấy nhiều người dùng 800" KHÔNG phải lý do.
  · Mục "Known limitations" viết thật. FAIL ghi vào đây là điểm cộng, giấu đi là điểm trừ.
  · Xoá toàn bộ comment HTML này trước khi commit.
-->

# Project 1 — Grounded Q&A over mixed-format documents

<!-- TODO: 2–3 câu. Vào gì → ra gì → điều gì khiến nó không tầm thường.
     Gợi ý góc kể: điểm kỹ thuật KHÔNG phải "làm LLM trả lời được" mà là
     "làm nó TỪ CHỐI đúng lúc, và mọi câu đều truy được về trang gốc". -->

```
question (VI) ──embed──► pgvector top-k ──threshold 0.35──► grounded prompt ──► Claude
                              │                    │                              │
                              │                    └─ below ⇒ refuse, 0 tokens    │
                              └─ chunk + doc_type + title + page/heading ─────────┴──► answer + [n] citations
```

## What it does

<!-- TODO: 3–5 gạch đầu dòng, mỗi dòng 1 năng lực CÓ THỂ KIỂM CHỨNG:
     - ingest PDF/DOCX/MD/TXT, giữ page/heading để trích nguồn
     - làm sạch PDF: header/footer lặp, số trang, câu bị ngắt dòng
     - hai lớp chống bịa: ngưỡng similarity (trước khi sinh) + prompt grounding (lúc sinh)
     - incremental ingest bằng file hash — chạy lại không nhân đôi dữ liệu
     - CLI streaming, có exit code phân loại, log JSONL mỗi truy vấn -->

## Corpus

<!-- TODO: mô tả bộ tài liệu thật đã ingest — bao nhiêu file, định dạng nào, chủ đề gì,
     tổng bao nhiêu chunk. Lấy số từ bảng "Phân bố theo định dạng" của ingest_docs.py.
     Nếu tài liệu riêng tư thì mô tả LOẠI, đừng nêu tên. -->

| doc_type | files | chunks |
|---|---|---|
| pdf | | |
| docx | | |
| md | | |

## Evaluation

<!-- TODO: dán KHỐI TỔNG KẾT + BẢNG từ evalRag/eval_results.md vào đây. -->

10 cases: 7 answerable (3 easy / 3 medium / 1 multi-hop) + 3 deliberately unanswerable.
Grading criteria were written **before** the first run — see `evalRag/eval_cases.json`.

Four independent rules per case:

| Rule | What it catches |
|---|---|
| `refusal` | fabrication on unanswerable questions, **and** false refusal on answerable ones |
| `keywords` | answer omits the concepts a correct answer must mention |
| `citation` | `[n]` markers pointing at sources that do not exist |
| `source` | retrieval pulled the wrong document even when the answer sounds right |

<!-- TODO: bảng kết quả ở đây -->

## Why these parameters

<!-- TODO: mỗi dòng 1 tham số + LÝ DO KÈM SỐ ĐO. Đây là mục người đọc kỹ nhất. -->

| Parameter | Value | Why — with the measurement behind it |
|---|---|---|
| chunk size / overlap | 800 / 100 | <!-- TODO: dẫn số liệu từ week5/indexTuning/compare_chunking.py --> |
| embedding model | `paraphrase-multilingual-MiniLM-L12-v2` (384d) | <!-- TODO: vì sao multilingual, vì sao chạy local --> |
| top-k | 3 | <!-- TODO: chi phí input token tăng ~tuyến tính theo k; đo được gì khi k=5? --> |
| similarity threshold | 0.35 | <!-- TODO: khoảng cách giữa top-1 của nhóm answerable và nhóm bẫy là bao nhiêu? --> |
| generation model | `claude-haiku-4-5` | <!-- TODO: chi phí thật đo được / câu hỏi --> |
| index | HNSW (`vector_cosine_ops`) | <!-- TODO: ở kích thước corpus này planner chọn gì? Trung thực nếu vẫn là Seq Scan --> |

## Known limitations

<!-- TODO: viết thật. Danh sách mồi — giữ những cái đúng, bỏ cái không áp dụng:
     - grading by keyword matching là PROXY: bắt được "thiếu khái niệm",
       không bắt được "nhắc đúng từ nhưng lập luận sai". LLM-as-judge: week 7.
     - ca multi-hop (A7) — kết quả thật là gì? nếu FAIL, nói rõ hybrid retrieval
       + reranking (week 6) là hướng chữa.
     - post-filter khi lọc doc_type: xin k=5 có thể nhận về ít hơn 5.
     - PDF scan không có OCR → 0 ký tự, pipeline bỏ qua file đó.
     - `run_once()` trộn việc với in ấn → eval phải ráp lại pipeline.
     - `NO_ANSWER` bị nhân bản ở 4 nơi trong repo.
     - hai câu SQL gần trùng giữa retriever.py và filter_metadata.py.
     - text trong bảng của DOCX chưa được đọc (`doc.paragraphs` bỏ qua `doc.tables`). -->

## Stack

Python 3.10+ · PostgreSQL 16 + pgvector (Docker) · sentence-transformers ·
Anthropic Claude (Haiku) · psycopg 3

## Run

```bash
# 1. database
cd week4/pgvector && docker compose up -d

# 2. ingest your own documents
cd ../../week5/ingestDocs && python ingest_docs.py "<folder>"

# 3. ask
cd ../askCli && python ask.py "<your question>" --k 3

# 4. reproduce the evaluation
cd ../evalRag && python run_eval.py
```

## Layout

| Module | Role |
|---|---|
| `week5/ingestDocs` | loaders (PDF/DOCX/MD/TXT) · PDF normalisation · schema migration · incremental ingest |
| `week5/indexTuning` | HNSW index, chunking and top-k experiments — the measurements behind the table above |
| `week5/askCli` | CLI: retrieval with threshold, streaming answers, error taxonomy, cost logging |
| `week5/evalRag` | 10-case evaluation set, 4-rule grader, markdown report |
