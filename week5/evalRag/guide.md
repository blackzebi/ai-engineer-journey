# Guide — T5 tuần 5 (30/07): ✅ Dự án 1 — RAG end-to-end + bộ eval 10 câu

> **Mục tiêu ngày:** chốt mốc Dự án 1 — chạy trên bộ tài liệu **thật** (không phải notes của
> chính mình), có bảng eval 10 câu, README kể được **vì sao** chọn từng tham số.
>
> ```
> eval_cases.json (10 ca, tiêu chí viết TRƯỚC khi chạy)
>        │
>        ▼
>   run_eval.py ──► retriever.retrieve ──► stream_answer.stream_and_collect
>        │                (tái dùng nguyên si — KHÔNG viết lại pipeline)
>        ▼
>   grader.grade_case ──4 luật──► CaseVerdict
>        │              refusal · keywords · citation · source
>        ▼
>   eval_results.md ──► dán vào week5/README.md ──► push + pin repo
> ```

Hôm nay là ngày **ráp và chứng minh**, không phải ngày học khái niệm mới. Phần code chỉ
chiếm nửa buổi; nửa còn lại là bộ tài liệu thật, README, và push. Đừng sa đà tối ưu grader.

---

## 0. Đọc lại tuần trước: cái gì dùng lại, cái gì phải sửa

### Dùng lại NGUYÊN SI (đừng viết lại)

| Từ đâu | Thứ dùng lại | Vì sao vẫn dùng được |
|---|---|---|
| `week5/askCli/retriever.py` | `retrieve(question, k, doc_type)` | Đã lọc ngưỡng 0.35 bên trong, trả `RetrievedChunk` đủ metadata. Đúng thứ eval cần đo. |
| `week5/askCli/stream_answer.py` | `stream_and_collect`, `estimate_cost`, `log_query`, `get_client`, `NO_ANSWER` | Trả về `(text, usage)` — có sẵn token để tính chi phí từng ca. |
| `week5/askCli/errors.py` | `retry_with_backoff` (gián tiếp, qua `stream_and_collect`) | 10 ca liên tiếp rất dễ dính 429; retry đã nằm sẵn trong đường gọi. |
| `week5/ingestDocs/ingest_docs.py` | Toàn bộ pipeline ingest | Bộ tài liệu thật hôm nay chỉ là **đổi folder đầu vào**, không sửa code. Cache hash lo phần chạy lại. |
| `week4/ragLite/questions.json` | Ca N2 (giá cổ phiếu Apple) | Giữ 1 ca xuyên hai tuần để so được bộ eval cũ với mới. |

### PHẢI sửa — 4 chỗ đau thật

**1. `NO_ANSWER` đang tồn tại ở 4 nơi.** Đây là chỗ nguy hiểm nhất hôm nay.

| Nơi | Dạng |
|---|---|
| `week4/ragLite/prompt_rag.py` — `SYSTEM_PROMPT` | literal trong chuỗi |
| `week4/ragLite/prompt_rag.py` — `build_rag_prompt()` | literal lần 2, trong cùng file |
| `week4/ragLite/rag_qa.py` | `NO_ANSWER = "..."` |
| `week5/askCli/stream_answer.py` | `NO_ANSWER = "..."` |

Bộ eval chấm ca `no_answer` bằng cách so **đúng chuỗi này**. Sửa prompt một chữ (thêm dấu
phẩy, đổi "tài liệu" thành "các đoạn trích") → cả 3 ca bẫy FAIL cùng lúc, và mình sẽ đi
debug nhầm phía model mất cả buổi.

→ Hôm nay: `grader.py` **import** `NO_ANSWER`, tuyệt đối không gõ lại chuỗi.
→ Commit riêng (sau khi chốt dự án): đưa `NO_ANSWER` về `prompt_rag.py` làm nguồn duy nhất,
`build_rag_prompt` và `SYSTEM_PROMPT` nội suy từ hằng đó, hai file kia import về.

**2. `ask.run_once()` trả `int` (exit code), không trả câu trả lời.**
Nó in ra màn hình rồi vứt `text` đi → eval không bám vào được, phải tự ráp lại
`retrieve` + `stream_and_collect`. Không phải bug, là lỗi thiết kế: hàm trộn *làm việc* với
*in kết quả* thì không tái dùng được.
→ Hôm nay: chấp nhận, ráp lại trong `run_one_case`.
→ Commit riêng: tách `answer_question(question, k) -> AnswerResult` + `print_result(result)`;
`main()` và `run_eval` cùng gọi hàm đầu.

**3. `stream_and_collect()` in thẳng ra stdout, không có chế độ im lặng.**
Chạy 10 ca là màn hình ngập chữ, bảng kết quả trôi mất.
→ Hôm nay: chữa cháy bằng `contextlib.redirect_stdout(io.StringIO())`.
→ Commit riêng: thêm tham số `quiet: bool = False`; khi `quiet` thì bỏ `print` nhưng **vẫn
gom `parts`** — đừng bỏ luôn vòng lặp `text_stream`, vì `get_final_message()` cần nó chạy hết
mới có `usage` đầy đủ.

**4. `week5/askCli/retriever.py::search_chunks` và `week5/indexTuning/filter_metadata.py::search_filtered`
là hai câu SQL gần trùng.** Ghi chú "dedup bằng commit riêng" đã có trong docstring từ T4 mà
chưa làm — hôm nay vẫn **chưa phải lúc** (đang chốt mốc). Ghi vào mục "giới hạn đã biết" của
README để nó không biến mất.

### Việc nhỏ ngoài plan, đáng làm, commit RIÊNG

- `logs/` đã có trong `.gitignore` ✅. Nhưng `eval_results.json` chứa **toàn văn câu trả lời**
  → cân nhắc: commit nó (bằng chứng portfolio, nên commit) nhưng đừng để lọt nội dung tài
  liệu riêng tư. Nếu bộ tài liệu thật là tài liệu công ty cũ → **không commit** `eval_results.json`,
  chỉ commit `eval_results.md` đã cắt ngắn.
- Xoá khoá `_huong_dan` trong `eval_cases.json` trước khi commit.
- Identifier trong `week4`/`week5`: đã rà, **100% tiếng Anh**, không có gì phải rename. Tốt.

### Chưa có, phải cài

Không cài gì mới. `requirements.txt` đã đủ (`pypdf`, `python-docx`, `pdfplumber`, `psycopg`,
`sentence-transformers`, `anthropic`, `python-dotenv`).

---

## Cấu trúc

```
week5/evalRag/
├── eval_cases.json      10 ca — VIẾT TAY, phần việc thật của block sáng
├── eval_cases.py        2 TODO  · dataclass + validate + summarize
├── grader.py            5 TODO  · 4 luật chấm + gộp verdict   ← phần khó THẬT của hôm nay
├── run_eval.py          4 TODO  · harness + bảng markdown + tổng kết
└── guide.md             (file này)

week5/README.md          ← viết ở block chiều, dán bảng eval vào
```

Tổng **11 TODO**. Cả 3 file `.py` đều self-test được **không cần DB / model / API**.

---

## Block 1 — AI CORE (09:00 – 11:00)

Thứ tự cố ý: viết dataset trước → viết grader → mới chạy. Ngược lại là chấm theo cảm tính.

### 09:00–09:40 · `eval_cases.json` — viết 10 câu hỏi (KHÔNG code)

Đây là việc **tay** và là việc quan trọng nhất buổi sáng. Mở bộ tài liệu thật ra đọc,
viết câu hỏi từ nội dung thật.

- [X] Điền `corpus_note`: bộ tài liệu gồm gì
- [X] 7 ca answerable: 3 `easy` · 3 `medium` · 1 `hard`
- [X] 3 ca no_answer — đọc kỹ phần `note` của N1: bẫy **cùng chủ đề** là bẫy có giá trị nhất,
      vì ngưỡng similarity không cứu được, chỉ prompt grounding mới chặn nổi
- [X] Mỗi ca answerable: 2–4 `expect_keywords` (danh từ/thuật ngữ, đừng chọn từ nối)
- [X] Xoá `_huong_dan`

> **Viết xong 10 ca rồi mới chạy.** Nhìn kết quả rồi mới nghĩ tiêu chí là tự lừa mình —
> đây là nguyên văn cột Ghi chú trong plan hôm nay.

### 09:40–10:00 · `eval_cases.py`

- [X] **TODO 1** `validate_cases` — trả list lỗi, không raise ở lỗi đầu tiên
- [X] **TODO 2** `summarize_dataset` — số liệu sinh ra từ dữ liệu, không gõ tay vào README

```bash
python eval_cases.py                 # self-test: bộ hỏng ra 7 lỗi, bộ tốt ra 0 lỗi
python eval_cases.py eval_cases.json # kiểm file thật -> phải "✅ Bộ eval hợp lệ"
```

### 10:00–10:45 · `grader.py` — phần khó thật của hôm nay

- [X] **TODO 3** `check_refusal` — phân biệt **BỊA** và **TỪ CHỐI OAN** (cách chữa ngược nhau)
- [X] **TODO 4** `check_keywords` — `all()` chứ không `any()`, và trả sớm cho ca bẫy
- [X] **TODO 5** `check_citations` — bắt **trích dẫn ma** `[4]` khi chỉ có 3 nguồn
- [X] **TODO 6** `check_source` — đo tầng **truy xuất**, tách khỏi tầng sinh
- [X] **TODO 7** `grade_case` — chạy đủ 4 luật rồi mới gộp; tách nhánh `record.error`

```bash
python grader.py     # 8 kịch bản giả, đúng 2 ca PASS
```

**Câu hỏi phải trả lời được sau block này** (viết vào
`python-knowledge/Tuan-05_.../T5/README.md`):

1. Cách chấm của tuần 4 để lọt loại lỗi nào? Vì sao 1 luật là không đủ?
2. `all()` vs `any()` cho keyword matching — chọn cái nào và **bù lại** bằng gì?
3. "Trích dẫn ma" là gì, vì sao nó nguy hiểm hơn bịa nội dung?
4. Vì sao `check_source` phải tách khỏi `check_keywords`? Cho một tình huống mà keywords
   PASS nhưng source FAIL, và giải thích vì sao tình huống đó nguy hiểm.
5. Vì sao tỉ lệ PASS phải tách theo `answerable` / `no_answer`? Vặn `MIN_SIMILARITY` lên
   thì hai nhóm đi về hướng nào?

### 10:45–11:00 · 🗣 Tiếng Anh — 5 từ

Lấy từ chính code hôm nay: *refusal · grounding · verdict · proxy metric · false positive*.

---

## Block 2 — PROJECT (11:00 – 14:00, nghỉ trưa 12:00–13:00)

### 11:00–11:30 · `run_eval.py`

- [X] **TODO 8** `run_one_case` — nuốt stdout, bắt Exception, nhánh 0 chunk
- [X] **TODO 9** `run_all` — 1 client cho cả lượt, in tiến độ từng ca, log kèm `eval_run_id`
- [X] **TODO 10** `render_markdown_table` — escape `\|`, FAIL cũng phải in đủ lý do
- [X] **TODO 11** `print_summary` — tách 2 nhóm, trung vị không phải trung bình, guard chia 0

> Nhớ `import statistics` ở đầu file khi làm TODO 11.

```bash
python run_eval.py --dry     # KHÔNG tốn tiền — kiểm grader + bảng + tổng kết trước
```

Chỉ khi `--dry` chạy sạch mới sang bước sau. Debug bảng bằng tiền thật là lãng phí.

### 11:30–12:00 · Bộ tài liệu THẬT — ra khỏi sân nhà

Chọn 5–10 file PDF/DOCX **ngoài** notes của mình (sách, tài liệu dự án cũ, docs kỹ thuật).

```bash
cd ../ingestDocs
python ingest_docs.py "D:/Study/tai-lieu-that"
```

Đọc kỹ bảng "Phân bố theo định dạng" ở cuối:

- `pdf` 5 file mà chỉ ~10 chunk → gần như chắc chắn **PDF scan**, `extract_text()` trả rỗng.
  Đổi file khác, đừng đi làm OCR hôm nay.
- nhóm `doc_type = NULL` → dữ liệu cũ từ trước migration, dọn hoặc chấp nhận và ghi vào README.
- Chạy lại đúng lệnh trên lần 2 → phải ra `skip=<tất cả>`, `new=0`. Đây là phép thử idempotent.

### 13:00–13:40 · Chạy thật + sửa lỗi lộ ra

```bash
python run_eval.py                # 10 ca thật
python run_eval.py --only A7      # lặp nhanh trên ca khó, đỡ tốn tiền
```

Lỗi hay lộ ra ở đây, theo thứ tự nên nghi ngờ:

1. **Từ chối oan hàng loạt** → `MIN_SIMILARITY = 0.35` quá cao cho bộ tài liệu này. Xem cột
   similarity trong `eval_results.json`, đặt ngưỡng vào **khoảng trống** giữa nhóm answerable
   và nhóm bẫy. Chỉnh bằng số đo, không bằng cảm giác.
2. **Ca `hard` (A7) FAIL** → k=3 không đủ để ghép 2 nguồn. Thử `--k 5`, ghi lại **cả hai** số
   liệu. Nếu vẫn FAIL: đây đúng là chỗ hybrid retrieval + reranking của tuần 6 sẽ chữa →
   **ghi vào README**, đừng cố sửa hôm nay.
3. **Câu trả lời trích đúng từ khoá nhưng nguồn sai file** → chunking hoặc normalize PDF.
   Xem `n_chunks` và `top_similarity` trước khi đụng vào chunker.

> **Đừng vặn `MIN_SIMILARITY` cho tới khi đủ 10/10.** Đó là overfit lên chính bộ eval của
> mình — bộ eval mất luôn giá trị. 8/10 kèm phân tích 2 ca FAIL mạnh hơn 10/10 đã tinh chỉnh.

### 13:40–14:00 · `week5/README.md` — portfolio

Khung đã dựng sẵn ở `week5/README.md`, điền vào. Phần **"Vì sao chọn tham số này"** là thứ
phân biệt project học việc với project của kỹ sư — mỗi con số phải có 1 câu lý do và
lý do đó phải là **số đo**, không phải "thấy nhiều người dùng thế".

- [X] Dán khối tổng kết + bảng từ `eval_results.md`
- [X] Ghi đủ mục "Giới hạn đã biết" (4 chỗ ở mục 0 + ca FAIL)
- [ ] Cập nhật `README.md` gốc: đổi "Featured project" sang Dự án 1, thêm dòng vào bảng tuần

```bash
cd ../..
git add week5/evalRag week5/README.md README.md
git commit -m "week5: 10-case eval harness with 4 grading rules + project 1 README"
git push
```

- [ ] Pin repo trên GitHub

### 13:55 · Nghiệm thu — 6 phép thử, làm đủ cả 6

```bash
python eval_cases.py                    # bộ hỏng 7 lỗi · bộ tốt 0 lỗi
python eval_cases.py eval_cases.json    # "✅ Bộ eval hợp lệ" + 7/3
python grader.py                        # đúng 2/8 kịch bản PASS
python run_eval.py --dry                # 9/10, A7 FAIL, KHÔNG gọi API
python run_eval.py                      # chạy thật, chi phí > 0 và < 0.1 USD
grep "<run_id>" ../askCli/logs/queries.jsonl | wc -l   # = 10, ca N3 có input_tokens 0
```

**Tiêu chí "xong" của hôm nay:**

- `eval_results.md` tồn tại, bảng 6 cột thẳng hàng, có **cả** ca PASS và ca FAIL (nếu có)
- `week5/README.md` có bảng eval + mục "vì sao chọn tham số" + mục "giới hạn đã biết"
- Repo đã push, README gốc trỏ tới Dự án 1
- Trả lời được 5 câu hỏi ở block 1 mà không mở lại code

---

## Nhìn trước: hôm nay ăn khớp với phần còn lại thế nào

| Ngày | Việc | Phụ thuộc gì từ hôm nay |
|---|---|---|
| T6 tuần 5 (31/07) | Ôn tập + interview Section E | Dùng chính Dự án 1 làm câu trả lời "tell me about a project" |
| T2 tuần 6 (03/08) | Keyword search BM25 | `find_repeated_lines` trong `normalize.py` chính là ý tưởng IDF — đọc lại trước |
| T3 tuần 6 (04/08) | Hybrid retrieval + RRF | Cần **đúng** `eval_results.json` hôm nay làm mốc gốc: hybrid có tốt hơn không thì so vào đây |
| T4 tuần 6 (05/08) | Reranking cross-encoder | Ca `hard` A7 FAIL hôm nay là ca để chứng minh rerank có tác dụng |
| Tuần 7 | Golden dataset + LLM-as-judge | `eval_results.json` + `logs/queries.jsonl` (tag `outcome=eval`) là đầu vào trực tiếp |

Nói cách khác: **bộ eval hôm nay là thước đo của cả tuần 6 và 7.** Làm ẩu hôm nay thì mọi
cải tiến 2 tuần tới đều không chứng minh được.

---

## Việc ngoài code trong plan hôm nay — đừng để trôi

**🗣 14:00–15:00 · Tiếng Anh**

- Podcast / docs 15–20 phút + 5 từ
- Viết 5 câu tiếng Anh mô tả Dự án 1 (dùng lại project pitch hôm T4), **nói to 2 lần**
  ⬅️ đây sẽ là câu trả lời cho *"tell me about a project you built"*

**💼 15:00–15:45 · Việc làm & nền tảng**

- Cập nhật CV + LinkedIn: thêm Dự án 1 vào mục Projects, **1 dòng có kết quả đo được**
  (ví dụ: *"RAG Q&A over mixed-format documents; 10-case eval, N/10 pass, ~$0.00X per query"*)
  ⬅️ từ tuần 1 CV đã chờ đúng lúc này
- Rà lại repo: 5 mini-project + Dự án 1; README gốc có mục lục dẫn tới từng project

**📓 15:45–17:00 · Ôn tập & nhật ký**

- Note luồng Dự án 1 hoàn chỉnh + bảng eval vào `python-knowledge/Tuan-05_.../T5/`
- Tick bảng lịch tuần + điền `T5_30-07_Du-An-1-RAG-End-to-End.xlsx`

---

## Nếu hôm nay bị nén thời gian (đang chạy bù để sang tuần 6)

Thứ tự **bỏ được**, từ dưới lên:

1. `check_source` (TODO 6) — trả `CheckResult("source", True, "bỏ qua")` tạm, ghi vào README
2. `summarize_dataset` (TODO 2) — gõ tay số liệu vào README, nhưng phải quay lại làm
3. Ca `hard` A7 — hạ xuống `medium`, ghi rõ trong README là bộ eval chưa có ca multi-hop

Thứ **không được bỏ**, vì tuần 6 phụ thuộc trực tiếp:

- 10 ca viết tay trong `eval_cases.json` (không có nó thì tuần 6 không có mốc để so)
- `check_refusal` + `check_keywords` (TODO 3, 4)
- Ingest bộ tài liệu thật + `eval_results.json` của một lượt chạy thật
