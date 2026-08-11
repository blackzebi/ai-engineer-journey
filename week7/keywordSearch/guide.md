# Guide — T2 tuần 7 (10/08): Keyword search (Postgres full-text) + bộ câu hỏi đo

> **Mục tiêu ngày:** Dựng nhánh tìm kiếm THỨ HAI — theo TỪ KHOÁ, thứ mà vector search làm dở.
> Cuối ngày phải có bộ câu hỏi để cả tuần đo được.
>
> ```
>                            câu hỏi tiếng Việt
>                                   |
>          +------------------------+------------------------+
>          |                                                 |
>   NHÁNH 1 (đã có, tuần 5)                        NHÁNH 2 (dựng hôm nay)
>   embed --> chunks.embedding                     plainto_tsquery --> chunks.tsv
>   `<=>` + index HNSW                             `@@` + index GIN
>   similarity 0..1, có ngưỡng 0.35                ts_rank, KHÔNG có ngưỡng
>          |                                                 |
>          +----------> compare_branches.py <----------------+
>                                   |
>                    bảng "câu nào vector thắng / keyword thắng"
>                                   |
>                        T3: RRF trộn theo THỨ HẠNG
> ```

Nhịp 7h (9:00–17:00, nghỉ 12:00–13:00). Ít task hơn tuần trước, mỗi task dài hơn — theo review tuần 5.

---

## 0. Đọc lại tuần trước: cái gì dùng lại, cái gì phải sửa

### Dùng lại NGUYÊN SI (đừng viết lại)

| Từ đâu | Thứ dùng lại | Vì sao vẫn dùng được |
|---|---|---|
| `week4/pgvector/vector_ops.py` | `get_conn` | Không đổi gì, hôm nay vẫn cùng một DB |
| `week4/ragPipeline/search_pg.py` | `embed_query` | Có cache `_model` cấp module → 8 câu chỉ nạp model 1 lần |
| `week5/askCli/retriever.py` | `search_chunks` | Bản 8 cột đầy đủ metadata — đúng thứ cần để so `source` |
| `week5/ingestDocs/db_docs.py` | schema `chunks` + `documents` | Hôm nay chỉ CỘNG THÊM cột `tsv`, không đụng gì cũ |
| `week5/indexTuning/hnsw_index.py` | mẫu `EXPLAIN ANALYZE` + cách đọc plan | Ý tưởng tái dùng — **nhưng không import được, xem bên dưới** |
| `week5/evalRag/eval_cases.py` | khuôn `load → validate → summarize` | Áp lại y hệt cho `questions.py`, đổi luật bên trong |

### PHẢI sửa — 5 chỗ đau thật

**1. `plan_uses_index()` của tuần 5 KHÔNG dùng lại được cho GIN — và nó im lặng trả sai.**

`week5/indexTuning/hnsw_index.py:plan_uses_index` tìm chuỗi `"Index Scan using <tên>"`.
Index GIN không bao giờ xuất hiện dưới dạng đó — Postgres dùng GIN qua **`Bitmap Index Scan on <tên>`**
(GIN là inverted index, trả về tập row-id không có thứ tự, phải gom bitmap rồi mới đọc heap).

→ Import nguyên si về dùng sẽ **luôn trả `False`**, và mình sẽ đi tạo lại index / đổi tham số /
nghi ngờ Postgres trong khi index vẫn chạy tốt. False negative tốn thời gian hơn không kiểm tra gì.
Đã viết `uses_gin_index()` riêng ở `keyword_search.py` (TODO 7) — **đừng "tối ưu" bằng cách gộp lại.**

**2. Unique index `chunks_source_idx` chặn việc nhân bản dữ liệu — đây là lý do món nợ 50k dòng chưa xong.**

`db_docs.py` có `CREATE UNIQUE INDEX chunks_source_idx ON chunks (source, chunk_index)`.
Nên `INSERT INTO chunks ... SELECT ... FROM chunks` kiểu ngây thơ sẽ **nổ ngay** vì trùng khoá.
Cộng thêm: sau hôm nay bảng có cột `tsv` GENERATED, nên `SELECT *` cũng hỏng (không được chèn
giá trị vào cột generated, và `id` đã có sẵn).
→ Cách đúng: đổi `source` thành `source || '#synthetic-N'`. Xem `seed_bulk_rows.py` (TODO 14).
Điểm sáng: ràng buộc đặt ở tầng DB (bài học tuần 5) hôm nay cứu chính mình.

**3. `print_verdict` ở `week5/indexTuning/compare_topk.py` — nợ 15 phút, sửa nốt hôm nay.**

Hiện tại nhánh `overlap == k and same_order` in kết luận rồi mới cảnh báo "chỉ đúng nếu Index Scan".
Cảnh báo bằng chữ thì lần sau đọc vội vẫn đọc sai. → Truyền `uses_index: bool` vào `print_verdict`
và **chỉ in câu kết luận về recall khi `uses_index is True`**; ngược lại in thẳng
"cả hai bên đều exact — không kết luận được gì về ANN". Commit riêng, không trộn vào code tuần 7.

**4. `retrieve()` KHÔNG dùng để đo được — phải gọi thẳng `search_chunks`.**

`retriever.retrieve()` bọc `apply_similarity_threshold`: cắt chunk dưới 0.35 và trả `[]` khi top-1 yếu.
Đúng cho APP, sai cho ĐO: ca khó sẽ hiện ra "vector trượt" trong khi thật ra nó tìm được ở hạng 6
với điểm 0.31 → kết luận sai, hướng chữa sai theo.
→ Nguyên tắc: **đo ở tầng thấp nhất có ý nghĩa, đừng đo qua lớp chính sách.**

**5. Repo đang có 3 bộ câu hỏi với 3 schema khác nhau — đừng để thành 4.**

| File | Trường | Trạng thái |
|---|---|---|
| `week4/ragLite/questions.json` | `q`, `expect`, `note` | Đã dùng, 5 câu |
| `week5/evalRag/eval_cases.json` | `question`, `expect_keywords`, ... | **Vẫn là template rỗng** — 11 TODO của T5 tuần 5 chưa xong |
| `week7/keywordSearch/questions.json` | + `query_kind`, `expect_winner` | Tạo hôm nay |

Bộ hôm nay **tách riêng là đúng** (đo TRUY XUẤT, không đo CÂU TRẢ LỜI — không gọi LLM, chạy 100 lần
ra kết quả y hệt). Nhưng phải biết rõ mình đang có 3 bộ, và ghi vào "Known limitations" của README.
Việc dọn `questions.json` tuần 4 về chung schema: **commit riêng, tuần 8.**

Kèm theo, một cái bẫy đã cài sẵn: `validate_cases()` tuần 5 kiểm tỉ lệ bằng **hằng cứng** 7/3
(`REQUIRED_ANSWERABLE`/`REQUIRED_NO_ANSWER`). Bộ hôm nay 6/2 → chạy qua validator tuần 5 sẽ nhận
2 lỗi hoàn toàn sai. `validate_questions()` hôm nay nhận hai ngưỡng qua **tham số** — đó là cách sửa.

### Không có gì phải cài thêm

Postgres full-text nằm sẵn trong core, **không cần extension**. `requirements.txt` đủ dùng.
(Đây là điểm đáng nói khi phỏng vấn: thêm một nhánh retrieval mà không thêm một dependency nào.)

---

## Cấu trúc

```
week7/keywordSearch/
├── guide.md                  ← file này
├── fulltext_schema.py        3 TODO  (cột tsv + GIN + IDF)
├── keyword_search.py         4 TODO  (tokenize, tsquery, search, kiểm plan)
├── questions.json            ← ⭐ dữ liệu, điền tay, 8 câu
├── questions.py              2 TODO  (validate, summarize)
├── compare_branches.py       4 TODO  (⭐ sản phẩm chính của ngày)
└── seed_bulk_rows.py         1 TODO  (nợ tuần 5, làm ở block 15:45)
                             ─────────
                             14 TODO
```

Thứ tự làm đã xếp sao cho **mọi file test được bằng `--dry` (không DB, không model) trước khi
chạm DB**. Vòng lặp sửa–chạy vài giây thay vì vài chục giây.

---

## Block 1 — AI CORE (09:00 – 11:00)

### 09:00–09:30 · Thấy vector search DỞ bằng mắt (KHÔNG code)

Chạy 3 truy vấn loại "tên riêng / mã số / số hiệu / viết tắt" trên kho hiện có:

```bash
cd week5/askCli
python retriever.py "<mã lỗi hoặc số hiệu có thật trong tài liệu>"
python retriever.py "<tên riêng chỉ xuất hiện 1 chỗ>"
python retriever.py "<từ viết tắt>"
```

Chép nguyên kết quả tệ vào note. **Đây là lý do tồn tại của cả tuần 7** — phải thấy nó dở
trước khi tin lý thuyết. 3 câu này chính là nguyên liệu cho K1/K2 ở `questions.json`.

### 09:30–10:15 · Postgres full-text — `fulltext_schema.py`

- [X] **TODO 1** `migrate_tsv_column` — cột `tsv` GENERATED + index GIN
- [X] **TODO 3** `idf` — công thức BM25, hàm thuần, test bằng `--dry` trước

Làm TODO 3 **trước** TODO 1 nếu Docker chưa Up — nó không cần DB.

### 10:15–10:45 · IDF trên kho THẬT — `fulltext_schema.py`

- [X] **TODO 2** `top_tokens` — `ts_stat` → (word, ndoc, nentry)

```bash
python fulltext_schema.py            # bảng 20 token phổ biến nhất + df/N + idf
```

**Câu hỏi phải trả lời được sau block này** (viết vào `python-knowledge/Tuan-07/T2/README.md`):

1. Vì sao dùng config `'simple'` mà không phải `'english'`? Cái giá phải trả với tiếng Việt là gì?
2. `ndoc` khác `nentry` thế nào? Cái nào là thứ IDF cần, và tính nhầm thì sai ra sao?
3. Vì sao IDF dùng `log`, và hai số `+0.5` / `+1` trong công thức BM25 chặn lỗi gì?
4. Vì sao biểu thức của cột GENERATED phải IMMUTABLE, và `to_tsvector(content)` một tham số hỏng ở đâu?
5. Việc lọc header/footer ở `week5/ingestDocs/normalize.py::find_repeated_lines` liên quan gì đến IDF?
   *(→ đây là câu chuyện "tôi tự nghĩ ra IDF trước khi biết tên nó" — rất đáng kể lúc phỏng vấn)*
6. Postgres `ts_rank` có tính IDF không? Nếu không thì nó thiếu gì so với BM25 thật?

### 10:45–11:00 · Tiếng Anh

- [X] 5 từ lấy trong docs Postgres full-text → `tu-vung-tu-code.md`
      (gợi ý: *inverted index, lexeme, stemming, stopword, relevance ranking, tokenization*)
- [ ] 🔊 **NÓI TO 60 GIÂY**: "chunking là gì và vì sao cần overlap" — **KHÔNG nhìn note**

> MỚI tuần này: 60 giây nói to mỗi ngày. Tri tự nhận "hiểu mà chưa giải thích rõ ràng được" —
> đây là cách vá rẻ nhất. Không cần hay, cần nói hết 60 giây không dừng.

---

## Block 2 — PROJECT (11:00 – 14:00, nghỉ trưa 12:00–13:00)

### 11:00–12:00 · `keyword_search.py`

- [X] **TODO 4** `tokenize_query` — chỉ phục vụ chẩn đoán, không dựng truy vấn
- [X] **TODO 5** `build_tsquery_expression` — ⭐ **chỗ đau thật của hôm nay**
- [X] **TODO 6** `search_by_keyword`
- [X] **TODO 7** `uses_gin_index` — nhớ `Bitmap Index Scan`, không phải `Index Scan`

```bash
python keyword_search.py --dry                       # làm cái này ĐẠT trước đã
python keyword_search.py "<câu hỏi có mã số>"
python keyword_search.py "<câu hỏi có mã số>" --mode and   # so với --mode or
```

> **Nếu ra 0 kết quả, làm ĐÚNG 3 bước này trước khi sửa SQL:**
> (a) đang ở `--mode or` chưa? `plainto_tsquery` AND cả câu hỏi 10 âm tiết → gần như chắc chắn 0 dòng
> (b) `python fulltext_schema.py --terms "<câu hỏi>"` — token nào có `df = 0`?
> (c) `SELECT tsv FROM chunks LIMIT 1;` — cột đã sinh chưa?
>
> Code mò ở chỗ này là mất 40 phút không lối thoát.

### 13:00–13:40 · ⭐ Bộ 8 câu hỏi đo — `questions.json` + `questions.py`

> **Đây là DỤNG CỤ ĐO dùng suốt cả tuần.** Không có nó thì T3/T4/T5 không chứng minh được gì.
> Điền `questions.json` bằng tay **trước**, code sau.

- [X] Điền 8 câu vào `questions.json`: 6 answerable (≥2 `keyword`, ≥2 `semantic`, ≥1 `mixed`) + 2 `no_answer`
- [X] Điền `expect_source` cho **mọi** ca answerable — không có nó thì không chấm được nhánh nào đúng
- [X] Điền `expect_winner` = dự đoán của mình, **ghi TRƯỚC khi chạy**
- [X] **TODO 8** `validate_questions`
- [X] **TODO 9** `summarize_questions`

```bash
python questions.py             # self-test bộ giả
python questions.py questions.json
```

> ⚠️ Plan ghi "nếu đã tự làm xong bộ 10 câu ngoài giờ thì dùng luôn, bỏ task này".
> Đã kiểm tra: `week5/evalRag/eval_cases.json` **vẫn là template rỗng** (mọi `question` là `""`),
> nên task này **không bỏ được**. Hai bộ đo hai thứ khác nhau, vẫn nên có cả hai.

### 13:40–14:00 · ⭐ `compare_branches.py`

- [X] **TODO 10** `rank_of_source` — trả `None` khi không thấy, **tuyệt đối không trả 0**
- [X] **TODO 11** `decide_winner` — **hạng NHỎ hơn là TỐT hơn**, viết ngược là bug im lặng hoàn hảo
- [X] **TODO 12** `compare_one_question`
- [X] **TODO 13** `render_comparison_table`

```bash
python compare_branches.py --dry
python compare_branches.py --out comparison.md
```

- [X] Commit & push `week7/`

### 14:00 · Nghiệm thu — 5 phép thử, làm đủ cả 5

```bash
# 1. Mọi hàm thuần chạy đúng, không cần hạ tầng
python fulltext_schema.py --dry && python keyword_search.py --dry \
  && python questions.py && python compare_branches.py --dry
#    → kỳ vọng: 4 dòng "pass", không traceback

# 2. Migration idempotent — chạy HAI lần, lần 2 phải im lặng thành công
python fulltext_schema.py && python fulltext_schema.py
#    → kỳ vọng: index list có 1 dòng chứa `USING gin`

# 3. OR phải trả về >= AND
python keyword_search.py "<câu K1>" --mode or
python keyword_search.py "<câu K1>" --mode and
#    → kỳ vọng: số kết quả OR >= AND. Ngược lại là logic đổi toán tử sai

# 4. Câu hỏi có mã số → chunk chứa mã đó đứng HẠNG 1 ở nhánh keyword
python keyword_search.py "<câu hỏi chứa mã số có thật>"

# 5. Bộ câu hỏi hợp lệ và bảng so sánh ra được
python questions.py questions.json && python compare_branches.py --out comparison.md
```

**Tiêu chí "xong" của hôm nay:**

1. `python -m py_compile` sạch cho cả 5 file · 14 TODO không còn `NotImplementedError`
2. Bảng `comparison.md` tồn tại, không vỡ cột, không có chữ `None`
3. ⭐ **Có ít nhất 1 ca keyword thắng VÀ ít nhất 1 ca vector thắng.**
   Đây là điều kiện nghiệm thu thật — nó là lý do tồn tại của cả tuần 7.
   Không đạt thì sửa **bộ câu hỏi** trước, đừng sửa code.
4. Note ghi được: số ca dự đoán đúng, và với mỗi ca đoán sai một câu **vì sao**
5. Ghi lại điểm `ts_rank` cao nhất — nó nhỏ hơn nhiều so với similarity `0.6x` của vector.
   Đó chính là bằng chứng cho "không được cộng hai điểm này", thứ cần cho RRF ngày mai.

---

## Block 3 — TIẾNG ANH (14:00 – 15:00)

- [ ] ⬅️ **(NỢ 4 TUẦN — ƯU TIÊN SỐ 1, LÀM NGAY ĐẦU BLOCK)** Ghi âm 2–3 phút mô tả mini-project bằng tiếng Anh
      → Chất liệu **đã viết sẵn**: 5 câu ở `Tuan-05/T3/tu-vung-tu-code.md` mục 4.
      Chỉ cần đọc to rồi bấm ghi. Không cần hoàn hảo — **ghi âm xong là ĐẠT**.
- [ ] Nghe lại bản ghi **1 lần**, ghi ra 3 chỗ muốn nói trơn hơn. **KHÔNG ghi âm lại hôm nay.**

---

## Block 4 — VIỆC LÀM & NỀN TẢNG (15:00 – 15:45)

- [ ] ⬅️ **(QUÁ HẠN)** BHTN: **GỌI** công ty cũ chốt giấy tờ — làm ngay đầu block, chỉ 1 cuộc gọi.
      Việc này phụ thuộc người khác nên càng trôi càng lâu. Mốc tự đặt "cuối tháng 7" đã qua.
- [ ] LinkedIn: 1 post ngắn hoặc tương tác + kết nối 1 HR.
      Chất liệu tốt sẵn có: câu chuyện *"tôi suýt ghi một kết luận sai vào CV"* (đo đạc tuần 5) —
      khác hẳn kiểu khoe project thông thường.

---

## Block 5 — ÔN TẬP & NHẬT KÝ (15:45 – 17:00)

- [ ] ⬅️ **(Nợ tuần 5)** **TODO 14** `build_duplicate_sql` → nhân bản lên ~50k dòng
      ```bash
      python seed_bulk_rows.py --dry
      python seed_bulk_rows.py --target 50000
      python ../../week5/indexTuning/hnsw_index.py     # LÚC NÀY mới có số thật
      python keyword_search.py "<câu bất kỳ>" --plan   # tìm "Bitmap Index Scan"
      python seed_bulk_rows.py --clean                 # ⚠️ dọn TRƯỚC khi đo chất lượng
      ```
      Làm 1 lần dùng cho cả 2 việc: HNSW (nợ tuần 5) và GIN (hôm nay).
- [ ] ⬅️ **(Nợ tuần 5, 15ph)** Sửa `print_verdict` trong `compare_topk.py`: chỉ kết luận về recall
      khi `uses_index == True`. Xem mục "PHẢI sửa" #3. **Commit riêng.**
- [ ] Note hôm nay vào `python-knowledge/Tuan-07/T2/` + tick bảng lịch tuần
      → full-text, IDF, vì sao vector dở với tên riêng, số đo index ở 50k dòng

---

## Nhìn trước: hôm nay ăn khớp với phần còn lại của tuần thế nào

| Ngày | Việc | Phụ thuộc gì từ hôm nay |
|---|---|---|
| T3 (11/08) | Hybrid retrieval + RRF | `search_by_keyword` + `search_chunks` trả về list **cùng hình dạng**; RRF trộn theo **thứ hạng** vì hai thang điểm không so được — kết luận này phải đến từ số đo hôm nay |
| T4 (12/08) | Reranking cross-encoder | Nhóm ca `both_miss` và ca "tìm được nhưng hạng thấp" trong `comparison.md` chính là danh sách việc của rerank |
| T5 (13/08) | Mini-project 6 — so 3 chế độ | Chạy đúng `questions.json` hôm nay qua vector-only / keyword-only / hybrid. Đổi bộ câu hỏi giữa chừng là mất khả năng so sánh |
| T6 (14/08) | Ôn tập interview | Câu chuyện IDF + bảng so sánh 2 nhánh là 2 câu trả lời mạnh nhất tuần này |

---

## Việc ngoài code trong plan hôm nay — đừng để trôi

| Việc | Ghi chú từ xlsx |
|---|---|
| ⬅️ Ghi âm tiếng Anh 2–3 phút | **NỢ 4 TUẦN** — chất liệu đã viết sẵn, chỉ cần bấm ghi |
| 🔊 Nói to 60 giây | MỚI tuần này, làm mỗi ngày |
| ⬅️ BHTN gọi công ty cũ | **QUÁ HẠN** — phụ thuộc người khác, càng trôi càng lâu |
| LinkedIn post + kết nối 1 HR | Dùng câu chuyện đo đạc tuần 5 |
| ⬅️ Nhân bản 50k dòng | Nợ tuần 5, dùng cho cả HNSW lẫn GIN |
| ⬅️ Sửa `print_verdict` | Nợ tuần 5, 15 phút, commit riêng |
| Note + tick lịch tuần | |
