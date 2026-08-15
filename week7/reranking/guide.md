# Guide — T4 tuần 7 (12/08): Reranking bằng cross-encoder

> **Mục tiêu ngày:** thêm tầng chấm lại cuối cùng — chính xác hơn nhiều, nhưng phải **ĐO ĐƯỢC
> cái giá phải trả bằng mili-giây**. Hôm nay không phải ngày làm cho kết quả đẹp hơn, mà là
> ngày biết chính xác mình đã trả bao nhiêu để mua được bao nhiêu.
>
> ```
>                             câu hỏi
>                                |
>        ┌───────────────────────┴───────────────────────┐
>        │  TẦNG 1 — RẺ, lấy RỘNG                        │
>        │  embed → search_chunks(20)                    │  ~vài chục ms
>        │  search_by_keyword(20)  →  fuse_rankings(RRF) │  (T2 + T3, KHÔNG sửa)
>        └───────────────────────┬───────────────────────┘
>                                | 20 ứng viên
>        ┌───────────────────────┴───────────────────────┐
>        │  TẦNG 2 — ĐẮT, chấm KỸ                        │
>        │  cross-encoder đọc 20 cặp (câu hỏi, chunk)    │  ~100–500 ms  ← ĐO SỐ NÀY
>        └───────────────────────┬───────────────────────┘
>                                | top-3
>                          câu trả lời
>
>   Đo cộng dồn:  vector  →  +keyword  →  +fusion  →  +rerank
>                 cột "thêm" của từng tầng = cái giá thật của riêng mảnh đó
> ```

---

## 0. Đọc lại tuần trước: cái gì dùng lại, cái gì phải sửa

### Dùng lại NGUYÊN SI (đừng viết lại một dòng nào)

| Từ đâu | Thứ dùng lại | Vì sao vẫn dùng được |
|---|---|---|
| `week7/hybridRetrieval/hybrid_search.py` | `hybrid_search` | Hôm nay **không sửa một dòng nào** trong file này. Nó đã nhận `top_k` và `candidate_pool` là hai tham số RIÊNG — hôm qua tưởng là thừa, hôm nay chính chỗ đó cho phép đổi vai từ "quyết định" sang "tiến cử" mà không phải refactor. |
| `week7/hybridRetrieval/rrf.py` | `fuse_rankings`, `BRANCH_*`, `RRF_K_CONSTANT` | `latency.py` cần chèn điểm dừng vào **giữa** hybrid (sau keyword, trước fusion) nên gọi trực tiếp `fuse_rankings`. |
| `week7/keywordSearch/compare_branches.py` | `rank_of_source` | Ăn bất cứ object nào có `.source`. ⚠️ `RerankedHit` **không** có `.source` (nó bọc hit gốc) → phải truyền `[r.hit for r in reranked]`. Đã viết sẵn trong `evaluate_all_questions`, đọc kỹ dòng đó. |
| `week7/keywordSearch/questions.py` + `questions.json` | `load_questions`, `validate_questions` | Vẫn **đúng bộ 8 câu** của T2/T3. Đổi bộ câu hỏi hôm nay là tự huỷ cả `comparison.md` lẫn bảng T3. |
| `week5/askCli/retriever.py` | `search_chunks` (⚠️ **không** phải `retrieve`) | Bẫy y hệt hôm qua, lý do y hệt: ngưỡng similarity là quyết định của **tầng app**, nằm sau pipeline. |
| `week4/ragPipeline/search_pg.py` | `embed_query`, và **khuôn `get_model()`** | `get_cross_encoder()` hôm nay là bản sao có chủ ý của `get_model()`: biến module + import lười. Đọc lại 8 dòng đó trước khi gõ. |
| `week5/indexTuning/hnsw_index.py` | **quy trình đo**, không phải code | `benchmark_query` đã dạy: chạy warm-up rồi mới lấy số ở lần thứ hai. Hôm nay áp dụng lại đúng nguyên tắc đó cho model thay vì cho Postgres. |

### ✅ ĐÃ CHỐT (15/08) — món nợ K1 (tsv/parser) dời hẳn sang **đầu Tuần 8**

> **KHÔNG làm trong tuần 7.** Phần dưới giữ lại làm tài liệu thi hành cho T2 tuần 8
> (17/8, block Ôn tập 15:45–17:00) — đã ghi vào `weekly-tracker.md` ở cả mục Tuần 7 lẫn Tuần 8.

**Lý do chốt như vậy:** sửa `tsv` giữa tuần là **đổi điều kiện đo giữa chừng** — đúng cái lỗi
mà quy trình đo rút ra từ tuần 5 đã cấm. Tuần 7 chốt milestone bằng bảng so 3 chế độ ở T5;
sửa vào T4/T5 thì `comparison.md` (T2), bảng 3 nhánh (T3), `rerank_shift.md` + `latency.md`
(T4) đứng trên nền cũ còn bảng T5 đứng trên nền mới — **bốn bảng, hai nền, hết so được**. Mà
so được giữa các ngày chính là toàn bộ lý do cả tuần dùng chung một bộ 8 câu.

Nguyên tắc rút ra: khi phải chọn giữa *"nền đo đúng hơn"* và *"nền đo nhất quán"*, trong một
tuần đang dở dang thì **nhất quán thắng**. Sửa nền là việc làm ở **ranh giới** giữa hai chu
kỳ đo, không phải giữa chu kỳ.

**Cái giá phải trả — ghi vào README `week7/` và giữ nguyên kể cả sau khi sửa:** nhánh keyword
tìm ra tài liệu đúng ở **4/8** câu; ca K1 kém là do **URL parser**, không phải lỗi ranking hay
chunking. Giới hạn đã ghi thành lời thì không còn là nợ giấu.

```bash
# 1. Đo TRƯỚC (để có số so sánh về cái giá dung lượng)
psql -c "SELECT pg_size_pretty(pg_total_relation_size('chunks'));"
#->  pg_size_pretty : 3472 kB (1 row)

# 2. BẮT BUỘC drop trước — ADD COLUMN IF NOT EXISTS sẽ im lặng KHÔNG đổi gì
#    (bẫy đã ghi trong guide T3; nhớ sửa luôn docstring 'Idempotent' cho đúng sự thật)
psql -c "ALTER TABLE chunks DROP COLUMN tsv;"

# 3. Thêm luồng thứ ba weight D vào biểu thức generated, rồi chạy lại
python ../keywordSearch/fulltext_schema.py

# 4. Xác nhận 27017 đã thành token độc lập
psql -c "SELECT to_tsvector('simple', regexp_replace('mongodb://localhost:27017/mydb','[^[:alnum:]]+',' ','g'));"

# 5. Đo SAU + chạy lại baseline
psql -c "SELECT pg_size_pretty(pg_total_relation_size('chunks'));"
python ../keywordSearch/compare_branches.py --out comparison.md
```

Biểu thức thêm vào (weight `D` = 0.1 theo `RANK_WEIGHTS = "{0.1, 0.2, 0.4, 1.0}"`, tức thấp
nhất — cố ý, để bản đập nát dấu câu không thổi phồng `ts_rank`):

```sql
setweight(to_tsvector('simple',
    regexp_replace(coalesce(content,''), '[^[:alnum:]]+', ' ', 'g')), 'D')
```

**Khi làm ở tuần 8:** ghi vào note dung lượng trước/sau (mốc hiện tại **3472 kB**, đo 12/08),
K1 đổi thứ hạng thế nào, và **chạy lại cả 4 bảng trong MỘT lượt** — `comparison.md` · bảng 3
nhánh T3 · `rerank_shift.md` · `latency.md`. Đó là một phép đo hoàn chỉnh, không phải một cái
fix.

⚠️ Và làm **trước** khi thử reranker đa ngữ ở tuần 8: sửa `tsv` đổi pool 20 ứng viên, đổi hai
biến cùng lúc thì không biết cái nào tạo ra khác biệt.

### PHẢI dọn — nợ đang lớn dần, không phải việc của hôm nay nhưng đừng để qua tuần

- `git status` có **13 file week1–week5 bị sửa mà chưa commit**, y nguyên như hôm qua đã nêu.
  Trong đó `week5/evalRag/eval_results.json`, `week4/semanticSearch/vectors.json`,
  `week3/notes.json` là **file kết quả sinh ra**, không phải code — chúng không nên nằm trong
  git. Đề xuất **commit riêng**: thêm chúng vào `.gitignore` + `git rm --cached`, kèm một
  dòng trong README nói kết quả sinh lại bằng lệnh nào. Để nguyên thì mỗi lần chạy lại là
  một diff rác, và ngày nào đó `git status` bẩn tới mức mình ngừng đọc nó — lúc ấy mới là
  lúc mất file thật.
- `comparison.md` **đang sửa mà chưa commit** (baseline đã chạy lại sau khi vá
  `prediction_was_right` — đúng, N1/N2 giờ là ✅). Commit nó **trước** khi bắt đầu code hôm
  nay, một mình một commit: đó là baseline của cả tuần.
- `.vscode/settings.json` vẫn đang được track dù `.gitignore` có `.vscode/`.
- `questions.json` → `corpus_note` vẫn còn `TODO: điền số dòng chunks thật`. Hôm nay bảng
  latency **bắt buộc** ghi số dòng bảng chunks — điền một lần, dùng cho cả hai chỗ.

### Tin tốt

- Soát `week4`, `week5`, `week7`: **không có identifier tiếng Việt nào**. Giữ nguyên chuẩn đó.
- `sentence-transformers==5.6.0` đã có sẵn trong `requirements.txt` → `CrossEncoder` dùng
  được ngay, **không cần cài thêm gì**. Chỉ tải trọng số model ~80MB lần đầu (cần mạng).

---

## Cấu trúc

```
week7/reranking/
├── rerank.py            ← 4 TODO   cross-encoder. Test được bằng SCORER GIẢ, không cần model
├── search_pipeline.py   ← 4 TODO   hybrid(20) → rerank → top-3, bảng "leo/tụt"
├── latency.py           ← 4 TODO   đo 4 tầng cộng dồn, warm-up, trung vị
└── guide.md
```

**12 TODO.** Thứ tự trên là thứ tự làm. `rerank.py --dry` chạy dưới 1 giây nhờ tham số
`scorer` — vòng lặp sửa–chạy ngắn nhất nằm đúng ở file chứa phần khó nhất.

---

## Block 1 — AI CORE (09:00 – 11:00)

### 09:00–09:15 · Trả lời bằng miệng trước khi mở editor

Ba câu, nói to, không nhìn note:

1. Bi-encoder mã hoá câu hỏi và tài liệu **riêng rẽ** → hệ quả gì cho việc index?
2. Cross-encoder đọc **cả cặp** → vì sao không tính trước được?
3. Kho 50k chunk, dùng cross-encoder cho cả kho thì mỗi câu hỏi tốn bao nhiêu lần chạy model?

Trả lời được cả 3 thì **kiến trúc 2 tầng không cần ai giải thích nữa** — nó là hệ quả duy
nhất có thể có. Trả lời chưa trôi thì đọc phần đầu docstring `rerank.py` rồi nói lại.

### 09:15–10:00 · TODO 1–2 · `rerank.py` — dựng cặp và chấm điểm

- [X] **TODO 1** `build_pairs` — câu hỏi + N chunk → N cặp, **giữ nguyên thứ tự tuyệt đối**
- [X] **TODO 2** `score_pairs` — MỘT lời gọi `predict` cho cả list, đổi `np.float32` → `float`

### 10:00–10:45 · TODO 3–4 · `rerank.py` — xếp lại và nhìn thấy nó xếp

- [X] **TODO 3** `rerank_hits` — chốt `rank_before` **trước** khi sort, cắt `top_k` **sau** khi sort
- [X] **TODO 4** `format_rerank_table` — bảng "cũ → mới" có ▲/▼ và dòng "bị đá khỏi top-k"

```bash
python rerank.py --dry      # phải xanh, dưới 1 giây
```

**Câu hỏi phải trả lời được sau block này** (viết vào `python-knowledge/Tuan-07/T4/README.md`):

1. Vì sao vector tài liệu của bi-encoder index được mà điểm cross-encoder thì không?
2. Điểm cross-encoder có so được với cosine similarity không, vì sao? Đây là loại điểm thứ
   mấy không cùng thang đo trong repo này?
3. Kiến trúc 2 tầng: chọn N thế nào? Điều gì xảy ra khi N = k?
4. Tham số `scorer` trong `score_pairs` giải quyết vấn đề gì? (gọi tên kỹ thuật đó ra)

### 10:45–11:00 · Tiếng Anh

- [X] 5 từ lấy trong docs `sentence-transformers` (gợi ý: *cross-encoder, to rerank, relevance
      score, inference, trade-off*)
- [X] 🔊 **NÓI TO 60 GIÂY**: *"vì sao index vector phải là approximate"* — không nhìn note

---

## Block 2 — PROJECT (11:00 – 14:00, nghỉ trưa 12:00–13:00)

### 11:00–11:20 · Quyết định món nợ K1 (mục 0 ở trên). Hạn cứng 11:40.

### 11:20–12:00 · TODO 5–8 · `search_pipeline.py`

- [X] **TODO 5** `search_with_rerank` — ⚠️ **bẫy lớn nhất của cả ngày**: `hybrid_search(...,
      top_k=candidate_pool)`, KHÔNG phải `top_k=top_k`
- [X] **TODO 6** `classify_shift` — `dropped` phải tách khỏi `demoted`; `not_in_pool` là lỗi
      tầng 1, rerank vô can
- [X] **TODO 7** `render_shift_table`
- [X] **TODO 8** `print_shift_verdict` — biết **nghi ngờ** khi bảng toàn màu hồng

```bash
python search_pipeline.py --dry
python search_pipeline.py "Hermes engine trong React Native là gì?"
# dòng "tầng 1 tiến cử N ứng viên" phải hiện N ~ 20. Hiện 3 = đang dính bẫy TODO 5.
```

### 13:00–13:45 · TODO 9–12 · `latency.py`

- [X] **TODO 9** `time_call` — warm-up, `perf_counter`, **nhân 1000**
- [X] **TODO 10** `measure_stages` — vòng ngoài là câu hỏi, vòng trong là tầng
- [X] **TODO 11** `render_latency_table` — cột "tầng này thêm" là cột đáng đọc nhất
- [X] **TODO 12** `print_latency_verdict` — cảnh báo khi rerank rẻ bất thường / chưa warm-up

### 13:45–14:00 · Nghiệm thu — 5 phép thử, làm đủ cả 5

```bash
# 1. Ba self-test, tất cả phải xanh và nhanh (không tải model)
python rerank.py --dry && python search_pipeline.py --dry && python latency.py --dry

# 2. Một câu hỏi, xem chunk có thật sự nhảy hạng không
python search_pipeline.py "Hermes engine trong React Native là gì?"
#    kỳ vọng: dòng "tầng 1 tiến cử ~20 ứng viên" + ít nhất một mũi tên ▲ hoặc ▼

# 3. Cả bộ 8 câu, ghi ra file
python search_pipeline.py --all --out rerank_shift.md
#    kỳ vọng: có ít nhất 1 ca demoted HOẶC dropped. Toàn promoted = đọc lại verdict, đừng mừng

# 4. Latency 4 tầng
python latency.py --repeat 5 --out latency.md
#    kỳ vọng: fusion thêm < ~2ms · rerank thêm hàng trăm ms · không có ⚠️ nào

# 5. Kiểm cú pháp toàn bộ
python -m py_compile rerank.py search_pipeline.py latency.py
```

**Tiêu chí "xong" của hôm nay** — đo được, không phải cảm giác:

- [X] `rerank_shift.md` tồn tại, có ít nhất 1 ca `promoted` và 1 ca `demoted`/`dropped`
- [X] `latency.md` tồn tại, có đủ 4 tầng × (trung vị + p95 + cột thêm)
- [X] Nói được thành lời: *"rerank 20 ứng viên thêm ~X ms trên CPU, đổi lại đoạn đúng leo lên
      hạng 1 ở Y/6 câu"* — có cả X lẫn Y
- [X] `git commit` week7/reranking/ (baseline `comparison.md` commit RIÊNG, trước đó)

### ⚠️ Đọc kết quả cho đúng — 3 cái bẫy diễn giải

**1. Rerank chấm dở chunk tiếng Việt là giả thuyết PHẢI KIỂM, không phải kết luận có sẵn.**
`ms-marco-MiniLM-L-6-v2` huấn luyện trên MS MARCO — **tiếng Anh**. Câu hỏi của mình toàn
tiếng Việt; kho thì 2 file DOCX tiếng Việt (Backend-NodeJS, Frontend Interview Prep) và 3
file tiếng Anh (React PDF, Full Stack PDF, nodejs TXT). Cách kiểm cụ thể: với mỗi ca
`dropped`, nhìn chunk **đã thay chỗ** đoạn đúng — nó thuộc file tiếng Anh hay tiếng Việt?
Nếu tiếng Anh thắng một cách có hệ thống thì giả thuyết đứng vững, và đó là **phát hiện của
ngày**, không phải thất bại. Ghi vào README kèm hướng đi: model rerank đa ngữ
(`bge-reranker-v2-m3`, `jina-reranker-v2-base-multilingual`) — nêu ra, **đừng đổi hôm nay**,
đổi model giữa chừng là huỷ điều kiện đo.

**2. Ca K1 vẫn không dùng để kết luận gì về rerank** (trừ khi đã sửa `tsv` sáng nay). Nguyên
nhân nằm ở parser, tầng 1, trước cả fusion.

**3. "Rerank thêm 200ms" không tự nó nói lên điều gì.** Đắt hay rẻ phụ thuộc ngân sách: RAG
hỏi-đáp mà LLM đã sinh câu trả lời trong 3–5 giây thì +200ms là ~5%. Autocomplete ngân sách
50ms thì bất khả thi. Câu kết luận phải nêu ngân sách, và phải kèm **CPU hay GPU**.

---

## Nhìn trước: hôm nay ăn khớp với phần còn lại của tuần thế nào

| Ngày | Việc | Phụ thuộc gì từ hôm nay |
|---|---|---|
| **T5 13/08** | ✅ Mini-project #6 — `compare_modes.py` so 3 chế độ | `search_with_rerank` là **chế độ thứ ba**. Bảng T5 gồm cột latency lấy thẳng từ `latency.py`. Hai file hôm nay phải import được sạch, không thì sáng T5 mất giờ đi sửa import. |
| **T5 13/08** | README week7 + push GitHub | Sơ đồ 2 tầng, phần "vì sao N=20/k=3/RRF k=60", mục **giới hạn đã biết** (config `simple` với tiếng Việt + ms-marco với tiếng Việt) — viết dần từ hôm nay, đừng để dồn |
| **T6 14/08** | Ôn interview Section E | Câu "bi-encoder vs cross-encoder" gần như chắc chắn có trong bank. Con số latency hôm nay là phần trả lời hay nhất |
| **Tuần 8** | Golden dataset + LLM-as-judge | `scorer` injection hôm nay chính là khuôn để mock LLM judge — nhớ pattern này |

---

## Việc ngoài code trong plan hôm nay — đừng để trôi

**🗣 14:00–15:00 · Tiếng Anh**

- [X] Podcast/video kỹ thuật 15–20 phút + 5 từ (không nghe được thì lấy từ docs
      `sentence-transformers` — miễn là có 5 từ ghi vào `tu-vung-tu-code.md`)
- [ ] Nhật ký tiếng Anh 3–4 câu: **cross-encoder khác bi-encoder chỗ nào**. Đây là câu hỏi
      phỏng vấn AI/LLM rất hay gặp — tập nói bằng tiếng Anh luôn

**💼 15:00–15:45 · Việc làm & nền tảng**

- [X] ⬅️ **(Nợ từ tuần 4)** Ghi kết quả mock interview vào `docs/interview-tracker.md` theo
      đúng format session log. Không ghi thì cơ chế spaced repetition của file đó không chạy
- [ ] Viết nháp **project pitch 5 câu** cho hệ retrieval: vấn đề → cách làm → **KẾT QUẢ ĐO
      ĐƯỢC**. Dùng lại cho CV, LinkedIn, và câu G8 "what have you built outside work".
      Hôm nay là ngày đầu tiên câu thứ ba có số thật để điền

**📓 15:45–17:00 · Ôn tập & nhật ký**

- [ ] Note bi-encoder vs cross-encoder + **bảng latency 4 tầng** vào
      `python-knowledge/Tuan-07/T4/` — ghi kèm số dòng bảng chunks, N/k, CPU/GPU, đã warm-up
- [ ] Tick bảng lịch tuần + điền xlsx hôm nay
