# Guide — T3 tuần 7 (11/08): Hybrid retrieval + Reciprocal Rank Fusion

> **Mục tiêu ngày:** hợp nhất hai bảng xếp hạng KHÔNG CÙNG THANG ĐO (cosine `[-1,1]` vs
> `ts_rank` không chặn trên) thành một kết quả duy nhất, rồi ĐO xem nó hơn/kém hai nhánh
> đơn ở đâu. Ngày kỹ thuật sâu nhất tuần.
>
> ```
>                         câu hỏi
>            +---------------+---------------+
>            |                               |
>   embed -> search_chunks            search_by_keyword
>   (week5/askCli, KHÔNG dùng retrieve)   (week7/keywordSearch, hôm qua)
>            |                               |
>      pool 20 ứng viên               pool 20 ứng viên
>            |                               |
>            +------> fuse_rankings <--------+          rrf.py — THUẦN, không chạm DB
>                          |                            đây là chỗ chứa toàn bộ phần đáng sai
>              Σ 1/(60 + rank) trên mỗi nhánh
>                          |
>              top-5 + rank_v + rank_k + score
>                          |
>              compare_three_ways.py -> bảng 3 cột
> ```

---

## 0. Đọc lại tuần trước: cái gì dùng lại, cái gì phải sửa

### Dùng lại NGUYÊN SI (đừng viết lại)

| Từ đâu | Thứ dùng lại | Vì sao vẫn dùng được |
|---|---|---|
| `week7/keywordSearch/keyword_search.py` | `search_by_keyword`, `DEFAULT_MODE` | `KeywordHit` đã được cố ý đặt **cùng 7 trường đầu** với `RetrievedChunk` tuần 5 — fusion chỉ cần `.id` và vị trí trong list, nên **không phải viết adapter nào cả**. Đây là khoản đầu tư của hôm qua đang trả lãi hôm nay. |
| `week5/askCli/retriever.py` | `search_chunks` (⚠️ **không** phải `retrieve`) | Trả top-k thô, đúng thứ fusion cần. |
| `week4/ragPipeline/search_pg.py` | `embed_query` | Import **lười** (trong hàm), vì nó kéo `sentence_transformers` ~5 giây. |
| `week7/keywordSearch/compare_branches.py` | `rank_of_source` | Ăn bất cứ object nào có `.source`, khớp một phần, không phân biệt hoa thường → `HybridHit` dùng được ngay. |
| `week7/keywordSearch/questions.py` + `questions.json` | `load_questions`, `validate_questions`, `RetrievalCase` | Cùng bộ 8 câu, cùng `expect_source` → bảng hôm nay **so trực tiếp được** với `comparison.md` hôm qua. Đổi bộ câu hỏi hôm nay là tự huỷ baseline của chính mình. |
| `week4/pgvector/vector_ops.py` | `get_conn` | Không đổi. |

### PHẢI sửa — 3 chỗ đau thật

**(1) 🐛 BUG IM LẶNG trong `compare_branches.py` — bảng hôm qua đang chấm SAI 2 dòng.**

`BranchComparison.prediction_was_right`:

```python
if self.predicted_winner == "either":
    return self.winner != BOTH_MISS
return self.predicted_winner == self.winner
```

`VALID_WINNER` có giá trị `"none"` (dự đoán: không nhánh nào tìm ra), nhưng `winner` thì
chỉ nhận `vector | keyword | tie | both_miss`. Nên với ca `no_answer`, `"none" == "both_miss"`
**không bao giờ đúng** → N1 và N2 trong `comparison.md` bị đánh ❌ trong khi kết quả
`both_miss` chính là kết quả **ĐÚNG** cho ca bẫy.

Hệ quả cụ thể: tỉ lệ dự đoán thật là **7/8**, đang bị ghi là **5/8**. Con số này sẽ được
chép vào note và README, rồi 3 tuần sau dùng làm mốc so sánh cho T4/T5 — sai từ gốc.

Sửa (một dòng, **commit riêng**, không trộn vào commit hybrid):

```python
if self.predicted_winner == "none":
    return self.winner == BOTH_MISS
```

Chạy lại `python compare_branches.py --out comparison.md` sau khi sửa để baseline đúng.

**(2) ⚠️ `retrieve()` vs `search_chunks()` — cái bẫy dễ dính nhất hôm nay.**

`week5/askCli/retriever.py::retrieve` gọi `apply_similarity_threshold()` và **trả về `[]`**
khi top-1 < 0.35. Nếu nhánh vector của hybrid gọi `retrieve` thay vì `search_chunks`, nó sẽ
im lặng biến mất khỏi fusion với đúng những câu hỏi khó — tức là những câu mà hybrid sinh ra
để cứu. Bảng vẫn ra, không lỗi nào, và kết luận sẽ là "hybrid vô dụng".

Ngưỡng similarity là quyết định của **tầng app** (trả lời hay từ chối), không phải của tầng
retrieval. Nó thuộc về **sau** fusion. `compare_branches.py` hôm qua đã làm đúng — giữ nguyên
nguyên tắc đó.

**(3) ✅ Ca K1 — ĐÃ CHẨN ĐOÁN XONG (11/08). Không phải bug của mình. Hoãn sửa sang T4.**

```
| K1 | keyword | Chuỗi kết nối MongoDB ở cổng 27017... | 1 | — | vector | ❌ |
```

Đo được bằng `python fulltext_schema.py --terms "<câu K1>"`:

```
token          df      idf
27017           0    6.695   <- không tồn tại trong index
mongodb         8    3.861
cổng            3    4.749
```

**Nguyên nhân:** text gốc trong PDF (trang 15, `Top 50 Full Stack`) là

```js
mongoose.connect('mongodb://localhost:27017/mydb', { useNewUrlParser: true, ...
```

Parser mặc định của Postgres **nhận ra đây là URL** và tách theo loại token
`protocol` / `url` / `host` / `url_path`, không tách theo dấu câu. Lexeme thực sự vào index:

```
'localhost:27017/mydb'      'localhost:27017'      '/mydb'
```

`27017` **không bao giờ tồn tại** như token độc lập → `df = 0`. Đây là parser làm **đúng**
việc của nó, không phải chunking sai, không phải ingest sai, không phải `search_by_keyword`
sai. `content` trong DB hoàn toàn nguyên vẹn — chỉ có cách sinh ra cột `tsv` là không hợp
với truy vấn kiểu "tìm mã/số hiệu nằm trong chuỗi kỹ thuật".

Kiểm lại 30 giây bất cứ lúc nào:

```sql
SELECT to_tsvector('simple', 'mongodb://localhost:27017/mydb');
SELECT alias, token FROM ts_debug('simple', 'mongodb://localhost:27017/mydb');
```

**Hướng sửa (để dành T4):** thêm luồng thứ ba vào cột `tsv` — bản `content` đã đập nát dấu
câu, gán weight `D` (thấp nhất) để không thổi phồng `ts_rank`:

```sql
setweight(to_tsvector('simple',
    regexp_replace(coalesce(content,''), '[^[:alnum:]]+', ' ', 'g')), 'D')
```

Được `27017`, `localhost`, `mydb`, `mongodb` thành token riêng, `content` giữ nguyên, không
cần re-ingest (`tsv` là cột generated, Postgres tự tính lại). Cái giá: `tsv` + index GIN
phình gần gấp đôi — đo trước/sau bằng `pg_total_relation_size`.

> **🪤 BẪY CHỜ SẴN khi sửa — đọc trước khi gõ vào T4:**
> `migrate_tsv_column()` dùng `ADD COLUMN IF NOT EXISTS`. Sửa biểu thức generated rồi chạy
> lại sẽ in `✅ Cột tsv + index chunks_tsv_gin_idx sẵn sàng` và **KHÔNG ĐỔI GÌ CẢ** — cột cũ
> giữ nguyên định nghĩa cũ. Docstring đang ghi "Idempotent", nhưng nó chỉ idempotent khi
> biểu thức *không đổi*. Phải `ALTER TABLE chunks DROP COLUMN tsv;` trước rồi mới chạy lại.
> Không biết chỗ này thì mất 40 phút tưởng fix không có tác dụng. **Sửa luôn docstring cho
> đúng sự thật khi làm.**

**Quyết định hôm nay (11/08): KHÔNG sửa.** Sửa `tsv` làm `comparison.md` hôm qua hết hiệu
lực → phải chạy lại baseline giữa buổi, đúng cái lỗi "đổi điều kiện đo giữa chừng" mà quy
trình tuần 5 đã rút ra. Đo hybrid theo đúng kế hoạch, T4 sửa một thể rồi đo lại cả 3 mốc.

**Hệ quả PHẢI ghi vào note + README khi đọc bảng hôm nay:**

- Nhánh keyword hiện tìm ra tài liệu đúng ở **4/8** câu. Con số hybrid hôm nay là số đo
  trên một nhánh đang bị giới hạn bởi parser — **không phải trần của phương pháp**.
- Riêng ca **K1**, kết quả kém là do **parser**, không phải do RRF. Đừng dùng ca này để
  kết luận gì về fusion.
- Ghi kèm câu này vào README tuần 7 — nó vừa là hạn chế đã biết, vừa là câu trả lời phỏng
  vấn rất tốt: *"keyword search thất bại ở đâu, và vì sao đó không phải lỗi ranking"*.

### Việc nhỏ nên dọn (commit riêng, không trộn)

- `questions.json` còn khoá `_huong_dan` — chính file đó ghi "Xoá khoá này trước khi commit".
- `corpus_note` còn `TODO: điền số dòng chunks thật`. Hôm nay note bắt buộc ghi **điều kiện
  đo**, nên đây là lúc điền: `python ../../week5/ingestDocs/ingest_docs.py` → bảng "Phân bố
  theo định dạng".
- `git status` đang có **12 file week1–week5 bị sửa mà chưa commit**. Trước khi commit hybrid,
  `git diff` xem chúng là gì — commit hybrid mà kéo theo diff lạ của week2 thì lịch sử repo
  hết đọc được.
- `.vscode/settings.json` đang được track dù `.gitignore` có `.vscode/` (đã track từ trước nên
  gitignore không có tác dụng). `git rm --cached .vscode/settings.json` nếu không cố ý.
- `.gitignore` thiếu newline cuối file.

### Tin tốt

Soát toàn bộ `week4`, `week5`, `week7`: **không có identifier tiếng Việt nào**. Tên hàm, biến,
hằng, trường dataclass đều tiếng Anh. Giữ nguyên chuẩn đó hôm nay.

### Chưa có, phải cài

Không có gì. Hôm nay không thêm dependency nào — RRF là số học thuần, đó cũng là một lý do
để thích nó hơn cross-encoder (T4 mới cần model mới).

---

## Cấu trúc

```
week7/hybridRetrieval/
├── rrf.py                  ← 5 TODO   THUẦN, không DB không model. LÀM TRƯỚC.
├── hybrid_search.py        ← 3 TODO   nối RRF vào 2 nhánh thật
├── compare_three_ways.py   ← 4 TODO   bảng 3 cột + đọc kết quả cho đúng
└── guide.md
```

**12 TODO.** Thứ tự trên là thứ tự làm: `rrf.py` chạy được trong 1 giây không cần Postgres,
nên vòng lặp sửa–chạy ngắn nhất nằm ở đúng file chứa phần khó nhất.

---

## Block 1 — AI CORE (09:00 – 11:00)

### 09:00–09:20 · Làm bằng tay TRƯỚC KHI GÕ (đừng bỏ qua)

Kế hoạch ghi rõ: *"Tự tính tay RRF cho 2 danh sách 5 phần tử trước khi code — 5 phút, hiểu
chắc hơn đọc 30 phút."*

Lấy đúng hai danh sách trong `rrf.py::self_check`:

```
vector : 17, 42,  8, 91, 63
keyword: 42,  5, 17, 63, 88
```

Trên giấy, tính `Σ 1/(60 + rank)` cho id 17 và id 42. Viết ra **trước** khi chạy code:
id nào thắng, và chênh nhau bao nhiêu. Rồi tính lại với `k = 1`. Thứ tự có đảo không?

### 09:20–09:40 · Tự chứng minh "không cộng thẳng điểm được"

Ghi vào note một ví dụ số cụ thể của **chính kho mình** (không phải ví dụ trong sách):
điểm `ts_rank` cao nhất đo được hôm qua là bao nhiêu, `similarity` cao nhất là bao nhiêu.
Rồi tính `0.6x + 0.0x` và nói xem nhánh keyword đóng góp bao nhiêu phần trăm vào quyết định.

### 09:40–11:00 · `rrf.py` — 5 TODO

- [X] **TODO 1** `reciprocal_rank_score` — công thức 1 dòng + guard `rank < 1`
- [X] **TODO 2** `fuse_rankings` — trái tim của ngày. 3 bẫy: id trùng, sort không tất định, cắt trước khi sắp
- [X] **TODO 3** `min_max_normalize` — mảnh của cách làm thay thế; ca `max == min` là ca thật
- [X] **TODO 4** `weighted_fusion` — viết để **so sánh**, không phải để dùng
- [X] **TODO 5** `describe_k_effect` — biến câu chữ về k thành bảng nhìn thấy được

```bash
python rrf.py            # phải in "pass" trong dưới 1 giây, không cần Postgres
python rrf.py --k-effect
```

**Câu hỏi phải trả lời được sau block này** (viết vào
`python-knowledge/Tuan-07/T3/README.md`):

1. Vì sao `similarity` và `ts_rank` không cộng thẳng được? Trả lời bằng **hai con số thật
   của kho mình**, không phải bằng lý thuyết.
2. Ở `k = 60`, điểm của hạng 1 và hạng 2 chênh nhau bao nhiêu **phần trăm**? Ở `k = 1` thì
   bao nhiêu? Con số đó nói gì về ý nghĩa của k?
3. Một chunk hạng 2 ở cả hai nhánh vs một chunk hạng 1 ở đúng một nhánh — RRF chọn ai, và
   ở giá trị k nào thì lựa chọn đảo chiều?
4. Weighted fusion (min-max + trọng số) hơn RRF ở điểm nào, kém ở điểm nào? Nêu **một** tình
   huống cụ thể mà mình sẽ chọn weighted.
5. Vì sao `fuse_rankings` chỉ nhận `list[int]` mà không nhận list chunk? (câu hỏi về **thiết
   kế**, không về thuật toán — và là câu hay bị hỏi trong phỏng vấn)

### 10:45–11:00 · Tiếng Anh (chen vào cuối block, theo khung giờ)

5 từ mới + **nói to 60 giây, không nhìn note**: *"ingest pipeline của tôi làm gì, theo thứ
tự nào"*. Đây là khái niệm Tri tự nhận là chưa giải thích rõ được — nói lại cho tới khi
không có chỗ ngập ngừng.

---

## Block 2 — PROJECT (11:00 – 14:00, nghỉ trưa 12:00 – 13:00)

> ~~11:00–11:15 · Kiểm ca K1~~ — **xong sáng 11/08**, nguyên nhân là URL parser của Postgres,
> hoãn sửa sang T4. Xem mục "PHẢI sửa" số 3. Block 2 bắt đầu thẳng vào `hybrid_search.py`.

### 11:00–12:00 · `hybrid_search.py` — 3 TODO

- [X] **TODO 6** `collect_candidates` — chạy 2 nhánh, trả (id-theo-hạng, bảng tra). ⚠️ `search_chunks`, **không** phải `retrieve`
- [X] **TODO 7** `hybrid_search` — lắp mảnh. Gán dataclass theo **tên**, đừng `HybridHit(*chunk)`
- [X] **TODO 8** `format_hybrid_hits` — bảng đủ **3 cột**: `rank_vector`, `rank_keyword`, `rrf_score`

```bash
python hybrid_search.py --dry                                  # dưới 1 giây
python hybrid_search.py "Hermes engine trong React Native là gì?"
```

### 13:00–13:45 · `compare_three_ways.py` — 4 TODO

- [X] **TODO 9** `classify_change` — nhớ chiều: hạng **nhỏ** hơn là tốt hơn
- [X] **TODO 10** `compare_one_question` — 3 nhánh trong **cùng một lần chạy**
- [X] **TODO 11** `render_three_way_table` — ký tự `|` trong câu hỏi phá cột
- [X] **TODO 12** `print_summary` — hàm này phải **cảnh báo khi kết quả quá đẹp**

### 13:45–14:00 · Nghiệm thu — 5 phép thử, làm đủ cả 5

```bash
# 1. Ba file self-test, không cần Postgres, mỗi lệnh dưới 1 giây
python rrf.py && python hybrid_search.py --dry && python compare_three_ways.py --dry

# 2. Ca K2 — hôm qua vector=3, keyword=1. Sau hybrid phải lên hạng 1, cột hiện "v=3 k=1"
python hybrid_search.py "Hermes engine trong React Native là gì?"

# 3. pool phải có tác dụng thật — hai lệnh này PHẢI ra kết quả khác nhau
python hybrid_search.py "..." --pool 5
python hybrid_search.py "..." --pool 40

# 4. k_constant phải có tác dụng thật — thứ tự phải đảo ở ít nhất 1 câu
python hybrid_search.py "..." --rrf-k 1
python hybrid_search.py "..." --rrf-k 200

# 5. Bảng cuối ngày
python compare_three_ways.py --k 10 --pool 20 --out three_ways.md
```

**Tiêu chí "xong" của hôm nay:**

1. Cột `vector` và `keyword` trong `three_ways.md` **trùng** với `comparison.md` hôm qua.
   Lệch = kho đã đổi → dừng, ghi vào note, đừng so tiếp.
2. ⭐ Có **ít nhất 1 ca `worse` hoặc `lost`**. Bảng toàn cải thiện là đang tự lừa mình.
3. **Không có ca `lost` nào** (hybrid không được làm mất thứ nhánh đơn đã tìm ra).
4. Với mỗi ca `improved` và mỗi ca `worse`, note có **một câu VÌ SAO**.
5. Note ghi đủ **điều kiện đo**: số dòng bảng `chunks`, `pool`, `rrf_k`, `top_k`, `mode`,
   đã warm-up model chưa. (Quy trình rút ra từ tuần 5: thiếu điều kiện đo thì 2 tuần sau
   con số vô dụng.)

---

## Nhìn trước: hôm nay ăn khớp với phần còn lại của tuần thế nào

| Ngày | Việc | Phụ thuộc gì từ hôm nay |
|---|---|---|
| T4 12/08 | Reranking bằng cross-encoder | Cross-encoder chấm lại **pool ứng viên** — chính là `collect_candidates` hôm nay. `candidate_pool` = 20 sẽ thành tham số quan trọng nhất của T4 (rerank 20 chunk tốn ~20 lần forward pass). Và bảng `three_ways.md` thành **baseline** để đo rerank có hơn RRF không. **➕ Nợ mang sang: sửa `tsv` (mục "PHẢI sửa" số 3) — làm ĐẦU buổi T4, trước khi rerank, rồi chạy lại cả `comparison.md` và `three_ways.md` một lượt.** |
| T5 13/08 | Mini-project 6 — so 3 chế độ | 3 chế độ đó nhiều khả năng là vector / hybrid / hybrid+rerank. `classify_change` và `render_three_way_table` hôm nay mở rộng thêm cột là dùng được. |
| T6 14/08 | Ôn interview Section E | "Vì sao RRF chứ không phải weighted sum" là câu E kinh điển. Câu trả lời tốt = bảng `--k-effect` + một con số thật của kho mình. |

---

## Việc ngoài code trong plan hôm nay — đừng để trôi

**14:00–15:00 · Tiếng Anh**

- Đọc to 15 phút: docs Postgres full-text hoặc một bài về RRF. Chú ý trọng âm:
  *reciprocal · fusion · ranking · to normalize · weighted*
- Nhật ký tiếng Anh 3–4 câu: **vì sao không cộng thẳng hai loại điểm được**. Giải thích
  được bằng tiếng Anh = hiểu thật.

**15:00–15:45 · Việc làm & nền tảng**

- ⬅️ 20 phút: ôn 2 câu yếu lâu nhất cho mock **Chủ nhật 11:00** — **A6** (React.memo /
  useCallback, đã lặp lỗi 2 buổi liền) và **C3** (N+1 query). Chỉ ĐỌC để nhớ.
- 15 phút: đọc lướt **E22–E27** trong `docs/interview-question-bank.md`. Section E mở khoá
  từ 25/07 mà chưa dùng lần nào; buổi CN sẽ có ~3 câu AI/LLM.

**15:45–17:00 · Ôn tập & nhật ký**

- Note RRF + bảng 3 nhánh vào `python-knowledge/Tuan-07/T3/` — **kèm điều kiện đo**.
- Tick bảng lịch tuần + điền `T3_11-08_Hybrid-Retrieval-RRF.xlsx`.
- Nhắn *"note kiến thức hôm nay"* để chạy quy trình chốt ngày.
