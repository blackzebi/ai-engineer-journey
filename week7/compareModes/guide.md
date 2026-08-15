# Guide — T5 tuần 7 (13/08): ✅ Mini-project #6 — so 3 chế độ retrieval

> **Mục tiêu ngày:** chốt milestone tuần — trả lời bằng **SỐ** câu hỏi "chế độ nào đáng dùng
> cho Dự án 1", không trả lời bằng cảm giác.
>
> ```
>                          8 câu hỏi (questions.json — dùng lại của T2)
>     +------------------------+------------------------+
>     |                        |                        |
> ① vector-only          ② hybrid (RRF)        ③ hybrid + rerank
> search_chunks           hybrid_search         search_with_rerank
>     |                        |                        |
>     +----------> rank_of_source + đồng hồ <-----------+
>                              |
>                metrics.py -> MRR · hit@3 · p50 ms
>                              |
>                 comparison.md -> README week7/ -> CV
> ```

---

## 0. Đọc lại tuần trước: cái gì dùng lại, cái gì phải sửa

### ⛔ Chặn trước — chế độ ③ hôm nay đang KHÔNG chạy được

`week7/reranking/` còn **12 `NotImplementedError`** (`rerank_hits`, `search_with_rerank`,
`build_pairs`, `score_pairs`, `format_rerank_table` + 4 hàm trong `latency.py`). Không có
`search_with_rerank` thì chế độ ③ không tồn tại, và mini-project hôm nay chỉ còn 2 chế độ.

Đã tính trước chuyện này, **không cần sửa gì**: `compare_modes.py` dò từng chế độ trước khi
chạy, in bảng với dòng ③ ghi `chưa có: NotImplementedError`, và vẫn ra kết luận trên 2 chế
độ. Nên thứ tự đúng hôm nay là: **làm block 09:00 trước (metrics.py — không phụ thuộc gì),
xong T4 lúc nào thì chế độ ③ tự xuất hiện trong bảng lúc đó.** Đừng để việc T4 chưa xong
chặn cả ngày.

Bốn TODO trong `latency.py` thì **không chặn hôm nay** — file này tự bấm giờ (lý do ở
`run_mode`, PHẦN 1). Chỉ `preload_models()` được dùng lại, mà hàm đó đã viết sẵn.

### Dùng lại NGUYÊN SI (đừng viết lại)

| Từ đâu | Thứ dùng lại | Vì sao vẫn dùng được |
|---|---|---|
| `keywordSearch/compare_branches.py` | `rank_of_source(hits, expect_source)` | Chỉ đọc `.source`, khớp một phần, không phân biệt hoa thường → ăn được `RetrievedChunk`, `HybridHit`, `KeywordHit` mà không cần adapter |
| `keywordSearch/questions.py` | `load_questions`, `validate_questions`, `RetrievalCase` | Bộ 8 câu là dụng cụ đo của cả tuần. Đổi nó hôm nay = huỷ baseline T2/T3 |
| `keywordSearch/questions.json` | nguyên file | — |
| `hybridRetrieval/hybrid_search.py` | `hybrid_search(conn, q, top_k, candidate_pool, k_constant)` | Chế độ ② y nguyên |
| `hybridRetrieval/rrf.py` | `DEFAULT_CANDIDATE_POOL=20`, `RRF_K_CONSTANT=60` | Giữ nguyên hằng để bảng so được với T3 |
| `week5/askCli/retriever.py` | `search_chunks` (**không phải `retrieve`**) | `retrieve` có ngưỡng similarity bên trong → nó lọc bớt kết quả trước khi mình kịp đo. Nguyên tắc đã chốt T3/T4: ngưỡng là việc của tầng app |
| `week4/ragPipeline/search_pg.py` | `embed_query` | — |
| `reranking/latency.py` | `preload_models()` | Đã viết sẵn, đã xử lý đúng chuyện "warm-up giả" (gọi `rerank_hits([])` không chạm model) |

### PHẢI sửa / phải biết — 5 chỗ đau thật

1. **`RerankedHit` không có `.source`.** Nó bọc hit gốc trong `.hit` (xem `rerank.py` dòng
   ~146). Gọi thẳng `rank_of_source(reranked, ...)` sẽ ném `AttributeError`.
   → Cách xử lý đã ghi sẵn trong `run_mode` Bước 3: bóc `[r.hit for r in reranked]` ngay tại
   chỗ dispatch. **Đừng sửa `rank_of_source`** — T2, T3 và hôm nay dùng chung nó; sửa nó là
   sửa ngược cả ba bảng đã đo xong.

2. **`search_with_rerank` trả TUPLE `(reranked, candidates)`**, không phải list. Quên unpack
   thì nổ ở `hit.source` giữa lần chạy 3 phút.

3. **`rerank.DEFAULT_TOP_K = 3` vs `rrf.DEFAULT_TOP_K = 5`** — hai hằng khác giá trị, TRÙNG
   TÊN. Import cả hai vào một file là cái sau đè cái trước, im lặng, và số trong bảng sẽ
   không phải số mình nghĩ. Hôm nay né bằng cách **không import cái nào**, dùng
   `DEFAULT_COMPARE_K = 10` của riêng file này. Nếu về sau cần cả hai: `import rrf` rồi
   `rrf.DEFAULT_TOP_K`.

4. **`week7/reranking/comparison.md` là bản sao BYTE-FOR-BYTE của
   `week7/keywordSearch/comparison.md`** (cùng 2055 byte, `diff` trùng khớp hoàn toàn). Nó
   là kết quả đo của **T2**, bị copy nhầm sang folder T4 — dễ bị đọc nhầm thành "kết quả
   rerank". → Xoá hoặc đổi tên thành `comparison.T2-copy.md` cho tới khi T4 chạy thật.
   Commit riêng, 1 dòng.

5. **`week7/` chưa có `README.md` cấp tuần** (week5 thì có). Hôm nay là ngày viết nó — xem
   block 13:00.

### Việc nhỏ ngoài lề (commit RIÊNG, đừng trộn vào commit hôm nay)

- `.claude/` đang untracked → thêm vào `.gitignore`.
- `.vscode/settings.json` vẫn được git theo dõi dù `.vscode/` đã có trong `.gitignore` —
  file bị track từ trước khi thêm luật ignore. `git rm --cached .vscode/settings.json` nếu
  muốn dọn.
- Identifier: đã soát `week7/`, **100% tiếng Anh**, không có gì phải rename. Tốt.

### Chưa có, phải cài

Không cần cài gì mới. Mọi thứ hôm nay đã có sẵn từ T2–T4.

---

## Cấu trúc

```
week7/compareModes/
├── rubric.md          ← điền TRƯỚC khi chạy (không code)
├── metrics.py         ← 5 TODO · hàm thuần, test không cần DB
├── compare_modes.py   ← 6 TODO · chạm DB + model
└── comparison.md      ← sinh ra khi chạy
```

Tổng **11 TODO**. Ít hơn mọi ngày trong tuần, có chủ ý: hôm nay phần nặng là **viết README
và kết luận**, không phải gõ code.

---

## Block 1 — AI CORE (09:00 – 11:00) · chấm thế nào cho đúng

### 09:00–09:45 · `rubric.md` — chốt tiêu chí TRƯỚC khi nhìn kết quả

- [X] Điền hết mục 1→4 của `rubric.md`. **Commit ngay sau khi điền** — dấu thời gian git là
      bằng chứng mình chốt trước, và đó chính là điều làm bảng đo đáng tin.

Vì sao đây là task đầu tiên của ngày chứ không phải code: viết tiêu chí sau khi thấy số thì
mình sẽ vô thức chọn ngưỡng nào làm cho chế độ mình thích thắng. Không cưỡng lại được bằng ý
chí, chỉ cưỡng lại được bằng thứ tự làm việc.

### 09:45–10:45 · `metrics.py` — 5 TODO

- [X] **TODO 1** `reciprocal_rank` — hạng → 1/hạng, trượt → 0.0
- [X] **TODO 2** `mean_reciprocal_rank` — trung bình RR, chặn danh sách rỗng
- [X] **TODO 3** `hit_at_k` — tỉ lệ câu có đoạn đúng trong top-k (nhớ `<=`, không phải `<`)
- [X] **TODO 4** `median_ms` — trung vị, không phải trung bình
- [X] **TODO 5** `distinct_source_count` — số nguồn khác nhau trong top-3 (tín hiệu cho N1/N2)

```bash
cd week7/compareModes
python metrics.py       # không cần Docker, không cần model — sửa/chạy 0.2 giây
```

Làm file này trước `compare_modes.py` vì nó **không phụ thuộc gì cả**: không DB, không model,
không T4. Vòng lặp sửa–chạy ngắn nhất trong ngày.

### 10:45–11:00 · 🗣 Tiếng Anh

- [X] 5 từ mới (lấy trong docstring `metrics.py` / docs về MRR)
- [ ] 🔊 **Nói to 60 giây, KHÔNG nhìn note:** *"hybrid retrieval là gì và vì sao cần nó"*
      — đúng câu sẽ dùng trong phỏng vấn tuần tới.

### Câu hỏi phải trả lời được sau block này
(viết vào `python-knowledge/Tuan-07_.../T5/README.md`)

1. Vì sao MRR dùng `1/rank` chứ không phải `1 - rank/k`? Trả lời bằng ví dụ hạng 1→2 và 9→10.
2. MRR = 0.25 nghĩa là gì, nói bằng một câu người thường hiểu được?
3. Ca nào làm MRR tăng mà hit@3 đứng im? Vì sao ca đó lại hay xảy ra với rerank?
4. Vì sao 2 câu no_answer phải bị loại khỏi MRR? Nếu để vào thì con số sai theo hướng nào?
5. Vì sao đo latency bằng trung vị chứ không trung bình, dù chỉ có 8 mẫu?

---

## Block 2 — PROJECT (11:00 – 14:00, nghỉ trưa 12:00–13:00)

### 11:00–12:00 · `compare_modes.py` — 6 TODO

- [X] **TODO 6** `probe_modes` — dò 3 chế độ trước khi tốn 3 phút chạy cả bộ
- [X] **TODO 7** `run_mode` — dispatch + bấm giờ. ⚠️ **Đọc kỹ "BẪY LỚN NHẤT HÔM NAY"** ở
      PHẦN 1 của hàm này trước khi gõ dòng nào
- [X] **TODO 8** `evaluate_case` — 1 câu hỏi × mọi chế độ khả dụng

```bash
python compare_modes.py --dry     # 3 hàm render chạy bằng dữ liệu giả, chưa cần DB
```

### 13:00–13:40 · Ba hàm in kết quả

- [X] **TODO 9** `render_mode_table` — sinh cột theo `modes`, đừng hard-code 3 cột
- [X] **TODO 10** `render_metric_summary` — chế độ hỏng **vẫn có dòng**, ghi rõ lý do
- [X] **TODO 11** `print_verdict` — 3 câu: tốt nhất là gì · đắt thêm bao nhiêu · **tệ đi ở đâu**

### 13:40–14:00 · README `week7/README.md`

Chép từ `rubric.md` mục 5 sang, thêm:

- Bài toán + sơ đồ **2 tầng** (retrieve rẻ, rộng → rerank đắt, hẹp)
- Bảng kết quả (dán thẳng `comparison.md`)
- **Vì sao chọn `N=20` / `k=3` / `RRF k=60`** — không phải "vì tutorial ghi thế":
  - `N=20`: rerank là O(N) lần gọi model. 20 × ~15ms ≈ 300ms, còn chịu được; 100 thì không.
  - `RRF k=60`: hằng gốc của bài báo RRF, làm phẳng chênh lệch giữa các hạng đầu — đã đo
    ảnh hưởng bằng `describe_k_effect` ở T3.
  - `k=3`: tầng sinh câu trả lời tuần 5 nhận 3 đoạn. Nhưng **đo thì đo ở k=10** — xem BẪY
    LỚN NHẤT.
- **Giới hạn đã biết** (mục này là thứ phân biệt project học việc với project thật):
  full-text config `simple` không stem tiếng Việt · cross-encoder `ms-marco-MiniLM` huấn
  luyện trên tiếng Anh mà 4/8 câu là tiếng Việt · n=8 câu, p50 với n=8 · kho chỉ 6 file.

### 14:00 · Nghiệm thu — 5 phép thử, làm đủ cả 5

```bash
cd week7/compareModes

# 1. Hàm thuần đúng trước đã
python metrics.py
#    -> 15/15 phép thử đạt. Đặc biệt dòng cuối "MRR tăng nhưng hit@3 đứng im" phải ✅

# 2. Ba hàm render đúng, chưa cần DB
python compare_modes.py --dry
#    -> bảng 1: mọi dòng cùng số cột, ký tự | trong câu hỏi đã thành /
#    -> bảng 2: ĐỦ 3 dòng, dòng ③ ghi "chưa có: NotImplementedError..."
#    -> verdict: dòng 📉 liệt kê đúng S1 (1 -> 4), KHÔNG phải K1

# 3. Chạy thật
docker compose up -d          # nếu Postgres chưa chạy
python compare_modes.py
#    -> dò chế độ in đúng 1 dòng ⚠️ (chế độ ③, nếu T4 chưa xong)
#    -> 8 dòng tiến trình, rồi 2 bảng + verdict, rồi 💾 comparison.md

# 4. Kiểm BẪY LỚN NHẤT bằng mắt: chế độ ③ (khi đã chạy được) phải có hạng > 3 ở ít nhất
#    một câu. Nếu mọi hạng của ③ đều nằm trong {1,2,3,—} thì đang đo ở top_k=3 -> sai.

# 5. Điều kiện nghiệm thu của tuần: trong comparison.md phải có ÍT NHẤT MỘT ca chế độ phức
#    tạp hơn lại TỆ HƠN. Không có -> đọc lại cảnh báo cuối print_verdict, đừng ăn mừng.
```

**Tiêu chí "xong" của hôm nay:**

- [X] `metrics.py` 15/15 · `compare_modes.py --dry` không traceback
- [X] `comparison.md` sinh ra từ lần chạy thật, có số của ít nhất 2 chế độ
- [X] `rubric.md` mục 5 đã điền, và **có ghi chỗ mình đoán sai**
- [ ] `week7/README.md` có mục "vì sao chọn tham số" và mục "giới hạn đã biết"
- [ ] Push GitHub + cập nhật README gốc của `ai-engineer-journey`

---

## Nhìn trước: hôm nay ăn khớp với phần còn lại thế nào

| Ngày | Việc | Phụ thuộc gì từ hôm nay |
|---|---|---|
| T6 (14/8) | Ôn tập + interview section E | Câu trả lời "chọn chế độ nào, vì sao" lấy thẳng từ `rubric.md` mục 5 |
| T7 (15/8) | Review tuần | `comparison.md` là bằng chứng của cả tuần 7 |
| Tuần 8 | Golden dataset + LLM-as-judge | `metrics.py` chính là hạt giống: MRR/hit@k giữ nguyên, thay phần "đoạn đúng" thủ công bằng judge. Bộ 8 câu sẽ nở thành 30–50 câu |
| Tuần 10 | FastAPI + Docker | Chế độ chốt hôm nay là chế độ đem đi bọc thành API. Chọn sai hôm nay = bọc nhầm thứ |

---

## Việc ngoài code trong plan hôm nay — đừng để trôi

**🗣 14:00–15:00 · Tiếng Anh**

- [ ] Podcast / docs 15–20 phút + 5 từ
- [ ] Viết **5 câu tiếng Anh** mô tả hệ retrieval 3 chế độ (dùng lại project pitch hôm T4),
      **nói to 2 lần**. Đây sẽ là câu trả lời cho *"tell me about a project you built"*.

**💼 15:00–15:45 · Việc làm & nền tảng**

- [ ] Cập nhật CV + LinkedIn: thêm **1 dòng có KẾT QUẢ ĐO ĐƯỢC** về hệ retrieval
      (số liệu tự đo là thứ hiếm trong CV người mới chuyển ngành — lấy thẳng từ bảng
      `render_metric_summary`, kèm cả cái giá latency, đừng chỉ khoe phần tốt)
- [ ] Theo dõi BHTN (kết quả cuộc gọi T2)
- [ ] Rà repo: `week5/` ✅ đã có README · `week7/` ❌ **chưa có** → viết ở block 13:40

**📓 15:45–17:00 · Ôn tập & nhật ký**

- [ ] Note luồng 3 chế độ + bảng kết quả vào `python-knowledge/Tuan-07_.../T5/`, kèm kết
      luận **chế độ nào đáng dùng cho Dự án 1 và VÌ SAO**
- [ ] Tick bảng lịch tuần + điền xlsx hôm nay
