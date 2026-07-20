# Guide — Semantic Search mini (local, miễn phí)

> Nợ T5 tuần 3, làm bù ở tuần 4. Mục tiêu: tự tay build một **semantic search** nhỏ để
> "thấy tận mắt" tìm-theo-nghĩa khác tìm-theo-chữ. Code khung ở
> [`semantic_search.py`](semantic_search.py) — bạn điền 4 chỗ `# TODO`.
> Concept nền + cosine tự cài: xem study notes
> `D:\Study\AI-Engineer-Study\python-knowledge\Tuan-03_.../T5/`.

## Bức tranh tổng thể

```
[~20 đoạn text]  --embed-->  [~20 vector 384c]  --lưu file-->  vectors.json   (Task 1: index 1 lần)
                                                                   │
câu hỏi  --embed-->  [1 vector]  --cosine với từng đoạn-->  xếp hạng --> top 3  (Task 2: query)
```

Đây chính là **retrieval** — trái tim của RAG (chủ đề tuần 4–5). Làm nhỏ ở đây để sau này chỉ việc
thay "list Python" bằng "vector DB (pgvector)".

## Vì sao dùng sentence-transformers (không phải API)?

| | sentence-transformers (all-MiniLM-L6-v2) | Voyage/OpenAI API |
|---|---|---|
| Chi phí | **Miễn phí** | Trả tiền theo token |
| Chạy ở đâu | **Local**, không cần key/mạng (sau khi tải model) | Gọi qua mạng |
| Số chiều vector | 384 | 1024–1536 |
| Khác biệt code | `model.encode(texts)` — **không** có `input_type` | phải phân biệt document vs query |

Đúng ghi chú tracker: *local, miễn phí, chưa cần vector DB, list Python là đủ*.

> ⚠️ **Caveat tiếng Việt:** `all-MiniLM-L6-v2` train chủ yếu trên tiếng Anh nên semantic tiếng Việt
> chỉ ở mức "tạm ổn". Nếu thấy kết quả lệch, đổi 1 dòng `MODEL_NAME` sang
> `paraphrase-multilingual-MiniLM-L12-v2` (đa ngôn ngữ, cũng miễn phí, 384 chiều) — code còn lại giữ nguyên.
> Đây cũng là 1 talking point phỏng vấn tốt: "chọn embedding model phải khớp ngôn ngữ của dữ liệu".

## Setup (1 lần)

```bash
pip install sentence-transformers
# thêm dòng sentence-transformers==<ver> vào requirements.txt (lấy từ: pip freeze | findstr sentence-transformers)
```

Không cần API key. Lần đầu chạy `build`, thư viện tự tải model (~90MB) về cache máy.

---

## Task 1 — Embed ~20 đoạn text, lưu vector ra file (`build_index`, `embed_texts`)

**Mục tiêu:** có sẵn "kho" đã embed để query. Chưa cần vector DB — list Python là đủ.

1. Sửa nội dung kho trong [`corpus.txt`](corpus.txt) — mỗi dòng 1 đoạn (1–3 câu), lấy từ notes tuần 1–3.
   Càng đa dạng chủ đề càng dễ thấy Task 3 hoạt động. (File đã có ~20 đoạn mẫu để bạn soát/thay.)
2. Điền **TODO 1** (`embed_texts`): `_model.encode(texts)` → `.tolist()`. Embed cả list 1 lần (batch).
3. Điền **TODO 3** (`build_index`): ghép `{"text", "vector"}` rồi `save_vectors`.

**Bẫy:**
- Đừng embed lại kho mỗi lần query → chạy `build` 1 lần, sau đó `search` đọc lại `vectors.json`.
- `ensure_ascii=False` khi dump (đã viết sẵn) để giữ tiếng Việt đọc được.
- Mọi đoạn phải cùng 1 model — trộn model khác nhau thì vector **không so được**.

**Xong khi:** `python semantic_search.py build` in ra "đã embed 20 đoạn, mỗi vector 384 chiều" và sinh `vectors.json`.

---

## Task 2 — Viết `search(query)`: top-3 gần nghĩa nhất + điểm cosine (`cosine_similarity`, `search`)

**Mục tiêu:** thấy tận mắt "tìm theo nghĩa" khác "tìm theo chữ".

1. Điền **TODO 2** (`cosine_similarity`): `dot / (norm_a * norm_b)`, guard chia 0 → trả 0.0. Chỉ dùng `math`.
2. Điền **TODO 4** (`search`): embed câu hỏi → cosine với từng đoạn → sort giảm dần → `top_k`.

**Xong khi:** `python semantic_search.py search "câu hỏi tiếng Việt"` in ra 3 đoạn gần nghĩa nhất kèm điểm.

---

## Task 3 — Test query KHÔNG trùng từ khoá vẫn tìm đúng (ghi 2–3 ví dụ vào README)

**Đây là demo mạnh khi phỏng vấn** — chứng minh semantic ≠ keyword.

Chọn câu hỏi **không dùng chung từ nào** với đoạn đích nhưng cùng nghĩa. Keyword search sẽ trượt,
semantic vẫn ra đúng. Gợi ý:

| Query (không trùng chữ) | Đoạn đích mong đợi | Vì sao keyword trượt |
|---|---|---|
| "làm sao gọi mạng bất đồng bộ trong Python?" | đoạn về `httpx` / `async` | query không có chữ "httpx"/"async" |
| "cách bắt AI trả về đúng định dạng máy đọc được" | đoạn structured output | query không có chữ "JSON"/"structured" |
| "hai con vật khác loài" | đoạn con mèo / con chó | query không có chữ "mèo"/"chó" |

Chép **2–3 ví dụ thật** (query + top-3 + nhận xét) vào `README.md`. Đây là "bằng chứng" mang đi phỏng vấn.

**Talking point (học thuộc 2–3 câu):**
- "Keyword search khớp *chuỗi ký tự*; semantic search khớp *nghĩa* qua khoảng cách vector."
- "Tôi embed cả kho và câu hỏi bằng cùng 1 model rồi xếp hạng bằng cosine similarity."
- "Chưa cần vector DB ở quy mô nhỏ — list + cosine là đủ; scale lên mới cần index (pgvector/HNSW)."

---

## Task 4 — Push lên GitHub + cập nhật README

**Checklist an toàn trước khi push:**
1. **KHÔNG commit secrets** (project này không dùng key nên yên tâm, nhưng vẫn check `git status`).
2. **`vectors.json`**: là output tái tạo được. Với repo học tập, commit 1 bản nhỏ để người khác chạy `search`
   ngay cũng ổn; hoặc gitignore và để README hướng dẫn chạy `build`. Chọn 1 và ghi rõ trong README.
3. Cập nhật `requirements.txt` thêm `sentence-transformers==...`.

Đang ở branch `dev`:

```bash
cd D:\Projects\ai-engineer-journey
git add week4/semantic_search/ requirements.txt
git status                       # xem chắc KHÔNG có gì nhạy cảm
git commit -m "week4: semantic search mini (local, sentence-transformers)"
git push origin dev
```

> Muốn lên `main` thì tạo Pull Request `dev → main`, đừng push thẳng vào `main`.

---

## Thứ tự làm gợi ý (~1h30)

1. Setup (10') → 2. `embed_texts` + `build_index` + soát `corpus.txt` (Task 1, 30') →
3. `cosine_similarity` + `search` (Task 2, 25') → 4. Test 2–3 ví dụ không trùng chữ (Task 3, 15') →
5. README + push (Task 4, 10').

## Tự kiểm tra đã hiểu

- [ ] Giải thích vì sao lưu vector ra file (cache) thay vì embed lại mỗi lần.
- [ ] Viết lại công thức cosine từ trí nhớ + vì sao phải guard chia 0.
- [ ] Nói được vì sao all-MiniLM yếu tiếng Việt và cách khắc phục (đổi model đa ngôn ngữ).
- [ ] Có 2–3 ví dụ query-không-trùng-chữ mà vẫn ra đúng đoạn.
