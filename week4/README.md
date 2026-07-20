# Semantic Search mini

Semantic search nhỏ chạy **local, miễn phí**: embed các đoạn text (notes tuần 1–3) bằng
`sentence-transformers`, lưu vector ra JSON, rồi `search(query)` trả top-3 đoạn gần nghĩa nhất
bằng **cosine similarity tự cài**. Không cần API key, không cần vector DB — list Python là đủ.

Mục tiêu học: hiểu **retrieval** — trái tim của RAG — trước khi lên pgvector ở phần sau tuần 4.

## Stack

- Python 3.10+
- [`sentence-transformers`](https://www.sbert.net/) — model `paraphrase-multilingual-MiniLM-L12-v2`
  (đa ngôn ngữ, tốt cho tiếng Việt, 384 chiều, chạy local)
- Cosine similarity tự viết bằng `math` (không dùng numpy)

## Cách chạy

```bash
pip install sentence-transformers

# embed kho (chạy 1 lần, sinh vectors.json). Lần đầu tự tải model về máy.
python semantic_search.py build

# query
python semantic_search.py search "con mèo là gì?"
```

Sửa nội dung kho trong `corpus.txt` (mỗi dòng 1 đoạn) → chạy lại `build`.
Đổi `MODEL_NAME` cũng phải chạy lại `build` (vector của model khác nhau không so được).

## Cấu trúc

```
week4/
├── semantic_search.py   # code chính (build + search)
├── corpus.txt           # kho ~20 đoạn text (mỗi dòng 1 đoạn)
├── vectors.json         # sinh ra bởi `build` (cache vector)
├── guide.md             # hướng dẫn làm 4 task
└── README.md
```

## Demo: semantic ≠ keyword

Cách kiểm tra top-1 tách biệt: điểm cosine của #1 phải bỏ xa #2.

**Sanity check** — `search "con mèo là gì?"`:

```
1. [0.6136] Con mèo không phải con chó nhưng cả hai đều là động vật sống trên Trái Đất.
2. [0.1389] System prompt định hướng cách model trả lời theo vai trò bạn đặt ra.
3. [0.0997] Trong Python bạn có thể dùng asyncio để chạy một hàm async.
```

→ #1 (0.61) bỏ xa #2 (0.14): ranking hoạt động đúng. (Ví dụ này còn *trùng chữ* "con mèo" nên
chưa chứng minh được semantic ≠ keyword — dùng 2 ví dụ dưới cho việc đó.)

**Ví dụ Task 3 — query KHÔNG trùng từ nào với đoạn đích** (chạy rồi điền kết quả thật):

### Ví dụ 1
- **Query:** `làm sao gọi mạng bất đồng bộ trong Python?` _(không có chữ "httpx"/"async")_
- **Top-3:**
  1. [0.5132] httpx là thư viện HTTP client hỗ trợ cả sync và async trong Python.
  2. [0.3649] Trong Python bạn có thể dùng asyncio để chạy một hàm async.
  3. [0.2756] REST API dùng các method GET/POST/PUT/DELETE trên tài nguyên qua URL.
- **Nhận xét:** ra đúng đoạn về `httpx`/`async` dù không trùng chữ → khớp *nghĩa*, không phải *chữ*.

### Ví dụ 2
- **Query:** `cách bắt AI trả về đúng định dạng máy đọc được` _(không có chữ "JSON"/"structured")_
- **Top-3:**
  1. [0.6287] Trong Anthropic có thể định nghĩa tool riêng để AI giải quyết việc bên ngoài như tính toán, hỏi ngày giờ.
  2. [0.4868] Structured output ép model trả JSON đúng schema để máy đọc được.
  3. [0.4659] System prompt định hướng cách model trả lời theo vai trò bạn đặt ra.
- **Nhận xét:** ra đúng đoạn về cách bắt AI trả về gần đúng câu hỏi.

## Kiến thức rút ra từ project

**RAG bắt đầu từ retrieval.** Toàn bộ project này chính là bước *retrieve* của RAG thu nhỏ:
`embed kho → embed câu hỏi → xếp hạng bằng khoảng cách vector → lấy top-k`. Tuần sau chỉ việc
thay "list Python + cosine" bằng "vector DB (pgvector)" là thành RAG thật.

**Embedding = biến chữ thành vector theo nghĩa.** Đoạn nào gần nghĩa thì vector gần nhau. Nhờ vậy
search khớp *ý* chứ không khớp *ký tự* → tìm đúng cả khi câu hỏi không chứa từ khoá của đoạn.

**Cosine similarity** đo độ giống về *hướng* của 2 vector: `dot(a,b) / (‖a‖·‖b‖)`, giá trị trong
`[-1, 1]`. Phải **guard chia 0** khi một vector có norm = 0. Điểm cosine chỉ có ý nghĩa **so thứ hạng
trong cùng 1 kho / cùng 1 model**, không so tuyệt đối giữa 2 model khác nhau.

**Chọn model phải khớp ngôn ngữ dữ liệu.** `all-MiniLM-L6-v2` thiên tiếng Anh → tiếng Việt kém.
Đổi sang `paraphrase-multilingual-MiniLM-L12-v2` (đa ngôn ngữ) cho kết quả tốt hơn hẳn mà không đổi code.
sentence-transformers **không có `input_type`** (khác Voyage/OpenAI) — dùng chung 1 model cho cả kho lẫn query.

**Cache vector.** Embedding của một đoạn không đổi → lưu `vectors.json` một lần, các lần `search` sau
chỉ đọc lại, khỏi embed lại (nhanh hơn, và nếu dùng API thì đỡ tốn tiền).

**Khi nào cần vector DB.** Quy mô nhỏ (vài chục–vài trăm đoạn): list + cosine duyệt tuyến tính là đủ.
Lớn hơn: cần index vector chuyên dụng (pgvector / HNSW) để search nhanh thay vì so từng phần tử.

### Talking points phỏng vấn

- "Keyword search khớp *chuỗi ký tự*; semantic search khớp *nghĩa* qua khoảng cách vector."
- "Embed cả kho và câu hỏi bằng cùng 1 model, xếp hạng bằng cosine similarity, lấy top-k."
- "Chọn embedding model phải khớp ngôn ngữ dữ liệu — tôi đổi sang model đa ngôn ngữ cho tiếng Việt."
- "Quy mô nhỏ chưa cần vector DB; scale lên mới cần index (pgvector/HNSW)."
