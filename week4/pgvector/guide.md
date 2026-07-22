# Guide — PostgreSQL + pgvector (T3 tuần 4)

> Hạ tầng mới hoàn toàn — **không vội, làm từng bước**. Mục tiêu hôm nay chỉ là:
> dựng được 1 Postgres có pgvector, kết nối từ Python, chạy `SELECT 1`, bật extension
> `vector` và hiểu kiểu cột `vector(n)`. Chưa ingest/query gì cả — đó là T4.
>
> Code khung ở [`pg_smoke_test.py`](pg_smoke_test.py) — bạn điền 4 chỗ `# TODO`.
> Concept nền (chunking, embedding, vì sao 384 chiều): study notes T2–T3.

## Bức tranh tổng thể

```
Docker Desktop
   └── container "rag-pg"  (image pgvector/pgvector:pg17)
          └── database "study_rag"
                 ├── EXTENSION vector          <- bật 1 lần
                 └── bảng có cột vector(384)    <- 384 = số chiều model embedding
                          ▲
     Python (psycopg) ────┘  kết nối qua connection string, chạy SQL
```

Đây là "cái hộp" sẽ thay cho `vectors.json` + list Python ở semantic search mini. T4 mới đổ chunk vào.

## Chuẩn bị (1 lần)

```bash
pip install "psycopg[binary]"
# rồi thêm vào requirements.txt:  pip freeze | findstr psycopg   -> chép dòng psycopg==... vào file
```

> Dùng **psycopg 3** (`import psycopg`), KHÔNG phải `psycopg2`. Bản `[binary]` khỏi cần build C.

---

## Task 1 — Cài Docker Desktop + kéo image `pgvector/pgvector:pg17`

**Mục tiêu:** có Docker chạy + image pgvector nằm sẵn trên máy.

- [X] **Cài Docker Desktop** (nếu chưa có): tải ở https://www.docker.com/products/docker-desktop → cài → mở lên → đợi icon Docker báo **"running"**.
- [X] Kiểm tra Docker sống:

  ```bash
  docker --version
  docker run --rm hello-world      # in "Hello from Docker!" là OK
  ```

- [X] **Kéo image pgvector** (Postgres 17 đã kèm sẵn extension vector):

  ```bash
  docker pull pgvector/pgvector:pg17
  ```

**Bẫy:**
- Windows: Docker Desktop cần **WSL2** bật. Nếu báo lỗi, mở PowerShell admin chạy `wsl --install` rồi khởi động lại.
- Kéo image lần đầu hơi lâu (~vài trăm MB) — bình thường.

**Xong khi:** `docker images` thấy dòng `pgvector/pgvector  pg17`.

---

## Task 2 — Chạy container + tạo DB `study_rag` + kết nối psycopg (SELECT 1)

**Mục tiêu:** container chạy, có database `study_rag`, Python chạy được `SELECT 1`.

### 2.1 Chạy container

Cách gọn nhất: để Postgres **tự tạo luôn** database `study_rag` qua biến `POSTGRES_DB`.

```bash
docker run -d --name rag-pg ^
  -e POSTGRES_PASSWORD=ragpass ^
  -e POSTGRES_DB=study_rag ^
  -p 5432:5432 ^
  pgvector/pgvector:pg17
```

> `^` là nối dòng trong **CMD Windows**. Nếu dùng **PowerShell** đổi `^` thành `` ` ``; hoặc viết hết **1 dòng**:
> `docker run -d --name rag-pg -e POSTGRES_PASSWORD=ragpass -e` POSTGRES_DB=study_rag -p 5432:5432 pgvector/pgvector:pg17`

Giải thích cờ: `-d` chạy nền · `--name` đặt tên · `-e` biến môi trường · `-p 5432:5432` map cổng máy→container.

Kiểm tra: `docker ps` thấy `rag-pg` status **Up**. Xem log: `docker logs rag-pg` (đợi dòng "database system is ready to accept connections").

> **Cách thủ công (nếu KHÔNG dùng `POSTGRES_DB`):** vào container tạo DB bằng tay — để hiểu chuyện gì xảy ra:
> ```bash
> docker exec -it rag-pg psql -U postgres -c "CREATE DATABASE study_rag;"
> ```

### 2.2 Lưu connection string vào `.env`

Thêm vào `.env` ở gốc repo (đã có mẫu trong `.env.example`):

```
DATABASE_URL=postgresql://postgres:ragpass@localhost:5432/study_rag
```

Cấu trúc: `postgresql://<user>:<password>@<host>:<port>/<database>`
→ user `postgres` (mặc định của image) · pass `ragpass` · host `localhost` · port `5432` · db `study_rag`.

### 2.3 Kết nối từ Python — điền TODO 1 & 2

Trong [`pg_smoke_test.py`](pg_smoke_test.py):

- **TODO 1** (`get_conn`): `return psycopg.connect(DATABASE_URL)`.
- **TODO 2** (`check_select_1`): mở cursor → `cur.execute("SELECT 1;")` → `cur.fetchone()` → khẳng định `== (1,)`.

**Bẫy:**
- `connection refused` → container chưa Up / sai port / quên `-p 5432:5432`.
- `password authentication failed` → sai user/pass trong `DATABASE_URL`.
- `database "study_rag" does not exist` → quên `POSTGRES_DB` và chưa tạo DB thủ công.

**Xong khi:** chạy `python pg_smoke_test.py` in `✅ SELECT 1 OK -> (1,)`.

---

## Task 3 — `CREATE EXTENSION vector` + hiểu kiểu cột `vector(n)`

**Mục tiêu:** bật pgvector và hiểu vì sao `n` (số chiều) phải khớp model embedding.

### 3.1 Bật extension — điền TODO 3

- **TODO 3** (`enable_pgvector`): `cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")` → đọc version từ `pg_extension` → **`conn.commit()`** (DDL cũng cần commit vì psycopg mặc định **không autocommit**).

Extension `vector` thêm cho Postgres: **kiểu dữ liệu `vector`** + các **toán tử khoảng cách** (`<->` L2, `<=>` cosine, `<#>` inner product) + index vector (HNSW/IVFFlat). Không bật extension thì `vector(384)` báo lỗi "type vector does not exist".

### 3.2 Hiểu `vector(n)` — điền TODO 4

`vector(n)` = cột chứa mảng **n số thực**. `n` = **số chiều** của vector embedding, **cố định cho cả bảng**.

| Model embedding | Số chiều → `vector(n)` |
|---|---|
| all-MiniLM-L6-v2 / paraphrase-multilingual-MiniLM-L12-v2 | **384** → `vector(384)` |
| OpenAI text-embedding-3-small | 1536 → `vector(1536)` |
| Voyage voyage-3 | 1024 → `vector(1024)` |

> 🔑 **`n` phải khớp đúng model bạn dùng để embed.** Model xuất ra 384 số mà cột khai `vector(1536)` (hoặc ngược lại) → insert **lỗi ngay**: `expected N dimensions, not M`. Đây chính là lý do phải nhớ "model nào → bao nhiêu chiều". Tuần này ta dùng model 384 chiều nên là `vector(384)`.

- **TODO 4** (`demo_vector_column`): tạo bảng `smoke_items(id, content, embedding vector(384))`, insert 1 vector 384 phần tử (truyền dạng chuỗi `str([...])` — pgvector nhận literal `'[0,0,...]'`), rồi `SELECT vector_dims(embedding)` phải ra **384**.
- **Thử cho hiểu:** sau khi chạy OK, cố insert 1 vector **dài 300** → xem Postgres báo lỗi số chiều → ghi lỗi đó vào notes. Vấp 1 lần nhớ mãi.

**Xong khi:** `python pg_smoke_test.py` in đủ 3 dòng ✅ (SELECT 1 · pgvector version · bảng vector demo dims=384).

---

## Thứ tự làm gợi ý (~1h)

1. Task 1 cài/kéo Docker (20–30', phần lớn là đợi tải) →
2. Task 2 `docker run` + `.env` + điền TODO 1,2 (20') →
3. Task 3 điền TODO 3,4 + thử insert sai chiều (15').

## Lệnh Docker hay dùng (lưu lại)

```bash
docker ps                 # container đang chạy
docker logs rag-pg        # xem log (chờ "ready to accept connections")
docker stop rag-pg        # tắt (KHÔNG mất dữ liệu)
docker start rag-pg       # bật lại
docker rm -f rag-pg       # xoá hẳn container (MẤT dữ liệu vì chưa gắn volume)
docker exec -it rag-pg psql -U postgres -d study_rag   # vào psql trong container
```

> Muốn **dữ liệu không mất** khi xoá container: thêm `-v rag_pgdata:/var/lib/postgresql/data` vào `docker run`
> (hoặc dùng [`docker-compose.yml`](docker-compose.yml) đã cấu hình sẵn volume: `docker compose up -d`).

## Tự kiểm tra đã hiểu

- [ ] Giải thích `-p 5432:5432` làm gì và vì sao thiếu nó thì Python không kết nối được.
- [ ] Vì sao gọi `conn.commit()` sau `CREATE EXTENSION` / `CREATE TABLE`?
- [ ] `vector(n)` — `n` lấy từ đâu? Điều gì xảy ra nếu insert sai số chiều?
- [ ] Nhớ được connection string đầy đủ và từng phần nghĩa là gì.

**Talking points phỏng vấn:**
- "pgvector = Postgres + kiểu `vector` + toán tử khoảng cách (`<=>` cosine) → dùng ngay DB quen thuộc làm vector store."
- "Số chiều cột `vector(n)` phải khớp model embedding; đổi model là phải migrate cột."
- "Chạy DB qua Docker để môi trường tái lập được; gắn volume để dữ liệu bền."
