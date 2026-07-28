"""
vector_ops.py — Bảng chunks + INSERT embedding + query top-k (T4 tuần 4, block AI core)

Nối tiếp `pg_smoke_test.py` (T3: kết nối + bật extension). Hôm nay làm 3 việc:
  Task 5. Tạo bảng thật `chunks(id, source, content, embedding vector(384))`  — vì sao 384
  Task 6. INSERT embedding từ Python + phân biệt `<=>` (cosine DISTANCE) vs `<->` (L2)
  Task 7. Query top-k: `ORDER BY embedding <=> %s LIMIT 5` — đối chiếu với cosine tự cài tuần 3

Chạy (container rag-pg phải đang Up):
    python vector_ops.py
"""

import os
import sys

import psycopg
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:ragpass@localhost:5432/study_rag",
)

# Khớp model embedding tuần này: paraphrase-multilingual-MiniLM-L12-v2 -> 384 chiều.
# Đổi model = đổi số này = phải DROP + tạo lại bảng (không migrate mềm được).
EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Helper — dùng lại cho ingest.py / search_pg.py
# ---------------------------------------------------------------------------
def get_conn() -> "psycopg.Connection":
    """Mở kết nối Postgres. Lỗi kết nối -> thoát kèm gợi ý sửa (giống pg_smoke_test)."""
    try:
        return psycopg.connect(DATABASE_URL)
    except psycopg.OperationalError as e:
        raise SystemExit(
            f"❌ Không kết nối được DB: {e}\n"
            "   - Container Up chưa?   docker ps\n"
            "   - DATABASE_URL đúng?   xem .env"
        )


def to_pgvector(vec: list[float]) -> str:
    """Đổi list Python -> literal pgvector: [0.1,0.2,...].

    pgvector nhận text dạng '[a,b,c]'. `str(list)` ra đúng dạng đó nên dùng luôn.
    (Cách "xịn" hơn: `pip install pgvector` rồi `register_vector(conn)` để truyền list trực tiếp
     — tuần này làm tay cho hiểu bản chất, tuần sau muốn gọn thì đổi.)
    """
    return str(list(vec))


# ---------------------------------------------------------------------------
# Task 5 — bảng chunks(id, source, content, embedding vector(384))
# ---------------------------------------------------------------------------
def create_chunks_table(conn: "psycopg.Connection", drop: bool = False) -> None:
    """Tạo bảng `chunks` — nơi sẽ chứa toàn bộ chunk của kho notes.

    Ý nghĩa từng cột:
      id        bigserial PK   -> khoá tự tăng
      source    text           -> file gốc của chunk (vd 'Tuan-02/CN/README.md').
                                  BẮT BUỘC cho RAG: sau này trả lời phải trích nguồn.
      content   text           -> chữ gốc của chunk (đưa vào prompt LLM ở tuần 5)
      embedding vector(384)    -> vector nghĩa của content

    Câu hỏi tự trả lời (ghi vào notes): vì sao 384? -> vì model
    paraphrase-multilingual-MiniLM-L12-v2 xuất ra 384 số. Cột phải khớp, sai là INSERT lỗi.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        if drop:
            cur.execute("DROP TABLE IF EXISTS chunks;")
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS chunks ("
            f"  id bigserial PRIMARY KEY,"
            f"  source text NOT NULL,"
            f"  content text NOT NULL,"
            f"  embedding vector({EMBED_DIM})"
            f");"
        )
    conn.commit()
    print(f"✅ Bảng chunks sẵn sàng (embedding vector({EMBED_DIM}))")


# ---------------------------------------------------------------------------
# Task 6 — INSERT embedding + phân biệt <=> và <->
# ---------------------------------------------------------------------------
def insert_chunk(
    conn: "psycopg.Connection",
    source: str,
    content: str,
    embedding: list[float],
) -> int:
    """INSERT 1 chunk, trả về id vừa tạo.

    Bẫy:
      - LUÔN dùng placeholder %s (psycopg tự escape). Đừng f-string SQL -> SQL injection.
      - Sai số chiều -> lỗi 'expected 384 dimensions, not N'.
      - Quên commit -> chạy xong, `docker exec psql` SELECT thấy 0 dòng, tưởng code sai.
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chunks (source, content, embedding) VALUES (%s, %s, %s) RETURNING id;",
            (source, content, to_pgvector(embedding)),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def insert_chunks_batch(
    conn: "psycopg.Connection",
    rows: list[tuple[str, str, list[float]]],
) -> int:
    """INSERT nhiều chunk 1 lần (dùng cho ingest.py). rows = [(source, content, embedding), ...].

    Vì sao batch: mỗi execute là 1 round-trip tới DB. 500 chunk insert lẻ = 500 round-trip.
    """
    params = [(s, c, to_pgvector(v)) for s, c, v in rows]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (source, content, embedding) VALUES (%s, %s, %s);",
            params,
        )
    conn.commit()
    return len(params)


def compare_operators(conn: "psycopg.Connection", query_vec: list[float]) -> None:
    """So sánh 3 toán tử khoảng cách trên CÙNG một query vector.

    🔑 Điểm dễ nhầm nhất tuần này:
        pgvector trả về **DISTANCE** (khoảng cách), KHÔNG phải similarity.
        -> Số NHỎ = GẦN = liên quan hơn.  Ngược hoàn toàn với cosine similarity tuần 3
           (số LỚN = giống hơn).
        -> Quy đổi:  cosine_similarity = 1 - (embedding <=> query)

    | Toán tử | Tên              | Ý nghĩa                          |
    |---------|------------------|----------------------------------|
    | `<=>`   | cosine distance  | 1 - cos(a,b), trong [0, 2]       |
    | `<->`   | L2 / Euclidean   | khoảng cách thẳng giữa 2 điểm    |
    | `<#>`   | negative inner product | -(a·b) — pgvector để dấu âm cho "nhỏ = gần" |

    Quan sát rồi ghi notes: thứ hạng theo <=> và theo <-> có giống nhau không?
    (Với vector đã chuẩn hoá norm=1 thì 2 thứ hạng TRÙNG nhau — sentence-transformers
     mặc định KHÔNG normalize, nên có thể lệch. Đây là 1 câu phỏng vấn hay.)
    """
    q = to_pgvector(query_vec)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, left(content, 50),"
            "       embedding <=> %s AS cosine_distance,"
            "       1 - (embedding <=> %s) AS cosine_similarity,"
            "       embedding <-> %s AS l2_distance "
            "FROM chunks ORDER BY cosine_distance LIMIT 5;",
            (q, q, q),
        )
        for row in cur.fetchall():
            print(row)


# ---------------------------------------------------------------------------
# Task 7 — query top-k
# ---------------------------------------------------------------------------
def search_top_k(
    conn: "psycopg.Connection",
    query_vec: list[float],
    k: int = 5,
) -> list[tuple[int, str, str, float]]:
    """Trả top-k chunk gần nghĩa nhất: (id, source, content, similarity).

    SQL xương sống của RAG:
        SELECT id, source, content, 1 - (embedding <=> %s) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s      -- sắp theo DISTANCE tăng dần = gần nhất trước
        LIMIT %s;

    Bẫy:
      - ORDER BY similarity DESC cũng ra kết quả đúng, NHƯNG sắp theo biểu thức 1-... thì
        index vector (HNSW/IVFFlat) KHÔNG dùng được. Luôn ORDER BY <toán tử khoảng cách>.
      - Chưa có index thì Postgres quét tuần tự — vài nghìn dòng vẫn nhanh, đủ cho tuần này.
        Index là chuyện tuần 6 (tối ưu retrieval).
    """
    q = to_pgvector(query_vec)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, source, content, 1 - (embedding <=> %s) AS similarity "
            "FROM chunks ORDER BY embedding <=> %s LIMIT %s;",
            (q, q, k),
        )
        return cur.fetchall()


def main() -> None:
    """Chạy thử 3 task với 1 vector giả (chưa cần model) để kiểm tra SQL đúng.

    Sau khi 3 task chạy được, sang ingest.py để đổ dữ liệu thật.
    """
    fake_vec = [0.0] * EMBED_DIM
    fake_vec[0] = 1.0

    print(f"🔌 {DATABASE_URL}")
    with get_conn() as conn:
        create_chunks_table(conn, drop=True)                      # Task 5
        new_id = insert_chunk(conn, "demo.md", "hello chunk", fake_vec)  # Task 6
        print("✅ INSERT id =", new_id)
        compare_operators(conn, fake_vec)                         # Task 6
        for row in search_top_k(conn, fake_vec, k=5):             # Task 7
            print(row)
    print("\n🎉 SQL nền đã chạy. Sang block PROJECT: ingest.py với dữ liệu thật.")


if __name__ == "__main__":
    main()
