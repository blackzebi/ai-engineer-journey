"""
pg_smoke_test.py — Smoke test PostgreSQL + pgvector (T3 tuần 4)

Mục tiêu: kết nối từ Python tới container Postgres, xác nhận:
  1. Kết nối OK  -> chạy được `SELECT 1`                        (Task 2)
  2. Bật extension `vector`                                     (Task 3)
  3. Hiểu kiểu cột vector(n): tạo bảng vector(384), insert, select  (Task 3)

Concept nền (vì sao vector, vì sao 384): study notes T3.

Chạy (sau khi đã `docker run` container + pip install psycopg[binary]):
    python pg_smoke_test.py
"""

import os
import sys

# psycopg 3 (KHÔNG phải psycopg2). Cài: pip install "psycopg[binary]"
import psycopg

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Đọc connection string từ .env (DATABASE_URL). Fallback = giá trị mặc định của
# lệnh docker run trong guide.md. Dạng: postgresql://<user>:<pass>@<host>:<port>/<db>
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:ragpass@localhost:5432/study_rag",
)

# Số chiều vector PHẢI khớp model embedding. all-MiniLM-L6-v2 / multilingual-MiniLM = 384.
# Đổi model (vd OpenAI text-embedding-3-small = 1536) thì đổi luôn con số này + tạo lại bảng.
EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Task 2 — kết nối + SELECT 1
# ---------------------------------------------------------------------------
def get_conn() -> "psycopg.Connection":
    """Mở kết nối tới Postgres bằng DATABASE_URL."""
    try:
        return psycopg.connect(DATABASE_URL)
    except psycopg.OperationalError as e:
        # psycopg gói MỌI lỗi kết nối (refused, sai pass, sai db) vào OperationalError
        # — KHÔNG phải ConnectionRefusedError của Python.
        raise SystemExit(
            f"❌ Không kết nối được DB: {e}\n"
            "   - Container Up chưa?        docker ps\n"
            "   - Đúng user/pass/db chưa?   xem lại DATABASE_URL"
        )


def check_select_1(conn: "psycopg.Connection") -> None:
    """Chạy `SELECT 1` — bằng chứng nhỏ nhất là 'Python nói chuyện được với DB'."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")
        row = cur.fetchone()
        assert row[0] == 1
    print("✅ SELECT 1 OK ->", row)


# ---------------------------------------------------------------------------
# Task 3a — bật extension vector
# ---------------------------------------------------------------------------
def enable_pgvector(conn: "psycopg.Connection") -> None:
    """Bật extension `vector` trong database hiện tại (chỉ cần 1 lần / database)."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        ver = cur.fetchone()
        conn.commit()
        print("✅ pgvector bật, version =", ver[0])


# ---------------------------------------------------------------------------
# Task 3b — hiểu kiểu cột vector(n)
# ---------------------------------------------------------------------------
def demo_vector_column(conn: "psycopg.Connection") -> None:
    """Tạo bảng có cột vector(EMBED_DIM), insert 1 vector, select lại.
    Mục tiêu: 'thấy tận mắt' vector(n) — n phải khớp số chiều model.
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS smoke_items;")
        cur.execute(
            f"CREATE TABLE smoke_items ("
            f"  id bigserial PRIMARY KEY,"
            f"  content text,"
            f"  embedding vector({EMBED_DIM})"
            f");"
        )

        demo_vec = [0.0] * EMBED_DIM
        demo_vec[0] = 1.0
        cur.execute(
            "INSERT INTO smoke_items (content, embedding) VALUES (%s, %s);",
            ("hello pgvector", str(demo_vec)),
        )
        cur.execute("SELECT id, content, vector_dims(embedding) FROM smoke_items;")
        rows = cur.fetchall()
    conn.commit()
    print("✅ Bảng vector demo:", rows)


def main() -> None:
    print(f"🔌 Kết nối: {DATABASE_URL}")
    with get_conn() as conn:
        check_select_1(conn)      # Task 2
        enable_pgvector(conn)     # Task 3a
        demo_vector_column(conn)  # Task 3b
    print("\n🎉 Smoke test xong. Hạ tầng pgvector sẵn sàng cho T4 (ingest + query top-k).")


if __name__ == "__main__":
    main()
