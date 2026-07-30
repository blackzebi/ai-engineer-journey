"""
db_docs.py — Mở rộng schema bảng chunks + chống ingest trùng

Nối tiếp `week4/pgvector/vector_ops.py`. Tuần 4 bảng `chunks` chỉ có:

    id · source · content · embedding vector(384)

Module này thêm 5 cột + 1 bảng, mỗi thứ giải một vấn đề cụ thể:

    doc_type      -> thống kê theo định dạng, phát hiện loader nào đọc hụt
    title         -> hiện tên tài liệu người-đọc-được khi trích nguồn
    page          -> trích nguồn kiểu 'file X, trang 12'
    heading       -> địa chỉ thay thế cho DOCX/MD (không có khái niệm trang)
    chunk_index   -> thứ tự chunk trong tài liệu; cùng `source` tạo khoá chống trùng
    bảng documents-> nhớ file nào đã ingest + hash nội dung -> chạy lại KHÔNG nhân đôi

Không sửa thẳng vector_ops.py của tuần 4: schema nên tiến hoá bằng MIGRATION cộng thêm,
không phải bằng DROP TABLE rồi viết lại.

Chạy migration (container rag-pg phải Up):
    python db_docs.py
"""

from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))

from vector_ops import EMBED_DIM, get_conn, to_pgvector  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def migrate_chunks_table(conn) -> None:
    """Thêm doc_type / title / page / heading / chunk_index vào bảng `chunks` đã có.

    `ADD COLUMN IF NOT EXISTS` -> idempotent: chạy 1 lần hay 10 lần đều cùng kết quả,
    không lỗi. Bắt buộc, vì trong thực tế không ai chắc migration đã chạy chưa.

    `page` để NULL được (không NOT NULL): DOCX/MD/TXT không có khái niệm trang. NULL ở đây
    nghĩa là "không áp dụng", khác hẳn "quên điền" — ép NOT NULL là buộc phải bịa số 0.

    Unique index đặt ở TẦNG DB chứ không chỉ check trong Python: code có bug, DB thì không.
    Có ràng buộc rồi thì INSERT trùng nổ NGAY thay vì âm thầm nhân đôi dữ liệu.
    """
    cols = [
        ("doc_type", "text"),
        ("title", "text"),
        ("page", "int"),
        ("heading", "text"),
        ("chunk_index", "int"),
    ]
    with conn.cursor() as cur:
        for name, dtype in cols:
            # f-string ở đây chấp nhận được vì tên cột do mình viết cứng, không phải input
            # người dùng. Với GIÁ TRỊ thì luôn dùng placeholder %s.
            cur.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {name} {dtype};")

        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS chunks_source_idx "
            "ON chunks (source, chunk_index);"
        )

    conn.commit()
    print("✅ Đã migrate bảng chunks")


def create_documents_table(conn) -> None:
    """Tạo bảng `documents` — sổ theo dõi file nào đã ingest, với nội dung nào.

    Lưu file_hash chứ không lưu mtime: mtime đổi khi chỉ copy file / đổi máy dù nội dung y
    nguyên -> ingest lại vô ích. Hash đổi <=> nội dung THẬT SỰ đổi. Đây là nền của
    incremental update (tuần 4 dùng reset=True, xoá sạch rồi nạp lại toàn bộ).

    ⚠️ `CREATE TABLE IF NOT EXISTS` là idempotent nhưng KHÔNG phải migration: nếu bảng đã
    tồn tại, nó bỏ qua hoàn toàn -> sửa định nghĩa ở đây rồi chạy lại thì DB không đổi gì.
    Muốn đổi schema của bảng đang có thì phải ALTER TABLE.
    """
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "  source       text PRIMARY KEY,"
            "  doc_type     text,"
            "  title        text,"
            "  file_hash    text NOT NULL,"
            "  n_chunks     int,"
            "  ingested_at  timestamptz DEFAULT now()"
            ");"
        )

    conn.commit()
    print("✅ Bảng documents sẵn sàng")


def file_hash(path: str) -> str:
    """SHA-256 của nội dung file.

    Đọc theo khối 64KB thay vì f.read() cả file: file 200MB đọc 1 phát là ngốn 200MB RAM.
    iter(callable, sentinel) = gọi hàm liên tục cho tới khi trả về giá trị sentinel (b"").
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def should_ingest(conn, source: str, fhash: str) -> str:
    """Trả 'skip' | 'new' | 'changed' — 3 nhánh của incremental ingest.

    'changed' phải tách riêng khỏi 'new' vì nó BẮT BUỘC xoá chunk cũ trước. Quên bước xoá
    = tài liệu tồn tại 2 phiên bản trong DB, retrieval trả về đoạn đã lỗi thời mà không
    cách nào biết. Lỗi im lặng -> phải chặn bằng thiết kế, không bằng trí nhớ.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT file_hash FROM documents WHERE source = %s;", (source,))
        row = cur.fetchone()

    if row is None:
        return "new"
    return "skip" if row[0] == fhash else "changed"


def delete_document(conn, source: str) -> int:
    """Xoá mọi chunk + dòng documents của 1 source. Trả số chunk đã xoá.

    Thứ tự: chunks TRƯỚC, documents SAU. Ngược lại mà crash giữa chừng thì còn chunk mồ côi,
    không dòng documents nào trỏ tới, không ai biết mà dọn.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source = %s;", (source,))
        n = cur.rowcount            # lấy NGAY, trước khi execute lệnh kế tiếp
        cur.execute("DELETE FROM documents WHERE source = %s;", (source,))
    conn.commit()
    return n


# rows = [(source, content, embedding, doc_type, title, page, heading, chunk_index), ...]
ChunkRow = tuple[str, str, list[float], str, str, int | None, str | None, int]


def insert_chunks_batch(conn, rows: list[ChunkRow]) -> int:
    """INSERT nhiều chunk kèm metadata. Bản 8 cột của hàm cùng tên ở vector_ops.py.

    8 cột = 9 cột của bảng trừ `id` (bigserial, DB tự sinh).

    Luôn dùng placeholder %s (psycopg tự escape) và luôn batch bằng executemany —
    mỗi execute là 1 round-trip tới DB, 500 chunk insert lẻ = 500 round-trip.
    """
    params = [
        (src, content, to_pgvector(vec), dtype, title, page, heading, idx)
        for (src, content, vec, dtype, title, page, heading, idx) in rows
    ]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks "
            "(source, content, embedding, doc_type, title, page, heading, chunk_index) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
            params,
        )

    conn.commit()
    return len(params)


def record_document(conn, source: str, doc_type: str, title: str, fhash: str, n_chunks: int) -> None:
    """Ghi/cập nhật 1 dòng trong `documents` sau khi ingest xong tài liệu đó.

    UPSERT bằng `ON CONFLICT (source) DO UPDATE` — gọn và an toàn hơn tự SELECT rồi rẽ
    nhánh INSERT/UPDATE (cách tự làm có race condition). `EXCLUDED` là bảng ảo chứa dòng
    định insert nhưng bị đụng khoá.

    ⚠️ Phải gọi SAU khi insert chunk thành công. Ghi trước mà insert lỗi giữa chừng -> DB
    tưởng đã nạp xong -> lần sau 'skip' -> tài liệu vĩnh viễn thiếu chunk.
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (source, doc_type, title, file_hash, n_chunks) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (source) DO UPDATE SET "
            "  doc_type = EXCLUDED.doc_type,"
            "  title = EXCLUDED.title,"
            "  file_hash = EXCLUDED.file_hash,"
            "  n_chunks = EXCLUDED.n_chunks,"
            "  ingested_at = now();",
            (source, doc_type, title, fhash, n_chunks),
        )

    conn.commit()


def stats_by_doc_type(conn) -> list[tuple]:
    """Phân bố chunk theo định dạng — dùng để kiểm tra sức khoẻ của pipeline.

    Cách đọc kết quả:
      - doc_type nào 0 chunk mà folder rõ ràng có file loại đó -> loader chết im lặng
      - pdf 3 file mà chỉ 5 chunk -> gần như chắc chắn PDF scan / extract_text() trả rỗng
      - số chunk/file lệch hẳn giữa các loại -> normalize hoặc chunker đang bất thường
      - có nhóm doc_type = NULL -> dữ liệu cũ từ trước khi migrate, cần dọn
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_type, COUNT(*) AS n_chunks, COUNT(DISTINCT source) AS n_files "
            "FROM chunks GROUP BY doc_type ORDER BY n_chunks DESC;"
        )
        return cur.fetchall()


def main() -> None:
    """Chạy migration rồi in schema hiện tại. Chạy 2 lần liên tiếp phải đều không lỗi."""
    with get_conn() as conn:
        migrate_chunks_table(conn)
        create_documents_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'chunks' ORDER BY ordinal_position;"
            )
            print(f"📋 Bảng chunks (embedding {EMBED_DIM} chiều):")
            for name, dtype in cur.fetchall():
                print(f"   - {name:<12} {dtype}")
    print("\n🎉 Schema sẵn sàng.")


if __name__ == "__main__":
    main()
