"""
seed_bulk_rows.py — Nhân bản chunks lên ~50k dòng để index có ý nghĩa (nợ tuần 5)

    INPUT :  bảng chunks đang có N dòng thật
    OUTPUT:  bảng chunks có ~50k dòng (N thật + phần nhân bản có nhãn rõ ràng)

Vì sao món này còn nợ từ tuần 5:
    week5/indexTuning/hnsw_index.py đã in sẵn cảnh báo "dưới 1000 dòng, planner chọn Seq
    Scan dù có index — mọi số ef_search là NHIỄU". Cảnh báo đó vẫn đang đúng, nên toàn bộ
    phần đo index của T3 tuần 5 hiện chưa có số thật.

Làm một lần dùng cho HAI việc:
    · HNSW (tuần 5) — cuối cùng cũng đo được Index Scan vs Seq Scan
    · GIN  (hôm nay) — keyword search cũng cần đủ dòng mới thấy Bitmap Index Scan

⚠️ Dữ liệu nhân bản KHÔNG dùng để đánh giá CHẤT LƯỢNG kết quả. 20 bản sao của cùng một
   chunk sẽ chiếm sạch top-k và làm mọi phép đo recall thành vô nghĩa. Nó chỉ để đo TỐC ĐỘ
   và xem PLAN. Dọn sạch bằng `--clean` trước khi chạy compare_branches.py.


    python seed_bulk_rows.py --dry            # xem câu SQL sẽ chạy, KHÔNG chạm DB
    python seed_bulk_rows.py --target 50000   # nhân bản
    python seed_bulk_rows.py --clean          # xoá sạch phần nhân bản
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))

from vector_ops import get_conn  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Hậu tố đánh dấu dòng nhân bản. Phải nhận diện được bằng LIKE để xoá sạch — dữ liệu rác
# không dọn được là dữ liệu rác vĩnh viễn.
SYNTHETIC_SUFFIX = "#synthetic-"
DEFAULT_TARGET_ROWS = 50_000


def build_duplicate_sql(copies: int) -> str:
    """Trả câu INSERT nhân bản mỗi chunk thành `copies` bản. Hàm thuần — không chạm DB.

    Ví dụ: build_duplicate_sql(19) -> câu SQL tạo 19 bản sao cho mỗi dòng gốc
    """

    return (
        "INSERT INTO chunks "
        "(source, content, embedding, doc_type, title, page, heading, chunk_index) "
        "SELECT c.source || '" + SYNTHETIC_SUFFIX + "' || g.i::text, "
        "       c.content, c.embedding, c.doc_type, c.title, c.page, c.heading, c.chunk_index "
        "FROM chunks c, generate_series(1, " + str(int(copies)) + ") AS g(i) "
        "WHERE c.source NOT LIKE '%" + SYNTHETIC_SUFFIX + "%';"
    )


def count_rows(conn) -> tuple[int, int]:
    """(số dòng thật, số dòng nhân bản)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FILTER (WHERE source NOT LIKE %s), "
            "       count(*) FILTER (WHERE source LIKE %s) FROM chunks;",
            (f"%{SYNTHETIC_SUFFIX}%", f"%{SYNTHETIC_SUFFIX}%"),
        )
        return cur.fetchone()


def clean_synthetic(conn) -> int:
    """Xoá sạch dòng nhân bản, trả số dòng đã xoá.

    Chạy CÁI NÀY trước khi đo chất lượng (compare_branches.py) — 20 bản sao của cùng một
    chunk sẽ chiếm sạch top-k và mọi kết luận về recall thành vô nghĩa.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source LIKE %s;", (f"%{SYNTHETIC_SUFFIX}%",))
        deleted = cur.rowcount
    conn.commit()
    return deleted


def main() -> None:
    target = DEFAULT_TARGET_ROWS
    if "--target" in sys.argv:
        target = int(sys.argv[sys.argv.index("--target") + 1])

    if "--dry" in sys.argv:
        print("Câu SQL sẽ chạy (đọc kỹ trước khi chạy thật):\n")
        print(build_duplicate_sql(19))
        return

    with get_conn() as conn:
        real, synthetic = count_rows(conn)
        print(f"📊 Trước: {real} dòng thật + {synthetic} dòng nhân bản")

        if "--clean" in sys.argv:
            print(f"🧹 Đã xoá {clean_synthetic(conn)} dòng nhân bản")
            return

        if real == 0:
            raise SystemExit("❌ Chưa có dòng thật nào. Chạy week5/ingestDocs/ingest_docs.py trước.")

        if synthetic:
            raise SystemExit("❌ Đang có sẵn dòng nhân bản. Chạy `--clean` trước để tránh nhân chồng.")

        copies = max(1, (target // real) - 1)
        print(f"🔁 Nhân mỗi dòng thành {copies} bản sao -> ~{real * (copies + 1)} dòng")

        with conn.cursor() as cur:
            cur.execute(build_duplicate_sql(copies))
        conn.commit()

        real_after, synthetic_after = count_rows(conn)
        print(f"📊 Sau: {real_after} thật + {synthetic_after} nhân bản = {real_after + synthetic_after}")

    print("\n👉 Bây giờ chạy lại 2 phép đo, LẦN NÀY số mới có nghĩa:")
    print("   python ../../week5/indexTuning/hnsw_index.py")
    print("   python keyword_search.py \"<câu hỏi bất kỳ>\" --plan")
    print("👉 Đo xong: `python seed_bulk_rows.py --clean` TRƯỚC khi chạy compare_branches.py")

    # ✅ ĐẠT khi:
    #   1. `--dry` in câu SQL có đủ 8 cột, không có 'id', không có 'tsv'
    #   2. Chạy thật xong tổng số dòng >= ~50k
    #   3. hnsw_index.py LÚC NÀY in [index] thay vì [SEQ SCAN] — đây là con số tuần 5 còn nợ
    #   4. keyword_search.py --plan hiện "Bitmap Index Scan on chunks_tsv_gin_idx"
    #   5. `--clean` đưa bảng về đúng số dòng thật ban đầu

if __name__ == "__main__":
    main()
