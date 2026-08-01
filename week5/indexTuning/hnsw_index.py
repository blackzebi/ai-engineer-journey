"""
hnsw_index.py — Index HNSW cho cột embedding + đo thời gian query trước/sau (T3 tuần 5)

    INPUT :  bảng `chunks` đã có dữ liệu (từ week5/ingestDocs/ingest_docs.py)
    OUTPUT:  1 index HNSW + số liệu thật: seq scan bao nhiêu ms, index scan bao nhiêu ms

    [chunks chưa index] --EXPLAIN ANALYZE--> Seq Scan, T0 ms
              |
              +-- CREATE INDEX USING hnsw (embedding vector_cosine_ops)
              |
    [chunks có index]  --EXPLAIN ANALYZE--> Index Scan, T1 ms
              |
              +-- đổi hnsw.ef_search 20/40/100 --> T1 đổi, recall đổi

So với tuần 4:
    tuần 4 (vector_ops.search_top_k):  ORDER BY embedding <=> %s  ->  quét toàn bảng
    hôm nay:                            đúng câu SQL đó, thêm index ->  Index Scan
                                        ^^^^ SQL không đổi một chữ, chỉ hạ tầng đổi.
    Đó là lý do tuần 4 bắt buộc `ORDER BY <toán tử>` chứ không `ORDER BY 1 - (...) DESC`:
    sắp theo biểu thức thì index không dùng được.

⚠️ Đọc kết quả cho đúng: dưới ~1000 dòng, planner chọn Seq Scan dù index tồn tại — và nó
   đúng. Khi đó mọi con số ef_search là NHIỄU, vì ef_search không tham gia vào Seq Scan.
   Muốn đo được lợi ích của index cần vài chục nghìn dòng trở lên.

Chạy (container rag-pg phải Up):
    python hnsw_index.py --dry          # self-check parser, KHÔNG cần DB
    python hnsw_index.py                # đo thật: before -> create -> after
"""

from __future__ import annotations

import os
import re
import sys
import time

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))

from vector_ops import EMBED_DIM, get_conn, to_pgvector  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Tên index đặt cứng để drop/create lại nhiều lần khi thử tham số.
HNSW_INDEX_NAME = "chunks_embedding_hnsw_idx"
IVFFLAT_INDEX_NAME = "chunks_embedding_ivfflat_idx"

# m = số cạnh mỗi node, ef_construction = độ rộng hàng đợi lúc BUILD.
# Cả hai đóng băng vào cấu trúc graph khi build — muốn đổi phải DROP + CREATE lại.
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 64

# ef_search = độ rộng hàng đợi lúc TRUY VẤN, đổi được runtime bằng SET.
# Đây là núm xoay recall/tốc độ mà production thực sự dùng. Luật: ef_search >= k.
EF_SEARCH_VALUES = [20, 40, 100]


def count_chunks(conn) -> int:
    """Số dòng trong bảng chunks — cần biết trước khi đo để không tự lừa mình."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks;")
        return cur.fetchone()[0]


def list_indexes(conn) -> list[tuple[str, str]]:
    """Index đang có trên bảng chunks: [(index_name, index_definition), ...].

    Dùng để kiểm chứng opclass: definition phải chứa `vector_cosine_ops`.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'chunks' ORDER BY indexname;"
        )
        return cur.fetchall()


def drop_index(conn, index_name: str) -> None:
    """DROP INDEX IF EXISTS — idempotent, gọi bao nhiêu lần cũng được."""
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {index_name};")
    conn.commit()


def probe_vector() -> list[float]:
    """Vector đơn vị 384 chiều dùng làm đầu dò khi đo TỐC ĐỘ.

    Đo tốc độ thì nội dung vector không quan trọng, nên không cần load model (tiết kiệm
    ~10s mỗi lần chạy). Đo CHẤT LƯỢNG kết quả thì mới cần vector thật — việc đó ở compare_topk.py.
    """
    vector = [0.0] * EMBED_DIM
    vector[0] = 1.0
    return vector


def explain_analyze(conn, sql: str, params: tuple) -> tuple[str, float]:
    """Chạy EXPLAIN (ANALYZE, BUFFERS) cho `sql`, trả (plan_text, execution_time_ms).

    ANALYZE chứ không EXPLAIN trần: EXPLAIN trần chỉ in kế hoạch DỰ ĐOÁN kèm `cost` là
    đơn vị nội bộ, không phải millisecond. ANALYZE thực sự chạy query rồi báo thời gian thật.

    ⚠️ ANALYZE thực thi query. Với SELECT thì vô hại, nhưng EXPLAIN ANALYZE một câu
       DELETE/UPDATE là dữ liệu bay thật. Muốn an toàn: bọc BEGIN ... ROLLBACK.

    Trả -1.0 thay vì raise khi regex không khớp: bản Postgres khác có thể in khác, đừng để
    cả buổi đo chết vì một dòng regex.
    """
    with conn.cursor() as cur:
        # Nối chuỗi ở đây an toàn: "EXPLAIN ..." là hằng do mình viết, GIÁ TRỊ vẫn đi qua params.
        cur.execute("EXPLAIN (ANALYZE, BUFFERS) " + sql, params)
        plan_text = "\n".join(row[0] for row in cur.fetchall())

        match = re.search(r"Execution Time: ([\d.]+) ms", plan_text)
        execution_ms = float(match.group(1)) if match else -1.0

    return (plan_text, execution_ms)


def plan_uses_index(plan_text: str, index_name: str | None = None) -> bool:
    """Plan này có dùng index scan không? Truyền `index_name` thì phải đúng index đó.

    Cần hàm riêng vì đây là bug im lặng kinh điển của pgvector: index tồn tại, query trả
    kết quả ĐÚNG, ở bảng nhỏ không cảm nhận được gì — nhưng Postgres đang Seq Scan.
    Phải kiểm tra bằng PLAN, không bằng cảm giác.

    Ba lý do index bị bỏ qua:
      1. opclass không khớp toán tử (vector_l2_ops + `<=>`) — không cảnh báo
      2. ORDER BY một BIỂU THỨC: `ORDER BY 1 - (embedding <=> %s) DESC`
      3. bảng quá nhỏ — planner tính seq scan rẻ hơn, và nó đúng

    Đừng kiểm tra bằng `"hnsw" in plan_text`: tên index chứa chữ "hnsw" nhưng dòng đó vẫn
    có thể là Seq Scan. Phải tìm cụm đầy đủ "Index Scan using <tên>".
    """
    if index_name is not None:
        return f"Index Scan using {index_name}" in plan_text

    return any(marker in plan_text for marker in ("Index Scan", "Index Only Scan"))


def create_hnsw_index(
    conn,
    m: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
) -> float:
    """Tạo index HNSW trên chunks(embedding) với opclass cosine. Trả thời gian BUILD (giây).

    ⚠️ Opclass PHẢI khớp toán tử query:
           vector_cosine_ops <-> `<=>`   ·   vector_l2_ops <-> `<->`   ·   vector_ip_ops <-> `<#>`
       Tạo bằng vector_l2_ops rồi query bằng `<=>` -> Seq Scan, không một lời cảnh báo.
       Cả pipeline tuần 4-5 dùng `<=>` nên ở đây bắt buộc là vector_cosine_ops.

    Vì sao HNSW mà không IVFFlat: IVFFlat build nhanh và nhẹ RAM hơn, nhưng recall kém hơn
    và **phải có dữ liệu trước khi tạo index** (cần dữ liệu để học cụm k-means) — tạo trên
    bảng rỗng cho cụm rác, recall thảm hại mà không báo lỗi. HNSW dựng graph lớn dần theo
    INSERT nên không dính bẫy đó.

    Ghi chú production: CREATE INDEX khoá bảng cho ghi. Bảng lớn thì dùng CREATE INDEX
    CONCURRENTLY — nhưng nó không chạy được trong transaction nên phải bật autocommit.
    """
    # Ép int vì m/ef_construction đi vào SQL qua f-string: WITH (...) là tham số cấu trúc,
    # placeholder %s không dùng được ở đó.
    m = int(m)
    ef_construction = int(ef_construction)

    drop_index(conn, HNSW_INDEX_NAME)
    started_at = time.perf_counter()

    with conn.cursor() as cur:
        cur.execute(
            f"CREATE INDEX {HNSW_INDEX_NAME} ON chunks "
            f"USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {m}, ef_construction = {ef_construction});"
        )
    conn.commit()

    return time.perf_counter() - started_at


def benchmark_query(
    conn,
    query_vector: list[float],
    k: int = 5,
    ef_search: int | None = None,
) -> dict:
    """Đo ĐÚNG câu SQL của vector_ops.search_top_k trong 1 điều kiện cụ thể.

    Trả {"ef_search", "execution_ms", "uses_index", "plan"} — dict thay vì tuple để chỗ gọi
    đọc theo tên, không phải nhớ thứ tự.

    Phải đo đúng câu SQL ứng dụng thật dùng: đo một câu khác đi một chữ là đo một thứ khác,
    số liệu vô nghĩa.

    ⚠️ `SET hnsw.ef_search` (không LOCAL) chỉ sống trong 1 session. Connection mới là về
       mặc định 40. Và ef_search phải >= k, nếu không HNSW trả về ít hơn k dòng mà không lỗi.
    """
    if ef_search is not None:
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {int(ef_search)};")

    query_literal = to_pgvector(query_vector)
    sql = ("SELECT id, source, content, 1 - (embedding <=> %s) AS similarity "
           "FROM chunks ORDER BY embedding <=> %s LIMIT %s;")
    params = (query_literal, query_literal, k)

    # Lần chạy đầu luôn chậm hơn vì Postgres còn nạp page từ disk vào shared_buffers.
    # Warm-up rồi mới lấy số ở lần thứ hai — thiếu bước này là kết luận sai về tốc độ.
    explain_analyze(conn, sql, params)
    plan_text, execution_ms = explain_analyze(conn, sql, params)

    return {"ef_search": ef_search, "execution_ms": execution_ms,
            "uses_index": plan_uses_index(plan_text, HNSW_INDEX_NAME),
            "plan": plan_text}


# Plan giả để self-check parser chạy được không cần DB — vòng lặp sửa–chạy vài giây.
SAMPLE_SEQ_SCAN_PLAN = """Limit  (cost=112.50..112.51 rows=5 width=48) (actual time=18.402..18.404 rows=5 loops=1)
  ->  Sort  (cost=112.50..115.00 rows=1000 width=48) (actual time=18.401..18.402 rows=5 loops=1)
        Sort Key: ((embedding <=> '[1,0,0]'::vector))
        ->  Seq Scan on chunks  (cost=0.00..95.00 rows=1000 width=48) (actual time=0.021..15.9 rows=1000 loops=1)
Planning Time: 0.180 ms
Execution Time: 18.451 ms"""

SAMPLE_HNSW_PLAN = """Limit  (cost=8.30..12.40 rows=5 width=48) (actual time=1.102..1.140 rows=5 loops=1)
  ->  Index Scan using chunks_embedding_hnsw_idx on chunks  (cost=8.30..40.10 rows=1000 width=48) (actual time=1.100..1.135 rows=5 loops=1)
        Order By: (embedding <=> '[1,0,0]'::vector)
Planning Time: 0.210 ms
Execution Time: 1.183 ms"""


def self_check_plan_parser() -> None:
    """4 assert cho plan_uses_index bằng plan giả — không cần DB."""
    assert plan_uses_index(SAMPLE_SEQ_SCAN_PLAN) is False
    assert plan_uses_index(SAMPLE_HNSW_PLAN) is True
    assert plan_uses_index(SAMPLE_HNSW_PLAN, HNSW_INDEX_NAME) is True
    assert plan_uses_index(SAMPLE_HNSW_PLAN, IVFFLAT_INDEX_NAME) is False
    print("✅ plan_uses_index: 4/4 assert pass (không cần DB)")


def print_benchmark_row(label: str, result: dict) -> None:
    """In 1 dòng kết quả đo, định dạng để copy thẳng vào note."""
    flag = "index" if result["uses_index"] else "SEQ SCAN"
    print(f"  {label:<28} {result['execution_ms']:>8.3f} ms   [{flag}]")


def main() -> None:
    if "--dry" in sys.argv:
        self_check_plan_parser()
        return

    query_vector = probe_vector()

    with get_conn() as conn:
        total = count_chunks(conn)
        print(f"📊 Bảng chunks: {total} dòng")
        if total < 1000:
            print("   ⚠️  Dưới 1000 dòng — planner có quyền chọn Seq Scan dù đã có index.")
            print("      Thấy SEQ SCAN ở quy mô này là BÌNH THƯỜNG, và mọi số ef_search bên")
            print("      dưới là NHIỄU: Seq Scan không dùng tới ef_search.")

        print("\n--- TRƯỚC khi có index ---")
        drop_index(conn, HNSW_INDEX_NAME)
        before = benchmark_query(conn, query_vector, k=5)
        print_benchmark_row("no index", before)

        print("\n--- Tạo index HNSW ---")
        build_seconds = create_hnsw_index(conn)
        print(f"  build xong trong {build_seconds:.2f} s")
        for name, definition in list_indexes(conn):
            print(f"  {name}: {definition}")

        print("\n--- SAU khi có index, đổi ef_search ---")
        for ef in EF_SEARCH_VALUES:
            print_benchmark_row(
                f"hnsw.ef_search = {ef}",
                benchmark_query(conn, query_vector, k=5, ef_search=ef),
            )

        print(f"\n📋 Copy vào note: {total} dòng · before {before['execution_ms']:.3f} ms "
              f"· build {build_seconds:.2f} s")

    # Kỳ vọng khi chạy đúng:
    #   - `--dry` in "4/4 assert pass"
    #   - list_indexes in ra definition CÓ chữ `vector_cosine_ops` (thiếu = sai opclass)
    #   - bảng >= vài chục nghìn dòng: dòng "no index" [SEQ SCAN], các dòng ef_search [index],
    #     ms tăng dần theo ef_search
    #   - bảng nhỏ: tất cả [SEQ SCAN] — ghi số dòng vào note và KHÔNG kết luận gì về ef_search


if __name__ == "__main__":
    main()
