"""
filter_metadata.py — Lọc theo metadata + thử k=3 vs k=5 (T3 tuần 5)

    INPUT :  1 câu hỏi + doc_type muốn lọc ('pdf' | 'docx' | 'md' | 'txt' | None)
    OUTPUT:  top-k đã lọc + plan cho biết index HNSW còn được dùng hay không

Đây là chỗ dùng tới 5 cột metadata đã thêm ở T2 (week5/ingestDocs/db_docs.py). Thêm cột xong
mà không query theo nó thì công migration là vô nghĩa.

Câu hỏi trung tâm — pre-filter vs post-filter:

    SELECT ... WHERE doc_type = 'pdf' ORDER BY embedding <=> %s LIMIT 5;

    (a) POST-filter : đi HNSW lấy ~ef_search ứng viên gần nhất -> lọc doc_type sau
        -> nhanh, nhưng nếu trong số ứng viên đó chỉ có 2 cái là pdf thì trả về 2 dòng
           thay vì 5. THIẾU KẾT QUẢ MÀ KHÔNG BÁO LỖI.
        -> dấu hiệu trong plan: "Index Scan ..." + "Filter:" (+ "Rows Removed by Filter: N")
    (b) PRE-filter  : lọc doc_type trước -> quét tuần tự phần còn lại
        -> đúng đủ k dòng, nhưng bỏ index -> chậm khi bảng lớn
        -> dấu hiệu trong plan: "Seq Scan" + "Filter:"

Hôm nay chỉ QUAN SÁT Postgres chọn đường nào. Cách xử lý thật (partial index theo doc_type,
tăng ef_search, hoặc lấy k lớn rồi lọc ở tầng ứng dụng) để tuần 6 khi làm hybrid retrieval.

⚠️ Planner đổi lựa chọn theo PHÂN BỐ DỮ LIỆU: doc_type chiếm 90% bảng thì nó chọn khác với
   doc_type chiếm 1%. Vì vậy số liệu loại này bắt buộc ghi kèm phân bố doc_type vào note.

Chạy (cần DB + đã chạy hnsw_index.py để có index):
    python filter_metadata.py "embedding là gì?"
    python filter_metadata.py "embedding là gì?" --doc-type pdf
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "ingestDocs"))

from db_docs import stats_by_doc_type  # noqa: E402
from hnsw_index import HNSW_INDEX_NAME, explain_analyze, plan_uses_index  # noqa: E402
from vector_ops import get_conn, to_pgvector  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
K_VALUES = [3, 5]

_model = None


def get_model():
    """Lazy load — import cục bộ để không nạp torch khi chỉ đọc module."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def print_doc_types(conn) -> None:
    """Dùng lại stats_by_doc_type của T2 — phân bố doc_type là điều kiện đo, phải ghi kèm."""
    print("📊 Phân bố chunk theo doc_type:")
    for doc_type, n_chunks, n_files in stats_by_doc_type(conn):
        print(f"   {str(doc_type):<8} {n_chunks:>6} chunk / {n_files} file")


def search_filtered(
    conn,
    query_vector: list[float],
    doc_type: str | None = None,
    k: int = 5,
) -> list[tuple[int, str, str, str, int | None, float]]:
    """top-k có (hoặc không) lọc doc_type -> [(id, source, doc_type, content, page, similarity)].

    Dựng 1 câu SQL + 1 mảnh WHERE tuỳ chọn thay vì 2 câu riêng: 2 câu riêng sẽ phân kỳ dần,
    sửa cột ở câu này quên câu kia. Chỉ nối phần CẤU TRÚC, GIÁ TRỊ luôn đi qua params.

    ⚠️ Thứ tự params phải khớp thứ tự dấu %s XUẤT HIỆN TRONG CÂU, không phải thứ tự mình nghĩ:
           SELECT ... 1 - (embedding <=> %s)   -> (1) vector
           WHERE doc_type = %s                 -> (2) doc_type   (chèn Ở GIỮA!)
           ORDER BY embedding <=> %s LIMIT %s  -> (3) vector, (4) k
       Đảo (1) và (2): nếu 2 tham số cùng là text thì CHẠY BÌNH THƯỜNG và ra kết quả rác.
       Vì vậy append params ĐÚNG lúc lắp mảnh SQL tương ứng, đừng viết trước rồi sửa.
    """
    query_literal = to_pgvector(query_vector)
    sql = ("SELECT id, source, doc_type, content, page, "
           "1 - (embedding <=> %s) AS similarity FROM chunks")
    params = [query_literal]

    if doc_type:
        sql += " WHERE doc_type = %s"
        params.append(doc_type)

    sql += " ORDER BY embedding <=> %s LIMIT %s;"
    params += [query_literal, k]

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return cur.fetchall()


def inspect_filter_plan(conn, query_vector: list[float], doc_type: str, k: int = 5) -> dict:
    """EXPLAIN ANALYZE cho câu CÓ WHERE và câu KHÔNG WHERE, trả 2 bên để so.

    Mỗi bên: execution_ms · uses_index · filter_position ('pre' | 'post' | 'unknown') · plan.

    Phải so với câu KHÔNG WHERE vì con số ms một mình không nói gì: "có WHERE mất 12 ms"
    vô nghĩa cho tới khi biết "không WHERE mất 1.2 ms" -> chậm 10 lần vì thêm một điều kiện
    tưởng như làm việc ÍT hơn. Đó mới là phát hiện.
    """
    def classify(plan_text: str) -> str:
        if plan_uses_index(plan_text, HNSW_INDEX_NAME) and "Filter:" in plan_text:
            return "post"
        if "Seq Scan" in plan_text and "Filter:" in plan_text:
            return "pre"
        return "unknown"

    query_literal = to_pgvector(query_vector)

    plain_sql = "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s;"
    plain_plan, plain_ms = explain_analyze(conn, plain_sql, (query_literal, k))

    filtered_sql = ("SELECT id FROM chunks WHERE doc_type = %s "
                    "ORDER BY embedding <=> %s LIMIT %s;")
    filtered_plan, filtered_ms = explain_analyze(conn, filtered_sql, (doc_type, query_literal, k))

    return {
        "without_filter": {"execution_ms": plain_ms,
                           "uses_index": plan_uses_index(plain_plan, HNSW_INDEX_NAME),
                           "filter_position": classify(plain_plan),
                           "plan": plain_plan},
        "with_filter": {"execution_ms": filtered_ms,
                        "uses_index": plan_uses_index(filtered_plan, HNSW_INDEX_NAME),
                        "filter_position": classify(filtered_plan),
                        "plan": filtered_plan},
    }


def compare_k_values(
    conn,
    query_vector: list[float],
    doc_type: str | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """Chạy cùng câu hỏi với k=3 và k=5, trả số liệu để chấm "phủ rộng vs loãng".

    {3: {rows, n_sources, avg_similarity, min_similarity}, 5: {...},
     "new_sources": [...], "tail_gap": 0.13}

    Hai khái niệm ĐỘC LẬP, không được kết luận một chiều:
      · PHỦ RỘNG = new_sources không rỗng — k=5 mang về source MỚI, prompt có thêm căn cứ
      · LOÃNG    = tail_gap lớn — 2 chunk thêm vào tụt điểm mạnh, chỉ ăn token và làm LLM
                   phân tán (lost-in-the-middle)
    Có thể vừa phủ rộng vừa loãng. new_sources rỗng + tail_gap lớn -> giữ k=3.

    tail_gap luôn >= 0 (thêm chunk thì điểm thấp nhất chỉ giảm hoặc bằng). Ra số ÂM là đang
    sort sai chiều ở đâu đó.

    ⚠️ Chấm trên 1 câu hỏi là chấm trên nhiễu — chạy ít nhất 3 câu trong questions.json.
    """
    k_values = k_values or K_VALUES
    report = {}

    for k in k_values:
        rows = search_filtered(conn, query_vector, doc_type, k=k)
        scores = [row[-1] for row in rows]      # similarity là cột cuối
        report[k] = {
            "rows": rows,
            "n_sources": len({row[1] for row in rows}),
            "avg_similarity": sum(scores) / len(scores) if scores else 0.0,
            "min_similarity": min(scores) if scores else 0.0,
        }

    smallest_k, largest_k = min(k_values), max(k_values)
    sources_small = {row[1] for row in report[smallest_k]["rows"]}
    sources_large = {row[1] for row in report[largest_k]["rows"]}
    report["new_sources"] = sorted(sources_large - sources_small)
    report["tail_gap"] = (report[smallest_k]["min_similarity"]
                          - report[largest_k]["min_similarity"])
    return report


def print_search_results(label: str, rows: list[tuple]) -> None:
    print(f"\n  --- {label} ({len(rows)} dòng) ---")
    for rank, (row_id, source, doc_type, content, page, similarity) in enumerate(rows, 1):
        where = f"trang {page}" if page is not None else "—"   # page NULL với docx/md/txt
        snippet = " ".join(content.split())[:80]
        print(f"  {rank}. [{similarity:.4f}] {doc_type}/{source} ({where})")
        print(f"     {snippet}...")


def parse_doc_type_arg() -> str | None:
    """Đọc `--doc-type <value>` từ argv. Không có thì trả None (không lọc)."""
    if "--doc-type" in sys.argv:
        index = sys.argv.index("--doc-type")
        if index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return None


def main() -> None:
    positional_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    doc_type = parse_doc_type_arg()
    # Giá trị của --doc-type cũng lọt vào positional_args (nó không bắt đầu bằng "--"), phải bỏ ra.
    if doc_type and doc_type in positional_args:
        positional_args.remove(doc_type)
    if not positional_args:
        raise SystemExit('Dùng: python filter_metadata.py "câu hỏi" [--doc-type pdf]')

    question = " ".join(positional_args)
    print(f"❓ {question}   (doc_type = {doc_type or 'KHÔNG lọc'})")
    query_vector = get_model().encode(question).tolist()

    with get_conn() as conn:
        print_doc_types(conn)

        report = compare_k_values(conn, query_vector, doc_type=doc_type)
        for k in K_VALUES:
            print_search_results(f"k = {k}", report[k]["rows"])
            print(f"     n_sources={report[k]['n_sources']} "
                  f"avg={report[k]['avg_similarity']:.4f} min={report[k]['min_similarity']:.4f}")
        print(f"\n  source MỚI khi tăng k: {report['new_sources'] or '(không có)'}")
        print(f"  tail_gap (điểm tụt ở đuôi): {report['tail_gap']:.4f}")
        print("  -> new_sources rỗng + tail_gap lớn = k=5 chỉ làm LOÃNG, nên giữ k=3.")

        if doc_type:
            plans = inspect_filter_plan(conn, query_vector, doc_type)
            print("\n📐 Plan — index còn được dùng khi có WHERE?")
            for label in ("without_filter", "with_filter"):
                info = plans[label]
                print(f"  {label:<16} {info['execution_ms']:>8.3f} ms  "
                      f"index={info['uses_index']}  filter={info['filter_position']}")
            print("\n  (plan đầy đủ — tìm dòng 'Rows Removed by Filter':)")
            print(plans["with_filter"]["plan"])

    # Kỳ vọng khi chạy đúng:
    #   - có --doc-type: MỌI dòng trả về đều đúng doc_type đó
    #   - số dòng trả về < k khi lọc = hiện tượng post-filter, ghi ngay vào note
    #   - tail_gap >= 0 (âm là sort sai chiều)
    #   - note phải ghi kèm phân bố doc_type, thiếu là số liệu vô dụng sau 2 tuần


if __name__ == "__main__":
    main()
