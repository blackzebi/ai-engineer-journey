"""
compare_topk.py — top-k của pgvector vs cosine tự cài bằng Python (nợ T4 tuần 4)

    INPUT :  1 câu hỏi + bảng `chunks` đã có dữ liệu
    OUTPUT:  2 bảng xếp hạng cạnh nhau + overlap@k + chênh lệch điểm

Vì sao món này còn nợ từ tuần 4:
    week4/ragPipeline/search_pg.py :: compare_with_python() so
        pgvector (kho notes .md)  vs  semantic_search.py (corpus.txt ~20 dòng)
    -> HAI KHO KHÁC NHAU, so xong không kết luận được gì.

Hôm nay làm đúng — CÙNG 1 KHO, cùng 1 vector câu hỏi:

    câu hỏi --embed--> query_vector
                          |
        +-----------------+------------------+
        |                                    |
    pgvector: ORDER BY <=> LIMIT k     Python: SELECT hết embedding về,
    (đi qua index nếu planner chọn)           tự tính cosine, sort, lấy k
        |                                    |
        +--------> so id, so thứ hạng <------+

Nếu 2 bên khác nhau, chỉ có 3 nguyên nhân — phải chỉ ra được là cái nào:
    1. ANN bỏ sót láng giềng thật (recall < 100%)
    2. cài sai công thức cosine (quên chia norm -> thành dot product)
    3. sai số float4 (Postgres) vs float64 (Python) — chỉ lệch ở chữ số thập phân thứ 7-8,
       KHÔNG đủ đổi thứ hạng. Thứ hạng khác nhiều thì đừng blame float.

⚠️ overlap@k = k/k chỉ có ý nghĩa khi ĐÃ XÁC NHẬN query đi qua index (`plan_uses_index`).
   Nếu planner chọn Seq Scan thì pgvector cũng đang chạy exact — so exact với exact thì
   trùng khít là điều bắt buộc xảy ra, không chứng minh gì về recall của ANN.

Chạy:
    python compare_topk.py --dry                        # self-check, KHÔNG cần DB/model
    python compare_topk.py "embedding là gì?"
"""

from __future__ import annotations

import math
import os
import sys
import time

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))

from vector_ops import EMBED_DIM, get_conn, search_top_k  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

# Kéo hết embedding về Python là O(n) băng thông. Ở kho vài nghìn chunk là chấp nhận được
# để đối chiếu. Ở 1 triệu chunk thì tuyệt đối không — bên Python sẽ không chạy nổi.
FETCH_LIMIT = 5000

_model = None


def get_model():
    """Lazy load, cache lại. Import cục bộ trong hàm để `--dry` chạy được mà không cần torch."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """cos(a, b) = dot / (norm_a * norm_b), trong [-1, 1]. Guard chia 0.

    Công thức copy từ week4/semanticSearch/semantic_search.py thay vì import, vì file đó
    khởi tạo `_model = SentenceTransformer(...)` ở MODULE LEVEL — chỉ cần import là nạp
    470MB chỉ để tính một phép nhân.
    (Bài học thiết kế: khởi tạo nặng ở module level làm module không import lại được.)
    """
    dot = sum(x * y for x, y in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(x ** 2 for x in vector_a))
    norm_b = math.sqrt(sum(y ** 2 for y in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def parse_pgvector(literal: str) -> list[float]:
    """Đổi literal pgvector "[0.1,-0.25,0.5]" -> [0.1, -0.25, 0.5]. Hàm ngược của to_pgvector().

    ⚠️ Bẫy im lặng: psycopg KHÔNG biết type `vector` (do extension thêm vào), nên SELECT
       cột embedding trả về một CHUỖI, không phải list. Đưa thẳng chuỗi vào cosine_similarity
       thì `zip(str, list)` vẫn chạy — nó zip từng KÝ TỰ với từng số — ra TypeError, hoặc
       tệ hơn là số vô nghĩa mà không lỗi.

    Không dùng eval() (chạy code lạ từ DB) cũng không dùng json.loads() (chết khi gặp 'NaN').
    Cách gọn hơn cho sau này: `pip install pgvector` rồi register_vector(conn) -> psycopg tự
    trả numpy array.
    """
    if isinstance(literal, (list, tuple)):
        return [float(x) for x in literal]

    body = str(literal).strip().strip("[]")
    if not body:
        return []

    return [float(part) for part in body.split(",")]


def fetch_all_rows(conn, limit: int = FETCH_LIMIT) -> list[tuple[int, str, str, list[float]]]:
    """[(id, source, content, embedding_as_list), ...] để tính cosine bên Python.

    Cố ý KHÔNG ORDER BY: đây là bên "exact", mục đích là lấy toàn bộ ứng viên rồi TỰ xếp
    hạng. Nhờ Postgres sắp trước là đã dùng chính thứ đang muốn kiểm chứng.

    WHERE embedding IS NOT NULL: bảng có thể còn dòng cũ chưa embed — cosine với None sẽ
    nổ ở một hàm khác, xa chỗ gây lỗi.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, source, content, embedding FROM chunks "
            "WHERE embedding IS NOT NULL LIMIT %s;",
            (limit,),          # dấu phẩy bắt buộc: (limit) chỉ là số trong ngoặc, không phải tuple
        )
        raw_rows = cur.fetchall()

        rows = [(row_id, source, content, parse_pgvector(embedding))
                for row_id, source, content, embedding in raw_rows]

        # Chạm limit = có thể còn dòng nữa -> bên exact thiếu ứng viên -> so sánh lệch oan.
        if len(rows) == limit:
            print(f"⚠️  Lấy đúng {limit} dòng = có thể còn nữa -> so sánh có thể lệch oan.")

        return rows


def topk_python(
    query_vector: list[float],
    rows: list[tuple[int, str, str, list[float]]],
    k: int = TOP_K,
) -> list[tuple[int, str, str, float]]:
    """Xếp hạng EXACT bằng cosine tự cài: [(id, source, content, similarity), ...] giảm dần.

    Đây là "chân lý" để đối chiếu: quét 100% ứng viên nên recall = 100% theo đúng định nghĩa.
    Khi 2 bên lệch, giả định mặc định là ANN bỏ sót, không phải "Python sai".

    ⚠️ Hai bên sắp NGƯỢC chiều nhau — chỗ nhầm nhiều nhất:
           Python : cosine SIMILARITY, lớn = giống  -> sorted(reverse=True)
           pgvector `<=>`: cosine DISTANCE, nhỏ = giống -> ORDER BY tăng dần
           quy đổi: similarity = 1 - distance
       Sort sai chiều thì ra top-k GHÉT NHẤT, và code chạy trơn tru.

    Dùng sorted() chứ không .sort(): .sort() sửa tại chỗ list của người gọi.
    """
    scored = [
        (row_id, source, content, cosine_similarity(query_vector, embedding))
        for row_id, source, content, embedding in rows
    ]

    scored = sorted(scored, key=lambda item: item[3], reverse=True)
    return scored[:k]


def compare_rankings(
    python_rows: list[tuple[int, str, str, float]],
    pgvector_rows: list[tuple[int, str, str, float]],
) -> dict:
    """So 2 top-k: {overlap, missing_ids, extra_ids, same_order, max_score_gap}.

    So theo `id` chứ không theo `content`: content có thể trùng nhau giữa 2 chunk khác nhau
    (phần overlap của cửa sổ trượt), và so chuỗi dài thì chậm.

    Hai mức "giống nhau" phải phân biệt khi ghi note:
      · overlap@k  = cùng TẬP id, bỏ qua thứ tự -> thước đo recall của ANN
      · same_order = cùng THỨ TỰ -> lệch thứ tự trong khi cùng tập thường chỉ là sai số float

    max_score_gap > ~0.01 ở cùng vị trí nghĩa là 2 bên đang xếp 2 chunk KHÁC nhau, không
    phải sai số float4 vs float64 (loại đó chỉ lệch cỡ 1e-8).
    """
    python_ids = [row[0] for row in python_rows]
    pgvector_ids = [row[0] for row in pgvector_rows]
    python_set, pgvector_set = set(python_ids), set(pgvector_ids)

    overlap = len(python_set & pgvector_set)
    missing_ids = sorted(python_set - pgvector_set)     # exact có, pgvector bỏ sót
    extra_ids = sorted(pgvector_set - python_set)       # pgvector có, exact không xếp vào top-k
    same_order = python_ids == pgvector_ids

    gaps = [abs(a[3] - b[3]) for a, b in zip(python_rows, pgvector_rows)]
    max_score_gap = max(gaps) if gaps else 0.0

    return dict(
        overlap=overlap,
        missing_ids=missing_ids,
        extra_ids=extra_ids,
        same_order=same_order,
        max_score_gap=max_score_gap,
    )


def print_ranking(title: str, rows: list[tuple], elapsed_seconds: float) -> None:
    print(f"\n=== {title} ({elapsed_seconds * 1000:.1f} ms) ===")
    for rank, (row_id, source, content, score) in enumerate(rows, 1):
        snippet = " ".join(content.split())[:70]     # bóp \n từ PDF về 1 dấu cách, kẻo vỡ bảng
        print(f"{rank}. id={row_id:<6} [{score:.6f}] {source}\n     {snippet}...")


def print_verdict(report: dict, k: int) -> None:
    print("\n=== KẾT LUẬN ===")
    print(f"  overlap@{k}      : {report['overlap']}/{k}")
    print(f"  same_order      : {report['same_order']}")
    print(f"  max_score_gap   : {report['max_score_gap']:.8f}")
    print(f"  exact có, pgvector bỏ sót : {report['missing_ids']}")
    print(f"  pgvector có, exact không  : {report['extra_ids']}")
    if report["overlap"] == k and report["same_order"]:
        print("  -> 2 bên khớp hoàn toàn.")
        print("     ⚠️ Chỉ được kết luận 'ANN không mất recall' NẾU plan cho thấy Index Scan.")
        print("        Nếu là Seq Scan thì cả 2 bên đều exact, trùng khít là hiển nhiên.")
    elif report["overlap"] == k:
        print("  -> cùng tập, khác thứ tự: gần như chắc là sai số float4 (DB) vs float64 (Python).")
    else:
        print("  -> khác TẬP. Loại trừ theo thứ tự: (a) cùng query_vector? (b) FETCH_LIMIT có")
        print("     cắt mất ứng viên? (c) index đúng opclass? Cả 3 ổn thì mới là ANN recall.")


def self_check() -> None:
    """Assert bằng dữ liệu giả — không cần DB, không cần model."""
    assert parse_pgvector("[0.5,-0.25,1]") == [0.5, -0.25, 1.0]
    assert parse_pgvector("[]") == []
    assert len(parse_pgvector("[" + ",".join(["0.1"] * EMBED_DIM) + "]")) == EMBED_DIM

    query_vector = [1.0, 0.0, 0.0]
    rows = [
        (1, "a.md", "chunk gan nhat", [1.0, 0.0, 0.0]),
        (2, "b.md", "chunk vuong goc", [0.0, 1.0, 0.0]),
        (3, "c.md", "chunk lech mot chut", [0.9, 0.1, 0.0]),
    ]
    ranked = topk_python(query_vector, rows, k=2)
    assert [row[0] for row in ranked] == [1, 3], f"thứ tự sai: {[r[0] for r in ranked]}"
    assert abs(ranked[0][3] - 1.0) < 1e-6

    # So một kết quả với chính nó — cách test rẻ nhất cho hàm so sánh.
    same = compare_rankings(ranked, ranked)
    assert same["overlap"] == 2 and same["same_order"] is True and same["max_score_gap"] == 0.0
    print("✅ self_check: parse_pgvector + topk_python + compare_rankings pass (không cần DB)")


def main() -> None:
    if "--dry" in sys.argv:
        self_check()
        return

    positional_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if not positional_args:
        raise SystemExit('Dùng: python compare_topk.py "câu hỏi" | python compare_topk.py --dry')
    question = " ".join(positional_args)
    print(f"❓ {question}")

    query_vector = get_model().encode(question).tolist()

    with get_conn() as conn:
        started_at = time.perf_counter()
        pgvector_rows = search_top_k(conn, query_vector, k=TOP_K)   # dùng lại nguyên si tuần 4
        pgvector_seconds = time.perf_counter() - started_at

        started_at = time.perf_counter()
        all_rows = fetch_all_rows(conn)
        python_rows = topk_python(query_vector, all_rows, k=TOP_K)
        python_seconds = time.perf_counter() - started_at

    print_ranking("pgvector (ORDER BY embedding <=> %s)", pgvector_rows, pgvector_seconds)
    print_ranking(f"Python exact (cosine tay trên {len(all_rows)} chunk)", python_rows, python_seconds)
    print_verdict(compare_rankings(python_rows, pgvector_rows), TOP_K)
    print(f"\n⏱  pgvector {pgvector_seconds * 1000:.1f} ms  vs  Python {python_seconds * 1000:.1f} ms")

    # Kỳ vọng khi chạy đúng:
    #   - `--dry` in "self_check ... pass"
    #   - similarity 2 bên đều nằm trong [-1, 1] và GIẢM dần từ trên xuống
    #   - max_score_gap cỡ 1e-8 (sai số float4 vs float64), không phải 1e-2
    #   - chi phí bên Python phần lớn là PARSE CHUỖI (n_chunk x 384 lần gọi float()),
    #     không phải phép nhân — nên ở kho lớn nó thua pgvector rất xa


if __name__ == "__main__":
    main()
