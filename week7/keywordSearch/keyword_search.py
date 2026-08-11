"""
keyword_search.py — Nhánh tìm kiếm THỨ HAI: theo từ khoá, bằng Postgres full-text

    INPUT :  câu hỏi tiếng Việt + k
    OUTPUT:  top-k chunk kèm điểm ts_rank và THỨ HẠNG (1..k)

    câu hỏi --plainto_tsquery('simple')--> tsquery ('mã' & 'lỗi' & 'iso')
                                              |
                        chế độ 'and': giữ nguyên      chế độ 'or': đổi & thành |
                                              |
                              WHERE tsv @@ query   <-- GIN lọc ở đây
                                              |
                              ORDER BY ts_rank(...) DESC LIMIT k

So với tuần 5:
    tuần 5 (askCli/retriever.py):  question --embed--> vector --`<=>` + HNSW--> top-k
    hôm nay:                       question --tsquery--> tsvector --`@@` + GIN--> top-k
                                            ^^^^ KHÔNG gọi model, KHÔNG tốn 470MB RAM,
                                            KHÔNG mất 5 giây khởi động. Đó cũng là một
                                            ưu điểm thật của nhánh keyword, không chỉ là
                                            chuyện chất lượng kết quả.

Điểm PHẢI nhớ cho ngày mai (T3 — RRF):
    similarity của vector nằm trong [-1, 1] và có ý nghĩa tuyệt đối (0.62 là "khá giống").
    ts_rank KHÔNG có thang cố định — nó phụ thuộc số từ khớp, độ dài chunk, cờ chuẩn hoá.
    0.08 có thể là kết quả tốt nhất kho. => KHÔNG BAO GIỜ cộng hai điểm này lại với nhau.
    Đó chính xác là lý do RRF (ngày mai) trộn theo THỨ HẠNG chứ không theo điểm.


Self-test bằng dữ liệu giả — không cần DB, không cần model:
    python keyword_search.py --dry
Chạy thật (cần Postgres Up + đã chạy fulltext_schema.py):
    python keyword_search.py "mã lỗi E402 nghĩa là gì"
    python keyword_search.py "mã lỗi E402 nghĩa là gì" --mode and
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))

from fulltext_schema import GIN_INDEX_NAME, RANK_WEIGHTS, TS_CONFIG, TSV_COLUMN_NAME  # noqa: E402
from vector_ops import get_conn  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_TOP_K = 5

# Chế độ ghép từ khoá. Đây là quyết định thiết kế quan trọng nhất của file này — xem
# giải thích trong build_tsquery_expression.
VALID_MODES = ("or", "and")
DEFAULT_MODE = "or"

# Cờ chuẩn hoá của ts_rank. 32 = rank/(rank+1), ép điểm về khoảng [0, 1).
# KHÔNG phải để so với cosine similarity (xem docstring module) mà để in ra cho dễ đọc
# và để so sánh giữa các truy vấn trong CÙNG nhánh keyword.
RANK_NORMALIZATION = 32

# Token 1 ký tự gần như luôn là nhiễu ('à', 'ở', 'g' còn sót sau khi tách). Giữ lại chữ số
# vì mã lỗi / năm / số hiệu là đúng thứ mà nhánh keyword sinh ra để bắt.
MIN_TOKEN_LENGTH = 2


@dataclass
class KeywordHit:
    """1 chunk khớp từ khoá. Cố ý ĐẶT CÙNG THỨ TỰ TRƯỜNG với RetrievedChunk tuần 5.

    Giống nhau 7 trường đầu, chỉ khác trường cuối: `score` (ts_rank) thay cho `similarity`
    (cosine). Tên khác nhau là CỐ Ý — hai con số này không cùng đơn vị, đặt trùng tên là
    mời gọi chính mình cộng chúng lại vào một ngày mệt mỏi nào đó.

    Giữ hình dạng song song để ngày mai (RRF) trộn hai danh sách mà không phải viết
    adapter — chỉ cần `.id` và vị trí trong list.
    """

    id: int
    source: str
    doc_type: str | None
    title: str | None
    page: int | None
    heading: str | None
    content: str
    score: float

    def location(self) -> str:
        """Địa chỉ người-đọc-được. Cùng luật với RetrievedChunk.location() tuần 5."""
        if self.page is not None:
            return f"trang {self.page}"
        return self.heading or "—"


def tokenize_query(question: str) -> list[str]:
    """Câu hỏi -> list token đã hạ chữ thường, bỏ dấu câu, bỏ token quá ngắn và trùng lặp.

    Ví dụ: 'Mã lỗi E402 là gì?' -> ['mã', 'lỗi', 'e402', 'là', 'gì']
    """

    raw_tokens = re.findall(r"\w+", (question or "").lower())

    tokens = [t for t in raw_tokens if len(t) >= MIN_TOKEN_LENGTH]

    return list(dict.fromkeys(tokens))


def build_tsquery_expression(mode: str = DEFAULT_MODE) -> str:
    """Trả về MẢNH SQL sinh ra tsquery từ một placeholder %s. Không chạm DB.

    Ví dụ: build_tsquery_expression('and') -> "plainto_tsquery('simple', %s)"
           build_tsquery_expression('or')  -> "replace(plainto_tsquery('simple', %s)::text, '&', '|')::tsquery"

    Vì sao 'or' là mặc định: plainto_tsquery nối MỌI token bằng AND, nên câu hỏi tiếng Việt
    6 âm tiết đòi một chunk chứa đủ cả 6 từ -> gần như chắc chắn 0 dòng. Nhánh keyword có
    nhiệm vụ tiến cử ứng viên, không phải phán quyết: trả rộng rồi để ts_rank xếp hạng.

    Vì sao đổi toán tử trên ĐẦU RA của Postgres thay vì tự ghép chuỗi từ tokenize_query():
    Python và Postgres tách token khác nhau ('ISO-9001' -> Postgres cho 3 lexeme, regex \\w+
    cho 2). Token tự ghép không nằm trong index -> 0 kết quả, im lặng, cực khó lần ra.

    Giới hạn: mẹo replace chỉ an toàn với đầu ra plainto_tsquery (luôn là lexeme nối bằng
    ' & '). websearch_to_tsquery có thể sinh phrase operator `<->` — replace mù sẽ phá nó.
    """

    if mode not in VALID_MODES:
        raise ValueError(f"mode phải thuộc {VALID_MODES}, nhận được {mode!r}")

    base = f"plainto_tsquery('{TS_CONFIG}', %s)"

    if mode == "and":
        return base
    return f"replace({base}::text, '&', '|')::tsquery"


def search_by_keyword(
    conn,
    question: str,
    k: int = DEFAULT_TOP_K,
    mode: str = DEFAULT_MODE,
) -> list[KeywordHit]:
    """top-k chunk khớp từ khoá, xếp theo ts_rank giảm dần.

    Ví dụ: search_by_keyword(conn, "mã lỗi E402", k=3) -> [KeywordHit, KeywordHit, KeywordHit]
    """

    tsquery_sql = build_tsquery_expression(mode)

    # tsquery bọc trong subquery `(SELECT ... ) q` chứ KHÔNG đặt thẳng làm FROM item:
    # FROM chỉ nhận tên bảng / lời gọi hàm / subquery, nên biểu thức có cast `::tsquery`
    # ở đó là syntax error (chế độ 'and' không cast nên lọt, 'or' thì chết — cùng một hàm).
    # Mảng RANK_WEIGHTS phải truyền vào ts_rank, nếu không thì setweight('A') cho title
    # bị bỏ qua hoàn toàn: vẫn chạy, vẫn ra kết quả, chỉ là title bị chấm ngang content.
    sql = (
        f"SELECT c.id, c.source, c.doc_type, c.title, c.page, c.heading, c.content, "
        f"       ts_rank('{RANK_WEIGHTS}', c.{TSV_COLUMN_NAME}, query, {RANK_NORMALIZATION}) AS score "
        f"FROM chunks c, (SELECT {tsquery_sql} AS query) q "
        f"WHERE c.{TSV_COLUMN_NAME} @@ query "
        f"ORDER BY score DESC "
        f"LIMIT %s;"
    )

    with conn.cursor() as cur:
        cur.execute(sql, (question, k))
        rows = cur.fetchall()

    return [KeywordHit(*row) for row in rows]


def uses_gin_index(conn, question: str, mode: str = DEFAULT_MODE) -> tuple[bool, str]:
    """Truy vấn keyword có THẬT SỰ đi qua index GIN không? Trả (dùng_index, plan_text).

    Ví dụ: (True, 'Bitmap Index Scan on chunks_tsv_gin_idx ...')

    KHÔNG dùng lại plan_uses_index() của week5/indexTuning/hnsw_index.py: hàm đó tìm chuỗi
    "Index Scan using <tên>" — dạng của B-tree/HNSW. GIN xuất hiện dưới dạng "Bitmap Index
    Scan on <tên>", nên dùng lại sẽ LUÔN trả False và dẫn tới đi sửa index đang chạy tốt.

    Lưu ý khi đọc kết quả: ORDER BY ts_rank không dùng được index. GIN chỉ lọc ra tập khớp,
    Postgres vẫn tính điểm cho cả tập đó rồi sort — truy vấn OR khớp 80% bảng thì gần như
    quét toàn bảng dù plan có Bitmap Index Scan.
    """

    tsquery_sql = build_tsquery_expression(mode)
    sql = (
        f"SELECT c.id FROM chunks c, (SELECT {tsquery_sql} AS query) q "
        f"WHERE c.{TSV_COLUMN_NAME} @@ query "
        f"ORDER BY ts_rank('{RANK_WEIGHTS}', c.{TSV_COLUMN_NAME}, query, "
        f"{RANK_NORMALIZATION}) DESC LIMIT %s;"
    )

    with conn.cursor() as cur:
        cur.execute("EXPLAIN (ANALYZE, BUFFERS) " + sql, (question, DEFAULT_TOP_K))
        plan_text = "\n".join(row[0] for row in cur.fetchall())

    markers = (f"Bitmap Index Scan on {GIN_INDEX_NAME}", f"Index Scan using {GIN_INDEX_NAME}")
    return (any(m in plan_text for m in markers), plan_text)


def format_keyword_hits(hits: list[KeywordHit], k: int) -> str:
    """In bảng xếp hạng keyword. Hình thức cố tình giống format_sources tuần 5."""
    if not hits:
        return ("  (0 kết quả — xem 3 bước chẩn đoán ở docstring của search_by_keyword, "
                "đừng sửa SQL trước)")

    lines = []
    for rank, hit in enumerate(hits[:k], 1):
        label = hit.title or hit.source
        snippet = " ".join(hit.content.split())[:70]
        lines.append(f"  {rank}. id={hit.id:<6} [{hit.score:.6f}] {label} · {hit.location()}")
        lines.append(f"       {snippet}...")
    return "\n".join(lines)


def self_check() -> None:
    """Assert bằng dữ liệu giả — không cần DB, không cần model."""
    tokens = tokenize_query("Mã lỗi E402, mã lỗi E403 nghĩa là gì?")
    assert tokens == ["mã", "lỗi", "e402", "e403", "nghĩa", "là", "gì"], f"sai: {tokens}"
    assert tokenize_query("") == []
    assert "e402" in tokenize_query("E402"), "phải hạ chữ thường"
    assert "a" not in tokenize_query("a bc"), "token 1 ký tự phải bị bỏ"

    and_sql = build_tsquery_expression("and")
    or_sql = build_tsquery_expression("or")
    assert and_sql.count("%s") == 1, f"phải có đúng 1 placeholder: {and_sql}"
    assert or_sql.count("%s") == 1, f"phải có đúng 1 placeholder: {or_sql}"
    assert "'|'" in or_sql and "'&'" in or_sql, "chế độ or phải đổi & thành |"
    assert TS_CONFIG in and_sql, "phải dùng chung hằng TS_CONFIG, không gõ lại chuỗi"

    try:
        build_tsquery_expression("xor")
    except ValueError:
        pass
    else:
        raise AssertionError("mode lạ phải ném ValueError, không được im lặng")

    print("✅ self_check: tokenize_query + build_tsquery_expression pass (không cần DB)")
    print(f"   and -> {and_sql}")
    print(f"   or  -> {or_sql}")


def main() -> None:
    if "--dry" in sys.argv:
        self_check()
        return

    mode = DEFAULT_MODE
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]

    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if mode in positional:
        positional.remove(mode)
    question = " ".join(positional)
    if not question:
        raise SystemExit('Dùng: python keyword_search.py "câu hỏi" [--mode or|and] | --dry')

    print(f"❓ {question}   (mode={mode})\n")

    with get_conn() as conn:
        hits = search_by_keyword(conn, question, k=DEFAULT_TOP_K, mode=mode)
        print(format_keyword_hits(hits, DEFAULT_TOP_K))

        used_index, plan_text = uses_gin_index(conn, question, mode=mode)
        print(f"\n🔎 GIN index: {'CÓ dùng' if used_index else 'KHÔNG dùng (Seq Scan)'}")
        if "--plan" in sys.argv:
            print(plan_text)

    # ✅ ĐẠT khi:
    #   1. `--dry` in "pass" và in ra 2 mảnh SQL, mỗi mảnh đúng 1 dấu %s
    #   2. Cùng một câu hỏi: `--mode or` trả về NHIỀU kết quả hơn (hoặc bằng) `--mode and`.
    #      Ngược lại là logic đổi toán tử đang sai
    #   3. Với câu hỏi chứa mã/số hiệu có thật trong kho, chunk chứa mã đó đứng hạng 1
    #   4. Điểm ts_rank nằm trong [0, 1) nhờ cờ chuẩn hoá 32, và GIẢM dần từ trên xuống
    #   5. Ghi vào note: điểm ts_rank cao nhất là bao nhiêu. Nó sẽ nhỏ hơn nhiều so với
    #      similarity 0.6x của vector — chính là bằng chứng cho câu "không được cộng hai
    #      điểm này", thứ mình cần cho RRF ngày mai

if __name__ == "__main__":
    main()
