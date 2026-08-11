"""
fulltext_schema.py — Thêm cột tsvector + index GIN vào bảng chunks, và đo IDF thật

    INPUT :  bảng `chunks` đã có dữ liệu (từ week5/ingestDocs/ingest_docs.py)
    OUTPUT:  cột `tsv` tự sinh + index GIN + bảng thống kê token (ndoc, IDF)

    [chunks: id, source, content, embedding, title, page, heading, chunk_index]
              |
              +-- ALTER TABLE ADD COLUMN tsv tsvector GENERATED ALWAYS AS (...) STORED
              |        (title -> hạng A, content -> hạng B, bằng setweight)
              |
              +-- CREATE INDEX ... USING gin (tsv)
              |
    [chunks có thêm nhánh tìm kiếm THỨ HAI]  --ts_stat-->  token nào xuất hiện ở bao
                                                            nhiêu chunk -> IDF

So với tuần 5:
    tuần 5:  chunks.embedding  --HNSW (vector_cosine_ops)-->  tìm theo NGHĨA
    hôm nay: chunks.tsv        --GIN  (tsvector_ops)      -->  tìm theo TỪ KHOÁ
                    ^^^^ cùng một bảng, cùng một dòng dữ liệu, hai cách đánh chỉ mục.
    Ngày mai (T3) sẽ trộn hai danh sách này lại bằng RRF.

⚠️ Postgres full-text KHÔNG PHẢI BM25. `ts_rank` chỉ nhìn tần suất từ + trọng số hạng
   TRONG một document; nó không hề biết một từ hiếm hay phổ biến trong TOÀN KHO — tức là
   KHÔNG có thành phần IDF. Đó là lý do file này có hàm `idf()` tự tính: để tự tay nhìn
   thấy thứ mà ts_rank đang bỏ qua. Tên file kế hoạch ghi "BM25" là nói về khái niệm,
   không phải nói Postgres cài BM25.


Self-test bằng số giả — không cần DB, không cần model:
    python fulltext_schema.py --dry
Chạy thật (container rag-pg phải Up):
    python fulltext_schema.py
    python fulltext_schema.py --terms "mã lỗi ISO 9001 là gì"
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))

from vector_ops import get_conn  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Tên đặt cứng để drop/create lại nhiều lần khi thử — giống HNSW_INDEX_NAME tuần 5.
TSV_COLUMN_NAME = "tsv"
GIN_INDEX_NAME = "chunks_tsv_gin_idx"

# Cấu hình từ điển full-text. 'simple' = KHÔNG stem, KHÔNG bỏ stopword, chỉ hạ chữ thường
# và cắt token. Bắt buộc dùng 'simple' cho tiếng Việt:
#   to_tsvector('english', 'các hoạt động') -> stemmer tiếng Anh cắt bừa đuôi từ tiếng Việt
# Cái giá của 'simple': tiếng Việt là ngôn ngữ ĐA ÂM TIẾT nhưng parser tách theo khoảng
# trắng, nên "công nghệ thông tin" thành 4 token rời 'công','nghệ','thông','tin'.
# -> tìm "thông tin" cũng khớp chunk chỉ có chữ "tin". Giới hạn này PHẢI ghi vào README.
TS_CONFIG = "simple"

# Trọng số ts_rank theo hạng, thứ tự mảng là {D, C, B, A} — dễ nhớ nhầm vì ngược bảng chữ cái.
# A = title (khớp ở tiêu đề đáng giá hơn), B = content.
RANK_WEIGHTS = "{0.1, 0.2, 0.4, 1.0}"


def migrate_tsv_column(conn) -> None:
    """Thêm cột `tsv` tự sinh từ title+content, rồi tạo index GIN. Idempotent.

    Ví dụ: chạy 2 lần liên tiếp -> lần 2 không lỗi, không đổi gì.

    Vì sao GENERATED ALWAYS AS ... STORED thay vì cột thường + UPDATE một lần: cột thường
    sẽ CŨ ÂM THẦM khi content đổi về sau, và kết quả tìm kiếm sai mà không ai báo. Cột
    generated được Postgres tính lại tự động -> không có chỗ nào để quên đồng bộ.

    coalesce(x, '') là BẮT BUỘC: to_tsvector(config, NULL) trả NULL, và NULL || bất kỳ đều
    ra NULL -> cả cột tsv của dòng đó thành NULL -> dòng biến mất khỏi mọi kết quả tìm kiếm,
    không lỗi, không cảnh báo.

    ⚠️ ALTER TABLE thêm cột STORED phải VIẾT LẠI TOÀN BỘ BẢNG và giữ khoá — trên bảng lớn
    ở production thì đây là thao tác gây downtime, không phải migration vô hại.
    """

    # cur.execute và conn.commit() PHẢI nằm trong cùng khối `with conn.cursor()`:
    # cursor đóng khi thoát khối, và nếu commit rơi ra ngoài thì ALTER TABLE bị rollback
    # âm thầm — triệu chứng chỉ hiện ra ở file khác dưới dạng "column c.tsv does not exist".
    with conn.cursor() as cur:
        cur.execute(
            f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {TSV_COLUMN_NAME} tsvector "
            f"GENERATED ALWAYS AS ("
            f"  setweight(to_tsvector('{TS_CONFIG}', coalesce(title, '')), 'A') || "
            f"  setweight(to_tsvector('{TS_CONFIG}', coalesce(content, '')), 'B')"
            f") STORED;"
        )

        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {GIN_INDEX_NAME} "
            f"ON chunks USING gin ({TSV_COLUMN_NAME});"
        )

    conn.commit()
    print(f"✅ Cột {TSV_COLUMN_NAME} + index {GIN_INDEX_NAME} sẵn sàng")


def top_tokens(conn, limit: int = 20) -> list[tuple[str, int, int]]:
    """Token phổ biến nhất trong kho: [(word, document_frequency, total_occurrences), ...].

    Ví dụ: [('của', 480, 1320), ('là', 455, 900), ('rag', 12, 41), ...]
    """

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT word, ndoc, nentry "
            f"FROM ts_stat('SELECT {TSV_COLUMN_NAME} FROM chunks') "
            f"ORDER BY ndoc DESC LIMIT %s;",
            (limit,),
        )
        return cur.fetchall()


def idf(document_frequency: int, total_documents: int) -> float:
    """IDF theo công thức BM25. df càng lớn -> giá trị càng thấp (có thể âm).

    Ví dụ: idf(1, 1000)   -> ~6.9   (từ cực hiếm, gần như chỉ điểm đích danh 1 chunk)
           idf(500, 1000) -> ~0.69  (nửa kho có từ này, gần như vô giá trị)
           idf(1000, 1000)-> ~0.0   (mọi chunk đều có -> không phân biệt được gì)

    Vì sao file này tự cài IDF trong khi Postgres đã có ts_rank: ts_rank KHÔNG PHẢI BM25.
    Nó chỉ nhìn tần suất từ + trọng số hạng TRONG một document, hoàn toàn không biết một từ
    hiếm hay phổ biến trong toàn kho. Hàm này để nhìn thấy bằng số thứ mà ts_rank bỏ qua.

    Hai hằng trong công thức: `+ 0.5` là smoothing (tránh chia 0 khi df = 0); `+ 1` bên
    trong log giữ IDF không bao giờ âm — nếu không, từ xuất hiện ở >50% kho sẽ có IDF âm và
    kéo TỤT điểm của chunk chứa nó, hành vi rất khó giải thích.
    """

    if total_documents <= 0:
        return 0.0

    numerator = total_documents - document_frequency + 0.5
    denominator = document_frequency + 0.5
    return math.log(numerator / denominator + 1)


def count_chunks(conn) -> int:
    """Tổng số chunk = N trong công thức IDF."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks;")
        return cur.fetchone()[0]


def describe_query_terms(conn, question: str) -> list[tuple[str, int, float]]:
    """Mỗi token trong câu hỏi -> (token, df, idf).

    Đây là công cụ chẩn đoán quan trọng nhất của ngày hôm nay: khi keyword search trả về
    rác, hàm này chỉ đúng thủ phạm — thường là 1-2 hư từ có df gần bằng N kéo cả kết quả
    đi lạc, hoặc token quan trọng nhất có df = 0 (viết sai chính tả / kho không có).

    Import cục bộ để tránh vòng lặp import: keyword_search cũng cần hằng của file này.
    """
    from keyword_search import tokenize_query

    total_documents = count_chunks(conn)
    tokens = tokenize_query(question)
    if not tokens:
        return []

    with conn.cursor() as cur:
        # ANY(%s) nhận một LIST Python và so với từng phần tử — gọn hơn tự ghép chuỗi
        # "IN (%s, %s, %s)" theo số lượng token, và không hở SQL injection.
        cur.execute(
            f"SELECT word, ndoc FROM ts_stat('SELECT {TSV_COLUMN_NAME} FROM chunks') "
            f"WHERE word = ANY(%s);",
            (tokens,),
        )
        frequency_by_word = dict(cur.fetchall())

    # Token không có trong kho -> df = 0 (không phải bỏ qua): df = 0 là THÔNG TIN, nó nói
    # "từ khoá này vô dụng vì kho không chứa nó", khác hẳn với "từ này quá phổ biến".
    return [
        (token, frequency_by_word.get(token, 0), idf(frequency_by_word.get(token, 0), total_documents))
        for token in tokens
    ]


def list_indexes(conn) -> list[tuple[str, str]]:
    """Index đang có trên bảng chunks. Copy từ week5/indexTuning/hnsw_index.py.

    (Hàm này đã tồn tại y hệt ở tuần 5; copy 6 dòng thay vì import cả module vì import
     hnsw_index kéo theo nguyên phần benchmark. Nếu tuần 8 cần lần thứ ba thì đó là lúc
     tách ra week_common/db_introspect.py — chưa phải bây giờ.)
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'chunks' ORDER BY indexname;"
        )
        return cur.fetchall()


def self_check() -> None:
    """Assert cho idf() bằng số giả — không cần DB."""
    rare = idf(1, 1000)
    common = idf(500, 1000)
    everywhere = idf(1000, 1000)
    missing = idf(0, 1000)

    assert rare > common > everywhere, f"IDF phải giảm dần: {rare} {common} {everywhere}"
    assert everywhere >= 0.0, f"IDF không được âm, đang là {everywhere}"
    assert missing > rare, "từ không tồn tại (df=0) phải có IDF cao nhất"
    assert idf(5, 0) == 0.0, "kho rỗng phải trả 0.0, không được chia 0"
    print("✅ self_check idf: 4/4 assert pass (không cần DB)")
    print(f"   df=0 -> {missing:.3f} · df=1 -> {rare:.3f} · "
          f"df=500 -> {common:.3f} · df=1000 -> {everywhere:.3f}")


def main() -> None:
    if "--dry" in sys.argv:
        self_check()
        return

    with get_conn() as conn:
        migrate_tsv_column(conn)

        total = count_chunks(conn)
        print(f"\n📊 Bảng chunks: {total} dòng")
        if total < 1000:
            print("   ⚠️  Kho nhỏ — thống kê IDF bên dưới chỉ mang tính minh hoạ, và index")
            print("      GIN có thể không được planner chọn. Xem việc nhân bản dữ liệu ở")
            print("      seed_bulk_rows.py trước khi kết luận bất cứ điều gì về tốc độ.")

        print("\n📋 Index hiện có:")
        for name, definition in list_indexes(conn):
            print(f"   {name}: {definition}")

        if "--terms" in sys.argv:
            question = sys.argv[sys.argv.index("--terms") + 1]
            print(f"\n🔍 Phân tích từ khoá của: {question!r}")
            print(f"   {'token':<20} {'df':>7} {'idf':>8}")
            for token, document_frequency, score in describe_query_terms(conn, question):
                flag = "  <- vô dụng" if score < 0.5 else ("  <- rất hiếm" if score > 5 else "")
                print(f"   {token:<20} {document_frequency:>7} {score:>8.3f}{flag}")
            return

        print(f"\n🔤 {20} token phổ biến nhất (df = số chunk chứa từ đó, N = {total}):")
        print(f"   {'token':<20} {'df':>7} {'df/N':>7} {'idf':>8}")
        for word, document_frequency, _ in top_tokens(conn, limit=20):
            ratio = document_frequency / total if total else 0.0
            print(f"   {word:<20} {document_frequency:>7} {ratio:>6.0%} "
                  f"{idf(document_frequency, total):>8.3f}")

    # ✅ ĐẠT khi:
    #   1. `--dry` in "4/4 assert pass"
    #   2. Chạy hai lần liên tiếp không lệnh nào lỗi (idempotent)
    #   3. list_indexes in ra một dòng có `USING gin`
    #   4. Bảng token: vài dòng đầu có df/N > 50% và idf < 1.0 -> đó chính là "từ vô giá trị"
    #      mà lý thuyết IDF nói tới, giờ nhìn thấy bằng số của KHO CỦA MÌNH
    #   5. `--terms "..."` với một câu hỏi có tên riêng: token tên riêng phải có idf cao
    #      hẳn so với các hư từ trong cùng câu

if __name__ == "__main__":
    main()
