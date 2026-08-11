"""
compare_branches.py — Chạy bộ 8 câu qua CẢ HAI nhánh, ra bảng "ai thắng ở đâu"

    INPUT :  questions.json + bảng chunks (đã có embedding VÀ cột tsv)
    OUTPUT:  bảng markdown: mỗi câu hỏi -> thứ hạng của tài liệu đúng ở mỗi nhánh

                              câu hỏi
                                 |
                +----------------+----------------+
                |                                 |
        vector-only                        keyword-only
    (week5 retriever.search_chunks)   (keyword_search.search_by_keyword)
                |                                 |
         rank của expect_source            rank của expect_source
                |                                 |
                +--------> decide_winner <--------+
                                 |
                    bảng markdown -> dán vào note + README

⭐ ĐÂY LÀ SẢN PHẨM QUAN TRỌNG NHẤT CỦA HÔM NAY. Cả tuần 7 (hybrid T3, rerank T4,
   mini-project T5) chỉ có ý nghĩa nếu bảng này cho thấy hai nhánh thắng ở NHỮNG CHỖ
   KHÁC NHAU. Nếu một nhánh thắng mọi ca thì hybrid không giải quyết vấn đề gì cả —
   và biết điều đó hôm nay còn hơn biết vào chiều thứ Sáu.

⚠️ KHÔNG so bằng ĐIỂM. similarity của vector và ts_rank của keyword không cùng đơn vị
   (xem docstring keyword_search.py). So bằng THỨ HẠNG của tài liệu đúng — đó cũng chính
   là ý tưởng nền của RRF ngày mai.


Self-test bằng kết quả giả — không cần DB, không cần model:
    python compare_branches.py --dry
Chạy thật (cần Postgres Up + fulltext_schema.py đã chạy + questions.json đã điền):
    python compare_branches.py
    python compare_branches.py --k 10 --out comparison.md
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))
sys.path.append(os.path.join(HERE, "..", "..", "week5", "askCli"))

from keyword_search import search_by_keyword  # noqa: E402
from questions import DEFAULT_QUESTIONS_PATH, RetrievalCase, load_questions, validate_questions  # noqa: E402
from vector_ops import get_conn  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# k lớn hơn top-k của app thật (3) là CỐ Ý: mình đang đo KHẢ NĂNG TÌM RA của mỗi nhánh,
# không đo trải nghiệm cuối. Tài liệu đúng nằm ở hạng 7 là thông tin quý — nó nói "nhánh
# này có tìm thấy, chỉ là xếp thấp", tức là reranking (T4) sẽ cứu được. Nếu chỉ lấy k=3
# thì ca đó hiện ra là "trượt hoàn toàn" và mình kết luận sai hướng chữa.
DEFAULT_COMPARE_K = 10

# Ngưỡng để gọi là "thắng rõ": chênh nhau ít nhất 2 hạng. Chênh 1 hạng (2 vs 3) là nhiễu,
# đổi một chữ trong câu hỏi là đảo ngược.
CLEAR_WIN_MARGIN = 2

VECTOR = "vector"
KEYWORD = "keyword"
TIE = "tie"
BOTH_MISS = "both_miss"


@dataclass
class BranchComparison:
    """Kết quả so sánh của MỘT câu hỏi.

    vector_rank / keyword_rank: thứ hạng 1-based của expect_source trong top-k của nhánh
                                đó, hoặc None nếu không xuất hiện trong top-k.
    """

    case_id: str
    question: str
    query_kind: str
    expect_source: str | None
    vector_rank: int | None
    keyword_rank: int | None
    winner: str
    predicted_winner: str

    @property
    def prediction_was_right(self) -> bool:
        """Dự đoán của mình có đúng không. 'either' coi là đúng khi kết quả không phải both_miss."""
        if self.predicted_winner == "none":
            return self.winner == BOTH_MISS
        return self.predicted_winner == self.winner


def rank_of_source(hits: list, expect_source: str | None) -> int | None:
    """Thứ hạng 1-based của chunk đầu tiên có source khớp `expect_source`. None nếu không có.

    Ví dụ: hits có source ['a.md', 'bao-cao-q3.pdf', 'c.md'], expect_source='bao-cao-q3'
           -> 2
    """

    if not expect_source:
        return None

    needle = expect_source.lower()

    for rank, hit in enumerate(hits, 1):
        if needle in (hit.source or "").lower():
            return rank

    return None


def decide_winner(vector_rank: int | None, keyword_rank: int | None) -> str:
    """So 2 thứ hạng -> 'vector' | 'keyword' | 'tie' | 'both_miss'. Hàm thuần, không chạm DB.

    Ví dụ: (1, 5)       -> 'vector'      (hạng NHỎ hơn là tốt hơn)
           (None, 2)    -> 'keyword'     (một bên trượt hẳn)
           (2, 3)       -> 'tie'         (chênh 1 hạng là nhiễu, không kết luận)
           (None, None) -> 'both_miss'

    So bằng THỨ HẠNG chứ không bằng ĐIỂM: cosine similarity có thang tuyệt đối [-1, 1] còn
    ts_rank thì không có thang cố định (phụ thuộc số từ khớp, độ dài chunk, cờ chuẩn hoá) —
    hai con số không cùng đơn vị. Đây cũng chính là ý tưởng nền của RRF ở T3.

    CLEAR_WIN_MARGIN = 2: chênh đúng 1 hạng là nhiễu, không đủ để gọi là thắng. Thà báo
    'tie' còn hơn dựng một kết luận trên chênh lệch không lặp lại được.
    """

    if vector_rank is None and keyword_rank is None:
        return BOTH_MISS
    if keyword_rank is None:
        return VECTOR
    if vector_rank is None:
        return KEYWORD
    if abs(vector_rank - keyword_rank) < CLEAR_WIN_MARGIN:
        return TIE

    return VECTOR if vector_rank < keyword_rank else KEYWORD


def compare_one_question(conn, case: RetrievalCase, k: int = DEFAULT_COMPARE_K) -> BranchComparison:
    """Chạy 1 câu hỏi qua cả hai nhánh -> BranchComparison.

    Ví dụ: compare_one_question(conn, case_K1) -> BranchComparison(vector_rank=7, keyword_rank=1, winner='keyword')
    """

    from retriever import search_chunks
    from search_pg import embed_query

    query_vector = embed_query(case.question)
    vector_hits = search_chunks(conn, query_vector, k=k)

    keyword_hits = search_by_keyword(conn, case.question, k=k)

    vector_rank = rank_of_source(vector_hits, case.expect_source)
    keyword_rank = rank_of_source(keyword_hits, case.expect_source)

    return BranchComparison(
        case_id=case.id,
        question=case.question,
        query_kind=case.query_kind,
        expect_source=case.expect_source,
        vector_rank=vector_rank,
        keyword_rank=keyword_rank,
        winner=decide_winner(vector_rank, keyword_rank),
        predicted_winner=case.expect_winner,
    )


def render_comparison_table(results: list[BranchComparison]) -> str:
    """Bảng markdown dán thẳng vào note/README. Hàm thuần — không chạm DB.

    Mong muốn:
        | id | loại | câu hỏi | vector | keyword | thắng | đoán đúng? |
        |---|---|---|---|---|---|---|
        | K1 | keyword | Mã lỗi E402...? | 7 | 1 | **keyword** | ✅ |
        | S1 | semantic | Làm sao cắt nhỏ...? | 1 | — | **vector** | ✅ |
    """

    def cell(rank: int | None) -> str:
        return "—" if rank is None else str(rank)

    lines = [
        "| id | loại | câu hỏi | vector | keyword | thắng | đoán đúng? |",
        "|---|---|---|---:|---:|---|---|",
    ]

    for r in results:
        question = r.question.replace("|", "/")[:50]
        mark = "✅" if r.prediction_was_right else "❌"
        lines.append(
            f"| {r.case_id} | {r.query_kind} | {question} | "
            f"{cell(r.vector_rank)} | {cell(r.keyword_rank)} | "
            f"**{r.winner}** | {mark} |"
        )

    return "\n".join(lines)


def print_summary(results: list[BranchComparison]) -> None:
    """Tổng kết + gợi ý đọc kết quả cho đúng."""
    from collections import Counter

    winner_counter = Counter(r.winner for r in results)
    right = sum(1 for r in results if r.prediction_was_right)

    print("\n=== TỔNG KẾT ===")
    for label in (VECTOR, KEYWORD, TIE, BOTH_MISS):
        print(f"  {label:<12}: {winner_counter[label]}")
    print(f"  dự đoán đúng : {right}/{len(results)}")

    print("\n=== ĐỌC KẾT QUẢ CHO ĐÚNG ===")
    if winner_counter[KEYWORD] == 0:
        print("  ⚠️ Keyword KHÔNG thắng ca nào. Trước khi kết luận 'keyword vô dụng', kiểm:")
        print("     (a) các ca K* có thật sự chứa mã/tên riêng không, hay mình viết chung chung?")
        print("     (b) `--mode or` chưa? plainto_tsquery AND cả câu là gần như chắc chắn 0 dòng")
        print("     (c) `python fulltext_schema.py --terms \"<câu K1>\"` — token mã số có df>0?")
    if winner_counter[BOTH_MISS]:
        print(f"  ⚠️ {winner_counter[BOTH_MISS]} ca CẢ HAI cùng trượt. Hybrid ngày mai KHÔNG cứu")
        print("     được nhóm này — vấn đề nằm ở chunking/ingest hoặc ở chính câu hỏi.")
        print("     Ghi riêng nhóm này vào note, đó là danh sách việc của T4/T5.")
    if winner_counter[TIE] >= len(results) // 2:
        print("  ⚠️ Quá nửa số ca hoà. Bộ câu hỏi chưa phân hoá đủ — hai nhánh đang tìm ra")
        print("     cùng một thứ. Sửa questions.json cho các ca K* 'keyword' hơn nữa.")
    if winner_counter[VECTOR] and winner_counter[KEYWORD]:
        print("  ✅ Hai nhánh thắng ở những ca KHÁC NHAU -> hybrid (T3) có đất dụng võ.")
        print("     Đây chính là bằng chứng cần cho phần mở đầu README tuần 7.")


def self_check() -> None:
    """Assert bằng dữ liệu giả — không cần DB, không cần model."""

    @dataclass
    class FakeHit:
        source: str

    hits = [FakeHit("notes/a.md"), FakeHit("docs/Bao-Cao-Q3-final.pdf"), FakeHit("notes/c.md")]
    assert rank_of_source(hits, "bao-cao-q3") == 2, "phải khớp một phần + không phân biệt hoa thường"
    assert rank_of_source(hits, "khong-co-file-nay") is None, "không khớp phải trả None, KHÔNG phải 0"
    assert rank_of_source(hits, None) is None
    assert rank_of_source([], "a.md") is None

    assert decide_winner(1, 5) == VECTOR, "hạng NHỎ hơn là tốt hơn"
    assert decide_winner(5, 1) == KEYWORD
    assert decide_winner(2, 3) == TIE, "chênh 1 hạng là nhiễu"
    assert decide_winner(None, 2) == KEYWORD
    assert decide_winner(2, None) == VECTOR
    assert decide_winner(None, None) == BOTH_MISS

    fake_results = [
        BranchComparison("K1", "Mã lỗi E402 là gì?", "keyword", "a.pdf", 7, 1, KEYWORD, "keyword"),
        BranchComparison("S1", "Cắt nhỏ tài liệu | thế nào?", "semantic", "c.md", 1, None, VECTOR, "vector"),
        BranchComparison("N1", "Mã E999?", "keyword", None, None, None, BOTH_MISS, "none"),
    ]
    table = render_comparison_table(fake_results)
    assert "|" in table and table.count("\n") == 4, "phải có 2 dòng đầu + 3 dòng dữ liệu"
    assert "None" not in table, "không được để chữ 'None' lọt vào bảng cho người đọc"
    assert "E402" in table

    print("✅ self_check: rank_of_source + decide_winner + render_comparison_table pass")
    print("\n" + table)


def main() -> None:
    if "--dry" in sys.argv:
        self_check()
        return

    k = DEFAULT_COMPARE_K
    if "--k" in sys.argv:
        k = int(sys.argv[sys.argv.index("--k") + 1])

    cases, corpus_note = load_questions(DEFAULT_QUESTIONS_PATH)
    errors = validate_questions(cases)
    if errors:
        print(f"❌ questions.json còn {len(errors)} lỗi — sửa hết rồi mới đo:")
        for e in errors:
            print(f"   - {e}")
        raise SystemExit(1)

    print(f"📚 corpus: {corpus_note}")
    print(f"🔬 So 2 nhánh trên {len(cases)} câu, top-{k}\n")

    results = []
    with get_conn() as conn:
        for case in cases:
            result = compare_one_question(conn, case, k=k)
            results.append(result)
            print(f"  {case.id:<4} vector={result.vector_rank}  "
                  f"keyword={result.keyword_rank}  -> {result.winner}")

    table = render_comparison_table(results)
    print("\n" + table)
    print_summary(results)

    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# So sánh vector-only vs keyword-only (top-{k})\n\n")
            f.write(f"Corpus: {corpus_note}\n\n")
            f.write(table + "\n")
        print(f"\n💾 Đã ghi {out_path}")

    # ✅ ĐẠT khi:
    #   1. `--dry` in "pass" + một bảng markdown mẫu không có chữ 'None'
    #   2. Chạy thật: mỗi ca in ra 2 con số (hoặc None) và một kết luận
    #   3. Bảng markdown dán vào .md preview không vỡ cột
    #   4. ⭐ Có ÍT NHẤT 1 ca keyword thắng và ÍT NHẤT 1 ca vector thắng.
    #      Đây là điều kiện nghiệm thu THẬT của hôm nay — nó là lý do tồn tại của cả tuần 7.
    #      Không đạt thì đọc phần "ĐỌC KẾT QUẢ CHO ĐÚNG" ở trên và sửa BỘ CÂU HỎI trước,
    #      đừng sửa code.
    #   5. Ghi vào note: số ca dự đoán đúng, và với mỗi ca đoán sai, một câu VÌ SAO

if __name__ == "__main__":
    main()
