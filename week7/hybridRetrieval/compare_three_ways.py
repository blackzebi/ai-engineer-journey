"""
compare_three_ways.py — Bộ 8 câu qua BA nhánh: vector-only · keyword-only · hybrid

    INPUT :  questions.json (dùng lại nguyên si của tuần 7 T2)
    OUTPUT:  bảng markdown 3 cột thứ hạng + kết luận "hybrid được gì, MẤT gì"

                              câu hỏi
                 +---------------+---------------+
                 |               |               |
          vector-only     keyword-only        hybrid
        (search_chunks)  (search_by_keyword)  (hybrid_search)
                 |               |               |
             rank của expect_source ở mỗi nhánh
                 +---------------+---------------+
                                 |
                          classify_change
             improved / rescued / same / worse / lost / still_miss
                                 |
                    bảng markdown -> note + README

⚠️ ĐIỀU KIỆN NGHIỆM THU: PHẢI CÓ ÍT NHẤT MỘT CÂU TỆ ĐI sau khi hybrid.
    Bảng nào cũng "cải thiện toàn bộ" là dấu hiệu đang tự lừa mình — hoặc bộ câu hỏi quá dễ,
    hoặc code đang lặng lẽ trả về đúng vector-only. RRF có đánh đổi thật: nó dìm chunk
    xuất-sắc-ở-một-nhánh xuống để nâng chunk khá-ở-cả-hai lên. Không thấy cái giá đó nghĩa là
    chưa đo được cái lợi.

So với tuần 7 T2 (compare_branches.py):
    T2: 2 cột, câu hỏi "nhánh nào mạnh hơn ở loại truy vấn nào"
    hôm nay: 3 cột, câu hỏi "gộp lại có HƠN cả hai không, và hơn ở đâu, thua ở đâu"
             ^^^ đây mới là con số đi vào README của Dự án 2

Self-test bằng kết quả giả — không cần DB, không cần model:
    python compare_three_ways.py --dry
Chạy thật (cần Postgres Up + fulltext_schema.py đã chạy):
    python compare_three_ways.py
    python compare_three_ways.py --k 10 --pool 20 --out three_ways.md
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "keywordSearch"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))
sys.path.append(os.path.join(HERE, "..", "..", "week5", "askCli"))

from compare_branches import rank_of_source  # noqa: E402
from hybrid_search import hybrid_search  # noqa: E402
from keyword_search import DEFAULT_MODE, search_by_keyword  # noqa: E402
from questions import (  # noqa: E402
    DEFAULT_QUESTIONS_PATH,
    RetrievalCase,
    load_questions,
    validate_questions,
)
from rrf import DEFAULT_CANDIDATE_POOL, RRF_K_CONSTANT  # noqa: E402
from vector_ops import get_conn  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Vẫn k=10 như T2 để bảng SO SÁNH ĐƯỢC với comparison.md. Đổi k hôm nay là tự tay huỷ
# baseline của chính mình — mọi so sánh sau đó chỉ còn là cảm giác.
DEFAULT_COMPARE_K = 10

# Chênh dưới ngưỡng này coi là "không đổi". Nhất quán với CLEAR_WIN_MARGIN của T2: chênh đúng
# 1 hạng là nhiễu, đổi một chữ trong câu hỏi là đảo ngược. Xây kết luận "hybrid tệ đi" trên
# chênh lệch không lặp lại được là cách nhanh nhất để đi sửa nhầm chỗ.
CHANGE_MARGIN = 2

IMPROVED = "improved"
WORSE = "worse"
SAME = "same"
STILL_MISS = "still_miss"
RESCUED = "rescued"
LOST = "lost"


@dataclass
class ThreeWayResult:
    """Kết quả 3 nhánh của MỘT câu hỏi.

    best_baseline_rank: hạng TỐT NHẤT mà một nhánh đơn lẻ đạt được. So hybrid với con số này
    chứ không so với riêng vector — nếu chỉ so với vector thì mọi ca keyword thắng đều thành
    "hybrid cải thiện", trong khi thật ra hybrid chỉ đang bắt kịp nhánh kia.
    """

    case_id: str
    question: str
    query_kind: str
    expect_source: str | None
    vector_rank: int | None
    keyword_rank: int | None
    hybrid_rank: int | None
    change: str

    @property
    def best_baseline_rank(self) -> int | None:
        ranks = [r for r in (self.vector_rank, self.keyword_rank) if r is not None]
        return min(ranks) if ranks else None


def classify_change(baseline_rank: int | None, hybrid_rank: int | None) -> str:
    """So hạng tốt nhất của nhánh đơn với hạng của hybrid -> nhãn. Hàm thuần, không chạm DB.

    Ví dụ: (5, 1)       -> 'improved'    (hạng NHỎ hơn là tốt hơn)
           (1, 4)       -> 'worse'       (cái giá của fusion — phải có ít nhất 1 ca thế này)
           (2, 3)       -> 'same'        (chênh 1 hạng là nhiễu)
           (None, 3)    -> 'rescued'     (không nhánh nào tìm ra, hybrid tìm ra)
           (2, None)    -> 'lost'        (nhánh đơn tìm ra, hybrid làm mất — CỜ ĐỎ)
           (None, None) -> 'still_miss'  (cả 3 cùng trượt)

    'rescued' và 'lost' tách riêng khỏi 'improved'/'worse' vì ý nghĩa hành động khác hẳn:
    'rescued' là bằng chứng mạnh nhất cho hybrid, đáng đưa vào README; 'lost' là CỜ ĐỎ —
    hybrid làm biến mất tài liệu mà nhánh đơn đã tìm được, dấu hiệu candidate_pool < k hoặc
    fusion cắt top_k trước khi sắp xếp. Gộp vào 'worse' là chôn mất tín hiệu chẩn đoán.
    """
    # Xử None TRƯỚC khi so số: `baseline_rank < hybrid_rank` với một bên là None sẽ ném
    # TypeError. May là ồn ào, nhưng vẫn nên chặn tại đây cho rõ ý.
    if baseline_rank is None and hybrid_rank is None:
        return STILL_MISS
    if baseline_rank is None:
        return RESCUED
    if hybrid_rank is None:
        return LOST

    if abs(baseline_rank - hybrid_rank) < CHANGE_MARGIN:
        return SAME
    # Hạng NHỎ hơn là TỐT hơn. Đảo chiều dấu so sánh ở đây là đảo ngược toàn bộ kết luận của
    # ngày, và bảng vẫn ra đẹp đẽ — bug im lặng đi thẳng vào note rồi nằm đó 3 tuần.
    return IMPROVED if hybrid_rank < baseline_rank else WORSE


def compare_one_question(
    conn,
    case: RetrievalCase,
    k: int = DEFAULT_COMPARE_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    k_constant: int = RRF_K_CONSTANT,
    mode: str = DEFAULT_MODE,
) -> ThreeWayResult:
    """Chạy 1 câu hỏi qua cả BA nhánh -> ThreeWayResult.

    Ví dụ: compare_one_question(conn, case_K2)
           -> ThreeWayResult(vector_rank=3, keyword_rank=1, hybrid_rank=1, change='improved')

    Chạy lại cả vector và keyword ở đây thay vì đọc số cũ từ comparison.md, vì kho có thể đã
    đổi (ingest thêm file, chạy lại chunking). Số cũ so với số mới là so hai điều kiện đo
    khác nhau. Ba con số phải sinh ra trong CÙNG một lần chạy mới so được.
    """
    from retriever import search_chunks  # KHÔNG phải retrieve — xem docstring hybrid_search.py
    from search_pg import embed_query

    # pool < k thì hybrid chỉ có `pool` ứng viên để trả `k` kết quả -> luôn thua hai nhánh
    # kia, và kết luận sẽ là "RRF không hoạt động" trong khi RRF vẫn đúng. Chặn ngay, đừng để
    # cấu hình vô nghĩa biến thành một dòng trong bảng đo.
    if candidate_pool < k:
        raise ValueError(
            f"candidate_pool ({candidate_pool}) phải >= k ({k}), "
            "không thì hybrid luôn thua oan"
        )

    query_vector = embed_query(case.question)
    vector_hits = search_chunks(conn, query_vector, k=k)
    keyword_hits = search_by_keyword(conn, case.question, k=k, mode=mode)
    hybrid_hits = hybrid_search(
        conn, case.question, top_k=k,
        candidate_pool=candidate_pool, k_constant=k_constant, mode=mode,
    )

    # rank_of_source (dùng lại của T2) ăn bất cứ object nào có `.source`, khớp một phần,
    # không phân biệt hoa thường -> HybridHit dùng được ngay, không cần adapter.
    vector_rank = rank_of_source(vector_hits, case.expect_source)
    keyword_rank = rank_of_source(keyword_hits, case.expect_source)
    hybrid_rank = rank_of_source(hybrid_hits, case.expect_source)

    # Tạo object TRƯỚC rồi mới tính change, để dùng property best_baseline_rank thay vì tự
    # viết lại min(...) ở đây — một chỗ tính, một chỗ sai nếu sai.
    result = ThreeWayResult(
        case_id=case.id, question=case.question, query_kind=case.query_kind,
        expect_source=case.expect_source, vector_rank=vector_rank,
        keyword_rank=keyword_rank, hybrid_rank=hybrid_rank, change=SAME
    )
    result.change = classify_change(result.best_baseline_rank, hybrid_rank)
    return result


def render_three_way_table(results: list[ThreeWayResult]) -> str:
    """Bảng markdown dán thẳng vào note/README. Hàm thuần — không chạm DB.

    Mong muốn:
        | id | loại | câu hỏi | vector | keyword | hybrid | đổi |
        |---|---|---|---:|---:|---:|---|
        | K2 | keyword | Hermes engine trong React Native... | 3 | 1 | 1 | same |
        | S2 | semantic | Vì sao một trang hiển thị danh sách... | 4 | 9 | 2 | **improved** |

    In cả 3 cột hạng chứ không chỉ cột 'đổi': người đọc (kể cả mình 3 tuần sau) cần thấy SỐ
    để tự kiểm tra kết luận. Bảng chỉ có nhãn 'improved' là bắt người đọc tin mình.
    """
    def cell(rank: int | None) -> str:
        return "—" if rank is None else str(rank)

    # `---:` là căn phải cột số — số thẳng hàng thì mắt bắt được xu hướng ngay.
    lines = [
        "| id | loại | câu hỏi | vector | keyword | hybrid | đổi |",
        "|---|---|---|---:|---:|---:|---|",
    ]

    highlight = {IMPROVED, WORSE, RESCUED, LOST}
    for r in results:
        # Ký tự `|` trong câu hỏi sẽ phá cột markdown — lỗi chỉ lộ ra khi preview file .md,
        # tức là sau khi đã dán vào note và tưởng xong.
        question = r.question.replace("|", "/")[:50]
        change = f"**{r.change}**" if r.change in highlight else r.change
        lines.append(
            f"| {r.case_id} | {r.query_kind} | {question} | "
            f"{cell(r.vector_rank)} | {cell(r.keyword_rank)} | "
            f"{cell(r.hybrid_rank)} | {change} |"
        )
    return "\n".join(lines)


def print_summary(results: list[ThreeWayResult]) -> str:
    """Tổng kết + cảnh báo khi bảng "đẹp quá mức". In ra và trả về text để ghi vào file luôn.

    Vì sao hàm tổng kết lại đi CẢNH BÁO khi kết quả TỐT: vì "cải thiện toàn bộ, không mất gì"
    gần như luôn là dấu hiệu code sai chứ không phải thuật toán giỏi. Hay gặp nhất là
    hybrid_search lặng lẽ trả về đúng danh sách vector-only (nhánh keyword rỗng sạch), nên nó
    "không bao giờ tệ hơn". Một hàm tổng kết trung thực phải nói ra điều đó, không phải chúc
    mừng.

    Trả về str thay vì chỉ print để ghi nguyên khối này vào file .md cùng bảng — bảng không
    kèm cách đọc thì 3 tuần sau nhìn lại chỉ còn là mấy con số.
    """
    # Counter trả 0 cho khoá chưa gặp (không ném KeyError). Tiện, nhưng nghĩa là gõ nhầm tên
    # nhãn sẽ im lặng in "0" -> luôn dùng HẰNG, đừng gõ chuỗi "improved" ở đây.
    counter = Counter(r.change for r in results)
    lines = ["=== TỔNG KẾT ==="]
    for label in (IMPROVED, RESCUED, SAME, WORSE, LOST, STILL_MISS):
        lines.append(f"  {label:<12}: {counter[label]}")

    lines.append("")
    lines.append("=== ĐỌC KẾT QUẢ CHO ĐÚNG ===")

    if counter[WORSE] == 0 and counter[LOST] == 0:
        lines.append("  ⚠️ KHÔNG có ca nào tệ đi. Phải có ít nhất 1 ca tệ đi mới là đo thật.")
        lines.append("     Kiểm theo thứ tự:")
        lines.append("     (a) nhánh keyword có trả về gì không? Rỗng sạch thì hybrid")
        lines.append("         = vector-only, và nó 'không bao giờ tệ hơn' một cách vô nghĩa")
        lines.append("     (b) candidate_pool có thật sự > k không?")
        lines.append("     (c) thử --rrf-k 1: k nhỏ ưu ái hạng 1 của một nhánh, phải thấy đổi")
    else:
        lines.append(f"  ✅ Có {counter[WORSE] + counter[LOST]} ca tệ đi -> phép đo này")
        lines.append("     đáng tin. Với MỖI ca, viết vào note một câu VÌ SAO nó tụt.")

    if counter[LOST]:
        lines.append(f"  🚩 {counter[LOST]} ca hybrid LÀM MẤT tài liệu mà nhánh đơn đã")
        lines.append("     tìm ra. Kiểm ngay: candidate_pool < k, hoặc fuse_rankings cắt")
        lines.append("     top_k TRƯỚC khi sắp xếp.")

    if counter[STILL_MISS]:
        lines.append(f"  ℹ️ {counter[STILL_MISS]} ca cả ba cùng trượt. Với ca no_answer")
        lines.append("     (N1, N2) thì đây là kết quả ĐÚNG. Với ca answerable thì vấn đề")
        lines.append("     nằm ở chunking/ingest — đó là danh sách việc của T4 (rerank).")

    if counter[RESCUED]:
        lines.append(f"  ⭐ {counter[RESCUED]} ca hybrid tìm ra thứ CẢ HAI nhánh đơn đều")
        lines.append("     bỏ sót. Đây là bằng chứng mạnh nhất cho hybrid — đưa vào README.")

    text = "\n".join(lines)
    print("\n" + text)
    return text


def self_check() -> None:
    """Assert bằng kết quả giả — không cần DB, không cần model."""

    assert classify_change(5, 1) == IMPROVED, "hạng NHỎ hơn là tốt hơn"
    assert classify_change(1, 4) == WORSE
    assert classify_change(2, 3) == SAME, "chênh 1 hạng là nhiễu"
    assert classify_change(3, 3) == SAME
    assert classify_change(None, 3) == RESCUED
    assert classify_change(2, None) == LOST
    assert classify_change(None, None) == STILL_MISS

    fake_results = [
        ThreeWayResult("K1", "Chuỗi kết nối MongoDB cổng 27017?", "keyword",
                       "Top 50 Full Stack", 1, None, 1, SAME),
        ThreeWayResult("K2", "Hermes engine trong React Native là gì?", "keyword",
                       "Frontend Interview Prep", 3, 1, 1, IMPROVED),
        ThreeWayResult("S2", "Vì sao một trang | danh sách tạo hàng loạt query?", "semantic",
                       "Backend-NodeJS", 4, 9, 2, IMPROVED),
        ThreeWayResult("M2", "Prop action của thẻ form React 19?", "mixed",
                       "React Interview", 1, 3, 4, WORSE),
        ThreeWayResult("N1", "Hash mật khẩu Argon2id memory cost?", "keyword",
                       None, None, None, None, STILL_MISS),
    ]

    assert fake_results[1].best_baseline_rank == 1, "best_baseline lấy hạng TỐT NHẤT của 2 nhánh"
    assert fake_results[0].best_baseline_rank == 1
    assert fake_results[4].best_baseline_rank is None

    table = render_three_way_table(fake_results)
    assert "None" not in table, "không được để chữ 'None' lọt vào bảng"
    assert table.count("\n") == len(fake_results) + 1, "phải có 2 dòng header + n dòng dữ liệu"
    assert "**improved**" in table and "**worse**" in table, "nhãn đáng chú ý phải in đậm"
    assert "danh sách tạo hàng loạt" in table, "ký tự | trong câu hỏi phải bị thay, không phá cột"

    summary = print_summary(fake_results)
    assert "TỔNG KẾT" in summary
    assert "✅" in summary, "bộ giả này CÓ ca worse -> phải khen, không cảnh báo"

    # Bộ toàn cải thiện phải bị NGHI NGỜ, không được khen — xem docstring print_summary.
    all_good = [r for r in fake_results if r.change in (IMPROVED, SAME)]
    summary_suspicious = print_summary(all_good)
    assert "⚠️" in summary_suspicious, "bộ toàn cải thiện PHẢI bị cảnh báo, không được khen"

    print("\n✅ self_check: classify_change + render_three_way_table + print_summary pass")
    print("\n" + table)


def parse_int_flag(flag: str, default: int) -> int:
    """Đọc `--flag <số>` từ sys.argv."""
    if flag not in sys.argv:
        return default
    return int(sys.argv[sys.argv.index(flag) + 1])


def main() -> None:
    if "--dry" in sys.argv:
        self_check()
        return

    k = parse_int_flag("--k", DEFAULT_COMPARE_K)
    candidate_pool = parse_int_flag("--pool", DEFAULT_CANDIDATE_POOL)
    k_constant = parse_int_flag("--rrf-k", RRF_K_CONSTANT)

    cases, corpus_note = load_questions(DEFAULT_QUESTIONS_PATH)
    errors = validate_questions(cases)
    if errors:
        print(f"❌ questions.json còn {len(errors)} lỗi — sửa hết rồi mới đo:")
        for e in errors:
            print(f"   - {e}")
        raise SystemExit(1)

    print(f"📚 corpus: {corpus_note[:120]}...")
    print(f"🔬 So 3 nhánh trên {len(cases)} câu · top-{k} · pool={candidate_pool} · rrf_k={k_constant}\n")

    results = []
    with get_conn() as conn:
        for case in cases:
            result = compare_one_question(
                conn, case, k=k, candidate_pool=candidate_pool, k_constant=k_constant
            )
            results.append(result)
            print(f"  {case.id:<4} v={result.vector_rank}  k={result.keyword_rank}  "
                  f"h={result.hybrid_rank}  -> {result.change}")

    table = render_three_way_table(results)
    print("\n" + table)
    summary = print_summary(results)

    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# So 3 nhánh: vector-only vs keyword-only vs hybrid (RRF)\n\n")
            # Ghi ĐIỀU KIỆN ĐO ngay trên bảng: thiếu nó thì 2 tuần sau con số vô dụng.
            f.write(f"**Điều kiện đo**: top-{k} · candidate_pool={candidate_pool} · "
                    f"rrf_k={k_constant} · mode={DEFAULT_MODE}\n\n")
            f.write(f"Corpus: {corpus_note}\n\n")
            f.write(table + "\n\n```\n" + summary + "\n```\n")
        print(f"\n💾 Đã ghi {out_path}")


if __name__ == "__main__":
    main()
