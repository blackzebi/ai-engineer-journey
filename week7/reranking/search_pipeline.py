"""
search_pipeline.py — Ghép hoàn chỉnh: hybrid(N=20) -> rerank -> top-3

    INPUT :  câu hỏi (hoặc cả bộ 8 câu trong questions.json)
    OUTPUT:  top-3 chunk cuối cùng + bảng "chunk nào leo/tụt vì rerank"

                                 câu hỏi
                                    |
                    ┌───────────────┴───────────────┐
                    │   TẦNG 1 — RẺ, lấy RỘNG        │
                    │   hybrid_search(pool=20)      │   vector + keyword + RRF
                    │   ~80 ms (đo 12/08)           │
                    └───────────────┬───────────────┘
                                    | 20 ứng viên
                    ┌───────────────┴───────────────┐
                    │   TẦNG 2 — ĐẮT, chấm KỸ        │
                    │   rerank_hits(top_k=3)        │   cross-encoder, 20 lần chạy model
                    │   +1519 ms (đo 12/08, CPU)    │
                    └───────────────┬───────────────┘
                                    | 3 chunk
                              câu trả lời

So với tuần 7 T3:
    T3:      hybrid_search(top_k=3) -> 3 chunk, hết
    hôm nay: hybrid_search(top_k=20) -> rerank -> 3 chunk
                             ^^^^ mảnh mới KHÔNG phải "thêm một hàm", mà là ĐỔI VAI của
                             hybrid: hôm qua nó QUYẾT ĐỊNH, hôm nay nó TIẾN CỬ. Ai quyết
                             định thì phải chính xác; ai tiến cử thì chỉ cần rẻ và đừng bỏ sót.

PATTERN (lặp lại khắp hệ thống thật, không riêng RAG):
    lọc RẺ trên tập LỚN  ->  chấm ĐẮT trên tập NHỎ
    Cùng khuôn: quảng cáo (candidate generation -> ranking model), tìm kiếm sản phẩm
    (Elasticsearch -> learning-to-rank), gợi ý bạn bè (ANN -> GNN scoring).
    Câu hỏi thiết kế luôn là N bao nhiêu:
        N quá nhỏ -> tầng 2 không cứu được gì. Ca cực đoan N = k: rerank thành trang trí đắt tiền.
        N quá lớn -> latency tăng gần TUYẾN TÍNH theo N (mỗi ứng viên là một lần chạy model).

⚠️ KẾT QUẢ ĐO ĐƯỢC 12/08 trên bộ 8 câu (xem rerank_shift.md):
    promoted 0 · unchanged 3 · demoted 2 · dropped 1 · not_in_pool 2
    Tức là ở kho này, với câu hỏi tiếng Việt, rerank KHÔNG cải thiện ca nào và làm hỏng 3 ca.
    Kiến trúc thì đúng; model thì sai ngôn ngữ. Đừng đọc bảng này thành "rerank vô dụng".

Self-test bằng dữ liệu giả — không cần DB, không cần model:
    python search_pipeline.py --dry
Chạy thật (cần Postgres Up + fulltext_schema.py đã chạy):
    python search_pipeline.py "Hermes engine trong React Native là gì?"
    python search_pipeline.py --all --out rerank_shift.md
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "hybridRetrieval"))
sys.path.append(os.path.join(HERE, "..", "keywordSearch"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))
sys.path.append(os.path.join(HERE, "..", "..", "week5", "askCli"))

from rerank import (  # noqa: E402
    DEFAULT_RERANK_POOL,
    DEFAULT_TOP_K,
    FakeHit,
    RerankedHit,
    format_rerank_table,
    keyword_overlap_scorer,
    rerank_hits,
)
from rrf import RRF_K_CONSTANT  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Nhãn dịch chuyển của đoạn ĐÚNG (expect_source) sau khi rerank. Dùng HẰNG chứ không gõ chuỗi
# rải rác — gõ nhầm "promotted" thì Counter im lặng đếm 0.
PROMOTED = "promoted"        # đoạn đúng leo lên
DEMOTED = "demoted"          # đoạn đúng tụt xuống — CÁI GIÁ
UNCHANGED = "unchanged"
DROPPED = "dropped"          # đoạn đúng có trong pool nhưng rerank đá khỏi top-k — CỜ ĐỎ
NOT_IN_POOL = "not_in_pool"  # tầng 1 đã bỏ sót, rerank vô can

# Chênh dưới ngưỡng này coi như không đổi. Giữ nguyên 2 như CHANGE_MARGIN của
# compare_three_ways để ba ngày T2/T3/T4 dùng CÙNG một định nghĩa "đổi thật sự".
SHIFT_MARGIN = 2


@dataclass
class ShiftResult:
    """Đoạn ĐÚNG của 1 câu hỏi đứng hạng mấy trước và sau rerank.

    rank_hybrid: hạng của expect_source trong 20 ứng viên do hybrid tiến cử (None = tầng 1
                 bỏ sót — rerank không có lỗi gì ở đây, đừng đổ tội nhầm chỗ).
    rank_final : hạng sau rerank, tính trong CÙNG 20 ứng viên đó (None với rank_hybrid khác
                 None là ca bất khả thi, vì rerank chỉ hoán vị chứ không vứt bớt).
    """

    case_id: str
    question: str
    query_kind: str
    expect_source: str | None
    rank_hybrid: int | None
    rank_final: int | None
    shift: str


def search_with_rerank(
    conn,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool: int = DEFAULT_RERANK_POOL,
    k_constant: int = RRF_K_CONSTANT,
) -> tuple[list[RerankedHit], list]:
    """Pipeline đầy đủ. Trả (top_k đã rerank, DANH SÁCH ỨNG VIÊN GỐC để so sánh).

    Ví dụ: search_with_rerank(conn, "Hermes là gì?", top_k=3, candidate_pool=20)
           -> ([RerankedHit x3], [HybridHit x20])

    ⚠️ `hybrid_search(..., top_k=candidate_pool)` — KHÔNG phải `top_k=top_k`. `top_k` của
    hybrid_search là số kết quả nó TRẢ VỀ; truyền 3 vào đó thì reranker chỉ nhận 3 ứng viên
    và toàn bộ kiến trúc 2 tầng sụp thành "sắp xếp lại 3 phần tử". Chạy đúng, không lỗi,
    latency vẫn tăng, và kết luận sẽ là "rerank tốn thời gian mà chẳng cải thiện gì".
    Đọc thành lời: lấy 20 từ mỗi nhánh, fusion, TRẢ VỀ CẢ 20 — cắt xuống 3 là việc của tầng sau.

    Trả về CẢ danh sách ứng viên gốc vì format_rerank_table cần nó để biết chunk nào bị đá
    khỏi top-k. Chạy hybrid lần hai để lấy lại là tốn thêm một lần embed và có thể ra khác.

    KHÔNG đặt ngưỡng similarity/score ở đây: ngưỡng là quyết định của TẦNG APP (trả lời hay
    từ chối), thuộc về SAU pipeline — cùng nguyên tắc đã chốt khi chọn `search_chunks` chứ
    không `retrieve`.
    """
    from hybrid_search import hybrid_search  # import lười: kéo theo sentence_transformers

    if candidate_pool < top_k:
        raise ValueError(
            f"candidate_pool ({candidate_pool}) phải >= top_k ({top_k}) — "
            "rerank không tạo thêm ứng viên, nó chỉ xếp lại"
        )

    candidates = hybrid_search(
        conn, question,
        top_k=candidate_pool,
        candidate_pool=candidate_pool,
        k_constant=k_constant,
    )

    reranked = rerank_hits(question, candidates, top_k=top_k)

    return reranked, candidates


def classify_shift(rank_hybrid: int | None, rank_final: int | None, top_k: int) -> str:
    """So hạng của đoạn ĐÚNG trước/sau rerank -> nhãn. Hàm thuần, không chạm DB.

    Ví dụ: (11, 1, 3)      -> 'promoted'    (hạng NHỎ hơn là tốt hơn)
           (1, 6, 3)       -> 'dropped'     (rerank đá đoạn đúng khỏi top-3 — CỜ ĐỎ)
           (5, 9, 3)       -> 'demoted'     (tụt nhưng vốn đã ngoài top-3)
           (2, 3, 3)       -> 'unchanged'   (chênh 1 hạng là nhiễu)
           (None, None, 3) -> 'not_in_pool' (tầng 1 bỏ sót, rerank vô can)

    'dropped' tách riêng khỏi 'demoted' vì ý nghĩa hành động khác hẳn: tụt từ 9 xuống 12
    chẳng ảnh hưởng câu trả lời cuối, còn 'dropped' là người dùng ĐANG có đoạn đúng trong tay
    và rerank lấy đi mất — hồi quy chất lượng thật, phải đếm riêng.

    'not_in_pool' không phải lỗi của rerank: rerank chỉ xếp lại thứ được đưa cho. Sửa thì
    phải sửa TẦNG 1 (tăng pool, sửa tsv, đổi chunking), không phải đổi model rerank.
    """
    # Xử None TRƯỚC khi so số: `rank_hybrid < rank_final` với một bên None ném TypeError.
    if rank_hybrid is None:
        return NOT_IN_POOL

    if rank_final is None:
        raise ValueError(
            "đoạn đúng có trong pool nhưng biến mất sau rerank — rerank_hits đang làm "
            "mất phần tử, kiểm lại chỗ cắt top_k"
        )

    # `a <= b < c` là so sánh dây chuyền của Python, đọc y như toán học.
    if rank_hybrid <= top_k < rank_final:
        return DROPPED

    if abs(rank_hybrid - rank_final) < SHIFT_MARGIN:
        return UNCHANGED

    # Hạng NHỎ hơn là TỐT hơn. Đảo dấu ở đây là đảo toàn bộ kết luận của ngày, và bảng vẫn ra đẹp.
    return PROMOTED if rank_final < rank_hybrid else DEMOTED


def render_shift_table(results: list[ShiftResult], top_k: int) -> str:
    """Bảng markdown: rerank làm đoạn đúng leo hay tụt. Hàm thuần — dán thẳng vào note/README.

    Mong muốn:
        | id | loại | câu hỏi | hybrid | +rerank | đổi |
        |---|---|---|---:|---:|---|
        | K2 | keyword | Hermes engine trong React Native là gì? | 4 | 1 | **promoted** |

    Đây là bảng duy nhất tách bạch được đóng góp của RERANK khỏi đóng góp của HYBRID — khi số
    xấu thì biết đổ tội đúng tầng.
    """
    def cell(rank: int | None) -> str:
        # "—" chứ không "None": not_in_pool là kết quả hợp lệ, in "None" làm người đọc tưởng lỗi.
        return "—" if rank is None else str(rank)

    # `---:` = căn phải cột số, để mắt bắt được xu hướng ngay.
    lines = [
        f"| id | loại | câu hỏi | hybrid | +rerank (top-{top_k}) | đổi |",
        "|---|---|---|---:|---:|---|",
    ]

    highlight = {PROMOTED, DEMOTED, DROPPED}
    for r in results:
        # Ký tự `|` trong câu hỏi phá cột markdown, và lỗi chỉ lộ ra lúc preview file .md —
        # tức là sau khi đã dán vào note và tưởng xong.
        question = r.question.replace("|", "/")[:50]
        shift = f"**{r.shift}**" if r.shift in highlight else r.shift
        lines.append(
            f"| {r.case_id} | {r.query_kind} | {question} | "
            f"{cell(r.rank_hybrid)} | {cell(r.rank_final)} | {shift} |"
        )

    return "\n".join(lines)


def print_shift_verdict(results: list[ShiftResult]) -> str:
    """Tổng kết + cảnh báo khi bảng "đẹp quá mức". In ra và TRẢ VỀ text để ghi thẳng vào file.

    Vì sao hàm tổng kết lại NGHI NGỜ kết quả tốt: reranker có đánh đổi thật, nên bảng 8/8
    promoted gần như luôn là một trong ba thứ — pool đang bằng top_k, bộ câu hỏi quá dễ, hoặc
    rank_hybrid đang lấy nhầm từ hybrid_search(top_k=3) chứ không phải pool 20.

    Trả về str thay vì chỉ print để ghi nguyên khối "cách đọc" này vào file .md cùng bảng.
    Bảng không kèm cách đọc thì 3 tuần sau chỉ còn là mấy con số.
    """
    # Counter trả 0 cho khoá chưa gặp (không ném KeyError) — tiện, nhưng nghĩa là gõ nhầm tên
    # nhãn sẽ im lặng in "0". Luôn dùng HẰNG ở đây.
    counter = Counter(r.shift for r in results)
    lines = ["=== RERANK LÀM GÌ VỚI ĐOẠN ĐÚNG ==="]
    for label in (PROMOTED, UNCHANGED, DEMOTED, DROPPED, NOT_IN_POOL):
        lines.append(f"  {label:<12}: {counter[label]}")

    lines.append("")
    if counter[DEMOTED] == 0 and counter[DROPPED] == 0:
        lines.append("  ⚠️ KHÔNG ca nào tệ đi. Kiểm 3 chỗ theo thứ tự:")
        lines.append("     (a) len(candidates) có thật sự ~20 không, hay đang là 3?")
        lines.append("     (b) đoạn đúng có vốn đã hạng 1 ở mọi câu không (bộ câu quá dễ)?")
        lines.append("     (c) rerank_hits có thật sự sort không — thử đảo chiều sort,")
        lines.append("         nếu kết quả KHÔNG đổi thì nó chưa bao giờ chạy")
    else:
        lines.append(f"  ✅ Có {counter[DEMOTED] + counter[DROPPED]} ca tệ đi -> phép đo")
        lines.append("     đáng tin. Viết vào note MỘT CÂU vì sao từng ca tụt.")
    if counter[DROPPED]:
        lines.append(f"  🚩 {counter[DROPPED]} ca rerank ĐÁ đoạn đúng khỏi top-k.")
        lines.append("     Xem chunk nào đã thay chỗ nó: tiếng Anh hay tiếng Việt?")
        lines.append("     Đây là chỗ kiểm giả thuyết 'ms-marco yếu với tiếng Việt'.")
    if counter[NOT_IN_POOL]:
        lines.append(f"  ℹ️ {counter[NOT_IN_POOL]} ca đoạn đúng KHÔNG nằm trong pool 20.")
        lines.append("     Đây là giới hạn của TẦNG 1, rerank vô can. Muốn cứu thì tăng")
        lines.append("     pool hoặc sửa tsv (ca K1), không phải đổi model rerank.")

    text = "\n".join(lines)
    print("\n" + text)
    return text


def evaluate_all_questions(conn, top_k: int, candidate_pool: int) -> list[ShiftResult]:
    """Chạy cả bộ 8 câu qua pipeline -> list ShiftResult.

    Gọi `search_with_rerank` với top_k=candidate_pool để có ĐỦ thứ hạng sau rerank của cả 20
    ứng viên -> tính được hạng của đoạn đúng dù nó rơi ra ngoài top-3. Cắt xuống 3 là việc
    của app, không phải của phép đo.

    `rank_of_source` (dùng lại của T2) ăn object có `.source`, mà `RerankedHit` không có
    `.source` trực tiếp (nó bọc hit gốc) -> phải truyền `[r.hit for r in reranked]`.
    """
    from compare_branches import rank_of_source
    from questions import DEFAULT_QUESTIONS_PATH, load_questions, validate_questions

    cases, corpus_note = load_questions(DEFAULT_QUESTIONS_PATH)
    errors = validate_questions(cases)
    if errors:
        print(f"❌ questions.json còn {len(errors)} lỗi — sửa hết rồi mới đo:")
        for e in errors:
            print(f"   - {e}")
        raise SystemExit(1)
    print(f"📚 corpus: {corpus_note[:110]}...\n")

    results = []
    for case in cases:
        reranked, candidates = search_with_rerank(
            conn, case.question, top_k=candidate_pool, candidate_pool=candidate_pool
        )
        rank_hybrid = rank_of_source(candidates, case.expect_source)
        rank_final = rank_of_source([r.hit for r in reranked], case.expect_source)
        shift = classify_shift(rank_hybrid, rank_final, top_k)
        results.append(ShiftResult(
            case_id=case.id, question=case.question, query_kind=case.query_kind,
            expect_source=case.expect_source, rank_hybrid=rank_hybrid,
            rank_final=rank_final, shift=shift,
        ))
        print(f"  {case.id:<4} hybrid={rank_hybrid}  rerank={rank_final}  -> {shift}")

    return results


def self_check() -> None:
    """Assert bằng dữ liệu giả + scorer giả — không cần DB, không cần model."""

    assert classify_shift(11, 1, 3) == PROMOTED
    assert classify_shift(1, 6, 3) == DROPPED, "vốn trong top-3, bị đẩy ra ngoài = CỜ ĐỎ"
    assert classify_shift(5, 9, 3) == DEMOTED, "tụt nhưng vốn đã ngoài top-3"
    assert classify_shift(2, 3, 3) == UNCHANGED, "chênh 1 hạng là nhiễu"
    assert classify_shift(3, 3, 3) == UNCHANGED
    assert classify_shift(None, None, 3) == NOT_IN_POOL
    try:
        classify_shift(2, None, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("đoạn đúng biến mất sau rerank là ca bất khả thi, phải nổ ValueError")

    fake_results = [
        ShiftResult("K2", "Hermes engine trong React Native là gì?", "keyword",
                    "Frontend Interview Prep", 4, 1, PROMOTED),
        ShiftResult("S1", "Làm sao để không gọi API | liên tục khi gõ?", "semantic",
                    "Frontend Interview Prep", 2, 2, UNCHANGED),
        ShiftResult("M2", "Prop action của thẻ form React 19?", "mixed",
                    "React Interview Questions", 1, 6, DROPPED),
        ShiftResult("N1", "Hash mật khẩu Argon2id memory cost?", "keyword",
                    None, None, None, NOT_IN_POOL),
    ]

    table = render_shift_table(fake_results, top_k=3)
    assert "None" not in table, "không được để chữ 'None' lọt vào bảng"
    assert table.count("\n") == len(fake_results) + 1, "2 dòng header + n dòng dữ liệu"
    assert "**promoted**" in table and "**dropped**" in table, "nhãn đáng chú ý phải in đậm"
    assert "không gọi API / liên tục" in table, "ký tự | phải bị thay, không được phá cột"

    verdict = print_shift_verdict(fake_results)
    assert "🚩" in verdict, "có ca dropped thì phải cắm cờ đỏ"
    assert "✅" in verdict, "có ca tệ đi -> phép đo đáng tin"

    rosy = [r for r in fake_results if r.shift in (PROMOTED, UNCHANGED)]
    assert "⚠️" in print_shift_verdict(rosy), "bộ toàn tốt PHẢI bị nghi ngờ, không được khen"

    # Pipeline end-to-end bằng scorer giả: chunk đúng nằm CUỐI pool 5 phần tử
    pool = [
        FakeHit(1, "Redux quản lý state."),
        FakeHit(2, "Node.js chạy trên V8."),
        FakeHit(3, "CSS flexbox căn giữa."),
        FakeHit(4, "Bridge của React Native."),
        FakeHit(5, "Hermes engine trong React Native là gì: JS engine của Meta."),
    ]
    reranked = rerank_hits("Hermes engine trong React Native là gì", pool, top_k=3,
                           scorer=keyword_overlap_scorer)
    assert reranked[0].id == 5 and reranked[0].rank_before == 5, \
        "chunk cuối pool phải leo lên hạng 1 — nếu không, pool đang bị cắt trước khi sort"

    print("\n✅ self_check: classify_shift + render_shift_table + print_shift_verdict pass")
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

    top_k = parse_int_flag("--k", DEFAULT_TOP_K)
    candidate_pool = parse_int_flag("--pool", DEFAULT_RERANK_POOL)

    from vector_ops import get_conn

    if "--all" in sys.argv:
        print(f"🔬 Cả bộ 8 câu · pool={candidate_pool} -> rerank -> top-{top_k}\n")
        with get_conn() as conn:
            results = evaluate_all_questions(conn, top_k=top_k, candidate_pool=candidate_pool)
        table = render_shift_table(results, top_k)
        print("\n" + table)
        verdict = print_shift_verdict(results)

        if "--out" in sys.argv:
            out_path = sys.argv[sys.argv.index("--out") + 1]
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("# Rerank làm gì với đoạn đúng (hybrid -> +cross-encoder)\n\n")
                # ĐIỀU KIỆN ĐO ghi ngay trên bảng — quy trình rút ra từ tuần 5.
                f.write(f"**Điều kiện đo**: candidate_pool={candidate_pool} · top_k={top_k} · "
                        f"rrf_k={RRF_K_CONSTANT} · model=cross-encoder/ms-marco-MiniLM-L-6-v2\n\n")
                f.write(table + "\n\n```\n" + verdict + "\n```\n")
            print(f"\n💾 Đã ghi {out_path}")
        return

    # Bỏ qua cả cờ lẫn GIÁ TRỊ của nó khi gom câu hỏi, kẻo "20" của `--pool 20` bị nối vào câu hỏi.
    question = " ".join(
        a for i, a in enumerate(sys.argv[1:], 1)
        if not a.startswith("--") and not (sys.argv[i - 1] in ("--k", "--pool"))
    )
    if not question:
        raise SystemExit(
            'Dùng: python search_pipeline.py "câu hỏi" [--k 3] [--pool 20] | --all | --dry'
        )

    print(f"❓ {question}")
    print(f"   TẦNG 1 hybrid(pool={candidate_pool}) -> TẦNG 2 rerank -> top-{top_k}\n")
    with get_conn() as conn:
        reranked, candidates = search_with_rerank(
            conn, question, top_k=top_k, candidate_pool=candidate_pool
        )
        print(f"  tầng 1 tiến cử {len(candidates)} ứng viên "
              f"({'ĐÚNG' if len(candidates) > top_k else '⚠️ SAI — pool đang bằng top_k'})\n")
        print(format_rerank_table(reranked, candidates))


if __name__ == "__main__":
    main()

# Kỳ vọng khi chạy:
#   `--dry`        : in "pass" trong dưới 1 giây
#   1 câu hỏi      : dòng "tầng 1 tiến cử N ứng viên" hiện N ~ 20, KHÔNG phải 3
#   `--all`        : bảng tổng kết có ít nhất 1 ca demoted hoặc dropped — toàn promoted là
#                    dấu hiệu code sai, không phải thuật toán giỏi
