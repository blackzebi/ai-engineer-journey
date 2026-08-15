"""
compare_modes.py — ✅ MINI-PROJECT #6: 8 câu hỏi × 3 CHẾ ĐỘ, trả lời bằng SỐ

    INPUT :  questions.json (dùng lại nguyên si của T2) + kho chunks hiện có trong Postgres
    OUTPUT:  bảng markdown (top-3 mỗi chế độ · hạng đoạn đúng · MRR · hit@3 · latency)
             + kết luận "chế độ nào đáng dùng cho Dự án 1 và VÌ SAO"

                            8 câu hỏi
        +-----------------------+-----------------------+
        |                       |                       |
   ① vector-only          ② hybrid (RRF)       ③ hybrid + rerank
   search_chunks           hybrid_search        search_with_rerank
   (1 tầng)                (2 nhánh, 1 tầng)    (2 nhánh, 2 tầng)
        |                       |                       |
        +-----------------------+-----------------------+
                                |
                    rank_of_source (dùng lại T2)  + đồng hồ
                                |
                    metrics.py: MRR · hit@3 · p50 ms
                                |
                        bảng markdown -> README week7/

⚠️ ĐIỀU KIỆN NGHIỆM THU: PHẢI GHI CẢ CA CHẾ ĐỘ PHỨC TẠP HƠN LẠI TỆ HƠN.
    Bảng mà "③ > ② > ①" trên mọi dòng gần như luôn là dấu hiệu code sai, không phải kiến
    trúc giỏi. Rerank có cái giá thật: nó tin vào cross-encoder, mà cross-encoder này
    (ms-marco-MiniLM) được huấn luyện trên tiếng Anh — với 4 câu hỏi tiếng Việt trong bộ 8
    câu, nó SẼ có ca chấm sai. Không thấy ca nào tệ đi nghĩa là chưa đo được gì.

So với tuần 7 T3 (compare_three_ways.py):
    T3: 3 cột vector/keyword/hybrid, so bằng THỨ HẠNG từng câu, mắt người tự đọc
    hôm nay: 3 chế độ THẬT SỰ DÙNG ĐƯỢC (bỏ keyword-only vì không ai deploy nó một mình),
             + gộp thành MRR/hit@3/latency  <-- mảnh mới
             ^^^ đây là bảng đi vào README của Dự án 1, không phải bảng để tự xem

Chế độ nào hỏng thì bảng vẫn ra: probe_modes() dò trước, dòng của chế độ đó ghi rõ lý do
thay vì ném traceback giữa lần chạy 3 phút.

Self-test bằng dữ liệu giả — không cần DB, không cần model:
    python compare_modes.py --dry
Chạy thật (cần Postgres Up + fulltext_schema.py đã chạy):
    python compare_modes.py
    python compare_modes.py --k 10 --pool 20 --out compare_modes.md
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "keywordSearch"))
sys.path.append(os.path.join(HERE, "..", "hybridRetrieval"))
sys.path.append(os.path.join(HERE, "..", "reranking"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))
sys.path.append(os.path.join(HERE, "..", "..", "week5", "askCli"))

from compare_branches import rank_of_source  # noqa: E402
from metrics import DEFAULT_HIT_K, ModeScore, distinct_source_count  # noqa: E402
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


# Vẫn k=10 như T2 và T3. Đổi k hôm nay là tự tay huỷ baseline của chính mình.
DEFAULT_COMPARE_K = 10

MODE_VECTOR = "① vector-only"
MODE_HYBRID = "② hybrid (RRF)"
MODE_RERANK = "③ hybrid + rerank"
MODE_ORDER = (MODE_VECTOR, MODE_HYBRID, MODE_RERANK)

# Câu hỏi vô hại dùng để dò xem chế độ nào chạy được, trước khi tốn 3 phút chạy cả bộ.
PROBE_QUESTION = "kiểm tra chế độ"
MISSING_RANK = 999


@dataclass
class CaseOutcome:
    """Kết quả của MỘT câu hỏi trên MỘT chế độ.

    rank             : hạng 1-based của đoạn đúng, None = trượt hoặc câu no_answer.
    latency_ms       : đo TRỌN từ chuỗi câu hỏi -> danh sách đã xếp hạng (kể cả embed_query).
    distinct_sources : số file nguồn khác nhau trong top-3 — tín hiệu cho 2 câu no_answer.
    top1_label       : nguồn của kết quả hạng 1, để đọc bảng còn biết nó trả về cái gì.
    """

    case_id: str
    question: str
    expect: str
    query_kind: str
    expect_source: str | None
    mode: str
    rank: int | None
    latency_ms: float
    distinct_sources: int
    top1_label: str

    @property
    def is_answerable(self) -> bool:
        return self.expect == "answerable"


def probe_modes(conn, k: int, candidate_pool: int) -> dict[str, str]:
    """Chạy thử MỖI chế độ đúng 1 lần -> {tên chế độ: "" nếu ok, else lý do hỏng}.

    Ví dụ: {"① vector-only": "", "② hybrid (RRF)": "", "③ hybrid + rerank": "UndefinedColumn: tsv"}

    Chạy cả bộ mất ~3 phút. Chế độ nổ ở câu số 7 tệ hơn nổ ở câu số 1: để lại 6 dòng dữ liệu
    dở dang trông y như thật. Dò 3 lần gọi rồi mới quyết định chạy gì.

    Bắt `Exception` rộng ở đây là CỐ Ý — mục đích của hàm đúng là "thử xem có chết không".
    Nhưng phải ghi lại lý do, không nuốt im lặng: nuốt lỗi là cách biến "DB chưa migrate"
    thành "hybrid không cải thiện gì".
    """

    status: dict[str, str] = {}

    for mode in MODE_ORDER:
        try:
            run_mode(conn, mode, PROBE_QUESTION, k=k, candidate_pool=candidate_pool)
            status[mode] = ""
        except Exception as err:
            status[mode] = f"{type(err).__name__}: {err}"[:90]
            print(f"  ⚠️ {mode} chưa chạy được -> {status[mode]}")
    return status


def run_mode(
    conn,
    mode: str,
    question: str,
    k: int = DEFAULT_COMPARE_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    k_constant: int = RRF_K_CONSTANT,
) -> tuple[list, float]:
    """Chạy 1 câu hỏi qua 1 chế độ -> (danh sách hit đã xếp hạng, latency ms).

    Ví dụ: run_mode(conn, MODE_HYBRID, "Hermes engine là gì?") -> ([HybridHit x10], 143.2)

    Bấm giờ NGAY TRONG hàm này để cả ba chế độ được đo ở cùng một ranh giới: từ chuỗi câu hỏi
    tới danh sách đã xếp hạng, kể cả embed_query. Bỏ embed_query ra ngoài là tặng chế độ
    vector-only ~40ms miễn phí.
    (latency.py của T4 làm việc KHÁC: lặp nhiều lần trên MỘT câu để tách chi phí từng tầng.
     Ở đây là 1 lần trên 8 câu để lấy trung vị — hai câu hỏi khác nhau.)
    """
    # Import LƯỜI: hai module dưới kéo theo sentence_transformers (~5 giây khởi động).
    # Để ở đầu file thì `--dry` — vốn không cần model — cũng phải chờ 5 giây mỗi lần sửa.
    from hybrid_search import hybrid_search
    from retriever import search_chunks  # search_chunks, KHÔNG phải retrieve: retrieve có
                                         # ngưỡng similarity bên trong, nó lọc bớt kết quả
                                         # trước khi mình kịp đo
    from search_pg import embed_query

    started_at = time.perf_counter()

    if mode == MODE_VECTOR:
        hits = search_chunks(conn, embed_query(question), k=k)

    elif mode == MODE_HYBRID:
        hits = hybrid_search(
            conn, question, top_k=k,
            candidate_pool=candidate_pool, k_constant=k_constant,
        )

    elif mode == MODE_RERANK:
        from search_pipeline import search_with_rerank
        # top_k=k (10), KHÔNG phải mặc định top_k=3 của search_with_rerank. Đo bằng 3 thì đoạn
        # đúng ở hạng 7 bị ghi là "trượt" cho riêng chế độ này, trong khi ① ② được đo tới hạng
        # 10 -> MRR của ③ bị chặt cụt CÓ HỆ THỐNG và bảng sẽ nói "rerank phá mọi thứ". Chạy
        # đúng, không lỗi, kết luận sai. Hiển thị 3 kết quả là việc của tầng app, không phải
        # của phép đo.
        reranked, _candidates = search_with_rerank(
            conn, question, top_k=k,
            candidate_pool=candidate_pool, k_constant=k_constant,
        )
        # RerankedHit KHÔNG có `.source`, nó bọc hit gốc trong `.hit`. Bóc ngay tại đây để
        # mọi thứ phía sau chỉ thấy MỘT loại object — thay vì sửa rank_of_source, thứ mà T2,
        # T3 và file này dùng chung.
        hits = [r.hit for r in reranked]

    else:
        raise ValueError(f"chế độ lạ: {mode!r}")

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return hits, elapsed_ms


def evaluate_case(
    conn,
    case: RetrievalCase,
    modes: list[str],
    k: int = DEFAULT_COMPARE_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
) -> list[CaseOutcome]:
    """Chạy 1 câu hỏi qua TẤT CẢ chế độ đang khả dụng -> 1 CaseOutcome cho mỗi chế độ.

    Ví dụ: evaluate_case(conn, case_K1, [MODE_VECTOR, MODE_HYBRID])
           -> [CaseOutcome(mode='① vector-only', rank=7, ...),
               CaseOutcome(mode='② hybrid (RRF)', rank=1, ...)]

    Vòng lặp là "mỗi CÂU chạy hết mọi chế độ", không phải "mỗi chế độ chạy hết mọi câu": ba
    con số của cùng một câu phải sinh ra trong cùng một điều kiện. Chạy hết ① rồi mới sang ②
    thì giữa hai lượt cache Postgres đã ấm lên, và mọi so sánh latency thành vô nghĩa.

    Câu no_answer có expect_source=None nên rank luôn là None. ĐÚNG như vậy — hai câu đó
    không có đoạn đúng nào để tìm; chúng được đo bằng distinct_sources.
    """

    outcomes: list[CaseOutcome] = []

    for mode in modes:
        hits, elapsed_ms = run_mode(
            conn, mode, case.question, k=k, candidate_pool=candidate_pool
        )
        rank = rank_of_source(hits, case.expect_source)
        top_hit = hits[0] if hits else None
        top1_label = getattr(top_hit, "source", "—") if top_hit else "—"

        outcomes.append(CaseOutcome(
            case_id=case.id,
            question=case.question,
            expect=case.expect,
            query_kind=case.query_kind,
            expect_source=case.expect_source,
            mode=mode,
            rank=rank,
            latency_ms=elapsed_ms,
            distinct_sources=distinct_source_count(hits, DEFAULT_HIT_K),
            top1_label=top1_label,
        ))

    return outcomes


def render_mode_table(outcomes: list[CaseOutcome], modes: list[str]) -> str:
    """Bảng markdown 1 dòng / 1 câu hỏi, các chế độ nằm cạnh nhau. Hàm thuần.

    Mong muốn:
        | id | loại | câu hỏi | ① rank | ② rank | ③ rank | ① ms | ② ms | ③ ms | nhận xét |
        |---|---|---|---:|---:|---:|---:|---:|---:|---|
        | K1 | keyword | Chuỗi kết nối MongoDB ở cổng 27017... | 7 | 1 | 1 | 41 | 138 | 402 | ② cứu |
        | N1 | no_answer | Hash mật khẩu bằng Argon2id... | — | — | — | 39 | 141 | 388 | 3 nguồn khác nhau |
    """

    by_case: dict[str, dict[str, CaseOutcome]] = {}
    for out in outcomes:
        by_case.setdefault(out.case_id, {})[out.mode] = out

    rank_headers = " | ".join(f"{m} rank" for m in modes)
    ms_headers = " | ".join(f"{m} ms" for m in modes)
    lines = [
        f"| id | loại | câu hỏi | {rank_headers} | {ms_headers} |",
        "|---|---|---|" + "---:|" * (2 * len(modes)),
    ]
    for case_id, per_mode in by_case.items():
        sample = next(iter(per_mode.values()))
        question = sample.question.replace("|", "/")[:50]
        kind = "no_answer" if not sample.is_answerable else sample.query_kind
        ranks = " | ".join(
            ("—" if per_mode[m].rank is None else str(per_mode[m].rank))
            if m in per_mode else "n/a"
            for m in modes
        )
        millis = " | ".join(
            f"{per_mode[m].latency_ms:.0f}" if m in per_mode else "n/a"
            for m in modes
        )
        lines.append(f"| {case_id} | {kind} | {question} | {ranks} | {millis} |")

    return "\n".join(lines)


def build_scores(outcomes: list[CaseOutcome], modes: list[str]) -> dict[str, ModeScore]:
    """Gom outcome thô -> ModeScore cho từng chế độ.

    Chỗ DUY NHẤT lọc câu answerable ra khỏi MRR. Để việc lọc ở đúng một chỗ, không rải ra
    nhiều hàm — rải ra là kiểu gì cũng có một chỗ quên.
    """
    scores = {mode: ModeScore(mode=mode) for mode in modes}

    for out in outcomes:
        score = scores.get(out.mode)
        if score is None:
            continue
        # latency: lấy CẢ 8 câu (câu no_answer vẫn tốn thời gian đúng như câu thường).
        score.latencies_ms.append(out.latency_ms)
        # hạng: CHỈ 6 câu answerable — xem docstring ModeScore.
        if out.is_answerable:
            score.answerable_ranks.append(out.rank)

    return scores


def render_metric_summary(scores: dict[str, ModeScore], modes: list[str]) -> str:
    """Bảng tổng: mỗi chế độ 1 dòng, MRR · hit@3 · p50 ms. Hàm thuần.

    Mong muốn:
        | chế độ | MRR (6 câu) | hit@3 | p50 latency | ghi chú |
        |---|---:|---:|---:|---|
        | ① vector-only | 0.51 | 0.50 | 41 ms | 1 tầng |
        | ② hybrid (RRF) | 0.72 | 0.83 | 138 ms | +97 ms so với ① |
        | ③ hybrid + rerank | — | — | — | chưa có: UndefinedColumn: tsv |

    Cột "so với ①" quan trọng hơn latency tuyệt đối: "138ms" chẳng nói gì với người đọc CV
    (máy khác, kho khác), còn "+97ms so với vector-only" thì đúng trên mọi máy.

    Chế độ không chạy được vẫn CÓ DÒNG, ghi rõ lý do. Bảng thiếu dòng thì người đọc tưởng chỉ
    có 2 chế độ được xem xét; bảng có dòng "chưa có" thì kể đúng câu chuyện.
    """

    baseline = scores.get(MODE_VECTOR)
    baseline_ms = baseline.p50_ms if baseline and baseline.available else None

    lines = [
        "| chế độ | MRR (6 câu) | hit@3 | p50 latency | ghi chú |",
        "|---|---:|---:|---:|---|",
    ]

    for mode in MODE_ORDER:
        score = scores.get(mode)
        if score is None or not score.available:
            reason = score.unavailable_reason if score else "không chạy"
            lines.append(f"| {mode} | — | — | — | chưa có: {reason} |")
            continue

        note = ""
        if baseline_ms is not None and mode != MODE_VECTOR:
            note = f"{score.p50_ms - baseline_ms:+.0f} ms so với ①"

        lines.append(
            f"| {mode} | {score.mrr:.2f} | {score.hit_rate:.2f} | "
            f"{score.p50_ms:.0f} ms | {note} |"
        )

    return "\n".join(lines)


def find_regressions(
    by_case: dict[str, dict[str, CaseOutcome]],
    before_mode: str,
    after_mode: str,
) -> list[str]:
    """Liệt kê ca mà đoạn đúng TỤT HẠNG khi đi từ before_mode sang after_mode.

    Ví dụ: find_regressions(by_case, MODE_VECTOR, MODE_HYBRID) -> ['K1 (1 -> 6)', 'S1 (2 -> 8)']

    So một chế độ với CHÍNH NÓ luôn trả về rỗng — nhưng cái rỗng đó vô nghĩa, không phải
    "không có ca nào tệ đi". Chặn ngay ở đây để người gọi không diễn giải nhầm.
    """
    if before_mode == after_mode:
        return []

    found: list[str] = []
    for case_id, per_mode in by_case.items():
        before = per_mode.get(before_mode)
        after = per_mode.get(after_mode)
        if not before or not after or not before.is_answerable:
            continue

        rank_before = before.rank if before.rank is not None else MISSING_RANK
        rank_after = after.rank if after.rank is not None else MISSING_RANK
        if rank_after > rank_before:
            # In "—" thay cho 999: người đọc bảng cần thấy "rơi khỏi top-k", không phải một
            # con số ma mình bịa ra để so sánh cho tiện.
            shown = "—" if rank_after >= MISSING_RANK else rank_after
            found.append(f"{case_id} ({rank_before} -> {shown})")
    return found


def print_verdict(scores: dict[str, ModeScore], outcomes: list[CaseOutcome], modes: list[str]) -> str:
    """Kết luận "chế độ nào đáng dùng cho Dự án 1 và VÌ SAO". In ra và trả text để ghi file.

    Đây là phần đi thẳng vào README và vào câu trả lời phỏng vấn. Bảng số ở trên là bằng
    chứng; đoạn này là lập luận.

    Ba câu bắt buộc, đúng thứ tự: (1) chế độ nào MRR/hit@3 cao nhất · (2) nó đắt thêm bao
    nhiêu ms và mức đó có đáng không · (3) nó làm TỆ ĐI ca nào. Bỏ câu 3 là biến bảng đo
    thành quảng cáo — cả file này sinh ra để trả lời câu 3.

    Hàm tổng kết CẢNH BÁO khi kết quả quá đẹp là cố ý: "chế độ phức tạp hơn thắng ở mọi câu"
    thường là dấu hiệu code sai chứ không phải thuật toán giỏi (hay gặp nhất: tầng phức tạp
    lặng lẽ trả về đúng danh sách của tầng đơn giản, nên nó "không bao giờ tệ hơn").
    """

    usable = [m for m in modes if scores.get(m) and scores[m].available]
    if not usable:
        return "❌ Không chế độ nào chạy được — chưa có gì để kết luận."

    lines: list[str] = []
    best = max(usable, key=lambda m: scores[m].mrr)
    lines.append(f"🏆 MRR cao nhất: {best} ({scores[best].mrr:.2f}, hit@3 {scores[best].hit_rate:.2f})")

    baseline = scores.get(MODE_VECTOR)
    if baseline and baseline.available and best != MODE_VECTOR:
        delta_mrr = scores[best].mrr - baseline.mrr
        delta_ms = scores[best].p50_ms - baseline.p50_ms
        lines.append(f"💰 Cái giá: +{delta_ms:.0f} ms để đổi lấy +{delta_mrr:.2f} MRR")
        if delta_mrr < 0.05:
            lines.append("   ⚠️ Chênh MRR dưới 0.05 trên 6 câu = NHIỄU. Không đủ để "
                            "kết luận chế độ nào hơn — cần bộ câu hỏi lớn hơn (tuần 8).")

    regressions = []
    by_case: dict[str, dict[str, CaseOutcome]] = {}
    for out in outcomes:
        by_case.setdefault(out.case_id, {})[out.mode] = out

    if best == MODE_VECTOR:
        # Mốc tự thắng: câu hỏi đúng không còn là "best tệ đi ở đâu" mà là "từng tầng thêm
        # vào tệ đi ở đâu". Đảo chiều so sánh thay vì im lặng bỏ qua.
        lines.append("🥇 Chế độ MỐC thắng — không tầng nào thêm vào có ích trên bộ này.")
        for mode in (m for m in usable if m != MODE_VECTOR):
            regressions = find_regressions(by_case, MODE_VECTOR, mode)
            if regressions:
                lines.append(f"📉 {mode} tệ hơn ① ở: {', '.join(regressions)}")
            else:
                lines.append(f"➖ {mode} không làm tệ đi ca nào, nhưng cũng không thắng ①")
    else:
        regressions = find_regressions(by_case, MODE_VECTOR, best)
        if regressions:
            lines.append(f"📉 Tệ đi ở: {', '.join(regressions)} — ghi vào README, đừng giấu")
        else:
            lines.append("⚠️ KHÔNG có ca nào tệ đi. Nghi ngờ trước khi ăn mừng: kiểm tra "
                         "xem chế độ phức tạp có đang lặng lẽ trả về y hệt chế độ đơn giản "
                         "không (so cột rank của ② và ③ — trùng cả 6 câu là cờ đỏ).")

    # Kết luận phải nêu ĐIỀU KIỆN (dòng ngay trên): nó chỉ đúng trong điều kiện đã đo. Bỏ điều
    # kiện đi thì "hybrid tốt hơn vector" thành tuyên bố phổ quát mà mình không có dữ liệu đỡ.
    text = "\n".join(lines)
    print(text)
    return text


def parse_int_flag(flag: str, default: int) -> int:
    """Đọc `--k 10` từ argv."""
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
    return default


def self_check() -> None:
    """Kiểm tra 3 hàm render bằng dữ liệu GIẢ — không DB, không model."""
    fake_modes = [MODE_VECTOR, MODE_HYBRID]

    def outcome(case_id, expect, kind, mode, rank, ms, sources=1):
        return CaseOutcome(
            case_id=case_id, question=f"câu hỏi {case_id} có ký tự | gây vỡ bảng",
            expect=expect, query_kind=kind, expect_source=None if expect == "no_answer" else "x.pdf",
            mode=mode, rank=rank, latency_ms=ms, distinct_sources=sources,
            top1_label="x.pdf",
        )

    fake_outcomes = [
        outcome("K1", "answerable", "keyword", MODE_VECTOR, 7, 40.0),
        outcome("K1", "answerable", "keyword", MODE_HYBRID, 1, 140.0),
        outcome("S1", "answerable", "semantic", MODE_VECTOR, 1, 42.0),
        outcome("S1", "answerable", "semantic", MODE_HYBRID, 4, 138.0),   # cố ý TỆ ĐI
        outcome("N1", "no_answer", "keyword", MODE_VECTOR, None, 39.0, sources=3),
        outcome("N1", "no_answer", "keyword", MODE_HYBRID, None, 141.0, sources=3),
    ]

    print("── render_mode_table ──")
    print(render_mode_table(fake_outcomes, fake_modes))

    scores = build_scores(fake_outcomes, fake_modes)
    scores[MODE_RERANK] = ModeScore(
        mode=MODE_RERANK, available=False,
        unavailable_reason="UndefinedColumn: cột tsv chưa có",
    )

    print("\n── render_metric_summary ──")
    print(render_metric_summary(scores, fake_modes))

    print("\n── print_verdict ──")
    print_verdict(scores, fake_outcomes, fake_modes)

    # ✅ ĐẠT khi:
    #   1. `python compare_modes.py --dry` chạy hết, không traceback, KHÔNG cần Docker.
    #   2. Bảng đầu: mọi dòng cùng số cột; ký tự `|` trong câu hỏi đã thành `/`.
    #   3. Bảng giữa: có ĐỦ 3 dòng, dòng ③ ghi "chưa có: UndefinedColumn...".
    #   4. Verdict: dòng 📉 liệt kê đúng S1 (1 -> 4), KHÔNG phải K1.
    #   5. MRR của ② tính trên đúng 2 câu (K1, S1) — N1 bị loại. Tự nhẩm: (1.0 + 0.25)/2
    #      = 0.625, bảng in ra `0.62` (f-string `.2f` làm tròn về số chẵn gần nhất, không
    #      phải làm tròn lên như ở trường — biết để khỏi tưởng mình tính sai).
    #      MRR của ① là (1/7 + 1.0)/2 = 0.571 -> in ra `0.57`.


def main() -> None:
    """Chạy thật: dò chế độ -> chạy 8 câu -> in 2 bảng + verdict -> ghi file."""
    if "--dry" in sys.argv:
        print("🧪 compare_modes.py — self-test bằng dữ liệu giả\n")
        self_check()
        return

    k = parse_int_flag("--k", DEFAULT_COMPARE_K)
    candidate_pool = parse_int_flag("--pool", DEFAULT_CANDIDATE_POOL)
    out_path = os.path.join(HERE, "comparison.md")
    if "--out" in sys.argv:
        out_path = os.path.join(HERE, sys.argv[sys.argv.index("--out") + 1])

    if candidate_pool < k:
        raise ValueError(f"pool ({candidate_pool}) phải >= k ({k}) — không thì hybrid thua oan")

    cases, dataset_version = load_questions(DEFAULT_QUESTIONS_PATH)
    # validate_questions TRẢ VỀ danh sách lỗi chứ không ném exception — gọi mà không đọc kết
    # quả là vô nghĩa, bộ câu hỏi hỏng vẫn chạy tiếp và bảng đo vẫn ra số.
    errors = validate_questions(cases)
    if errors:
        print("❌ Bộ câu hỏi có vấn đề, dừng lại trước khi đo:")
        for error in errors:
            print(f"   - {error}")
        return
    print(f"📋 {len(cases)} câu hỏi (bộ {dataset_version}) · k={k} · pool={candidate_pool}\n")

    # Nạp model TRƯỚC vòng lặp: tách chi phí KHỞI ĐỘNG khỏi chi phí XỬ LÝ. Dùng lại hàm
    # có sẵn của T4 thay vì tự viết — nó đã xử lý đúng chuyện warm-up giả (xem latency.py).
    from latency import preload_models
    preload_models()

    with get_conn() as conn:
        print("🔍 Dò chế độ khả dụng...")
        status = probe_modes(conn, k=k, candidate_pool=candidate_pool)
        modes = [m for m in MODE_ORDER if status.get(m) == ""]
        print(f"   -> chạy {len(modes)}/{len(MODE_ORDER)} chế độ\n")

        outcomes: list[CaseOutcome] = []
        for index, case in enumerate(cases, 1):
            print(f"  [{index}/{len(cases)}] {case.id} — {case.question[:45]}...")
            outcomes.extend(evaluate_case(conn, case, modes, k=k, candidate_pool=candidate_pool))

    scores = build_scores(outcomes, modes)
    for mode in MODE_ORDER:
        if mode not in scores:
            scores[mode] = ModeScore(mode=mode, available=False, unavailable_reason=status.get(mode, "?"))

    table = render_mode_table(outcomes, modes)
    summary = render_metric_summary(scores, modes)
    print("\n" + table + "\n\n" + summary + "\n")
    verdict = print_verdict(scores, outcomes, modes)

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"# So 3 chế độ retrieval — bộ câu hỏi {dataset_version}\n\n")
        handle.write(f"k={k} · candidate_pool={candidate_pool} · RRF k={RRF_K_CONSTANT}\n\n")
        handle.write(table + "\n\n" + summary + "\n\n```\n" + verdict + "\n```\n")
    print(f"\n💾 Đã ghi {out_path}")


if __name__ == "__main__":
    main()
