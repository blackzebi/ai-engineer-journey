"""
latency.py — ĐO CÁI GIÁ của từng tầng, bằng mili-giây, trên cùng 8 câu hỏi

    INPUT :  questions.json (8 câu, dùng lại nguyên si của T2)
    OUTPUT:  bảng markdown 4 tầng CỘNG DỒN + phần chênh lệch mà mỗi tầng thêm vào

        tầng 1  vector             embed_query + search_chunks
        tầng 2  + keyword          thêm search_by_keyword          } = hybrid_search
        tầng 3  + fusion           thêm fuse_rankings              }
        tầng 4  + rerank           thêm cross-encoder trên 20 cặp

        mỗi tầng CHỨA tầng trước -> cột "thêm" = tầng n − tầng n−1 chính là cái giá THẬT
        của riêng mảnh mới. Đo rời từng mảnh rồi cộng lại là con số khác (bỏ sót chi phí
        ghép nối, và đếm hai lần phần embed).

SỐ ĐO ĐƯỢC 12/08 (8 câu · repeat=5 · pool=20 · CPU):
        vector      73.0 ms   |  + keyword  79.9 ms  (+6.9)
        + fusion    80.7 ms   (+0.8)       |  + rerank  1599.3 ms  (+1518.7)
    Hai con số đáng nhớ: fusion gần như MIỄN PHÍ (nó là vài phép chia trên 40 số), còn rerank
    đắt gấp ~19 lần toàn bộ phần retrieval cộng lại. Đó là bài học kiến trúc của cả tuần,
    gói gọn trong một cột.

VÌ SAO NGÀY NÀY QUAN TRỌNG HƠN NÓ TRÔNG CÓ VẺ:
    "Tôi thêm reranker, kết quả tốt hơn" — ai cũng nói được, không kiểm chứng được.
    "Rerank 20 ứng viên thêm ~1519ms trung vị trên CPU, p95 1813ms, và trên bộ 8 câu tiếng
     Việt nó không cải thiện ca nào" — đây là câu của kỹ sư: có số, có điều kiện đo, và
    trung thực cả về phần không đẹp.

BA QUY TẮC ĐO — vi phạm một cái là cả bảng vô nghĩa:
    1. WARM-UP rồi mới lấy số. Lần chạy đầu gánh: load model (~5 giây), Postgres nạp page từ
       disk vào shared_buffers, Python import module. Đo lần đầu là đo thời gian KHỞI ĐỘNG
       chứ không phải thời gian XỬ LÝ. (Đã gặp đúng bài này ở benchmark_query tuần 5.)
    2. TRUNG VỊ chứ không trung bình. Một lần GC hoặc một lần OS scheduler ngắt là đủ tạo ra
       giá trị gấp 5 lần; trung bình bị nó kéo đi, trung vị thì không.
    3. GHI ĐIỀU KIỆN ĐO kèm số: số dòng bảng chunks, pool, top_k, máy nào, đã warm-up chưa.
       Thiếu nó thì 2 tuần sau con số vô dụng — quy trình rút ra từ tuần 5.

Self-test bằng hàm giả (time.sleep) — không cần DB, không cần model:
    python latency.py --dry
Chạy thật (cần Postgres Up + fulltext_schema.py đã chạy):
    python latency.py --repeat 5 --out latency.md
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "hybridRetrieval"))
sys.path.append(os.path.join(HERE, "..", "keywordSearch"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))
sys.path.append(os.path.join(HERE, "..", "..", "week5", "askCli"))

from rerank import DEFAULT_RERANK_POOL, DEFAULT_TOP_K, rerank_hits  # noqa: E402
from rrf import BRANCH_KEYWORD, BRANCH_VECTOR, RRF_K_CONSTANT, fuse_rankings  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Số lần chạy LẤY SỐ cho mỗi (tầng, câu hỏi). 3 là tối thiểu để trung vị có nghĩa; 5 ổn định
# hơn. 8 câu × 4 tầng × 5 lần = 160 lượt, trong đó 40 lượt chạy cross-encoder — vài phút.
DEFAULT_REPEAT = 3

# Số lần chạy VỨT ĐI trước khi lấy số.
DEFAULT_WARMUP = 1

STAGE_VECTOR = "vector"
STAGE_KEYWORD = "+ keyword"
STAGE_FUSION = "+ fusion (RRF)"
STAGE_RERANK = "+ rerank (cross-encoder)"

# Thứ tự CỘNG DỒN. Đổi thứ tự ở đây là đổi ý nghĩa cột "thêm".
STAGE_ORDER = (STAGE_VECTOR, STAGE_KEYWORD, STAGE_FUSION, STAGE_RERANK)


@dataclass
class StageTiming:
    """Số đo của MỘT tầng trên toàn bộ bộ câu hỏi.

    samples_ms: mọi lần đo của mọi câu hỏi, gộp chung. Giữ RAW chứ không chỉ giữ trung vị —
                để lát nữa muốn tính p95 hay vẽ phân bố thì không phải chạy lại 4 phút.
    """

    name: str
    samples_ms: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms) if self.samples_ms else 0.0

    @property
    def p95_ms(self) -> float:
        """Phân vị 95: 5% lần chậm nhất nằm trên mức này.

        Người dùng cảm nhận được p95, không cảm nhận được trung vị — trang web "thỉnh thoảng
        lag" chính là p95 xấu.
        """
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        # index = ceil(0.95*n) - 1, kẹp trong [0, n-1]. Với n nhỏ (40 mẫu) đây là xấp xỉ thô
        # — ghi rõ trong note là "p95 trên 40 mẫu", đừng trình bày như số liệu production.
        index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999) - 1))
        return ordered[index]


def time_call(fn: Callable[[], object], repeat: int = DEFAULT_REPEAT,
              warmup: int = DEFAULT_WARMUP) -> list[float]:
    """Chạy `fn` (warmup + repeat) lần, VỨT phần warmup, trả list mili-giây của phần còn lại.

    Ví dụ: time_call(lambda: search(conn, "câu hỏi"), repeat=3, warmup=1)
           -> [42.1, 39.8, 40.5]      (4 lần chạy, bỏ lần đầu)

    `fn` không nhận tham số nào: hàm đo hoàn toàn không biết gì về thứ nó đang đo, chỗ gọi bọc
    sẵn mọi thứ vào một `lambda`. Nhờ vậy self_check đo được `time.sleep` mà không cần Postgres.

    Warm-up nằm TRONG hàm này (không gọi riêng bên ngoài) vì quên warm-up là lỗi im lặng: kết
    quả vẫn ra, chỉ là cao gấp trăm lần ở dòng đầu tiên.

    `perf_counter` chứ không `time.time`: time.time là đồng hồ treo tường, có thể NHẢY LÙI khi
    hệ điều hành đồng bộ NTP giữa lúc đang đo -> thời lượng ÂM.
    """
    for _ in range(warmup):
        fn()

    duration_ms = []
    for _ in range(repeat):
        started_at = time.perf_counter()
        # Giữ kết quả vào biến: nếu fn lỡ trả generator thì không có gì được tính, số đo ra
        # ~0.001ms — con số đó ở bảng phải làm mình giật mình đi kiểm.
        _result = fn()
        # perf_counter trả GIÂY. Quên ×1000 thì mọi kết luận lệch 1000 lần, mà "42.1" nhìn
        # rất hợp lý nên không ai nghi ngờ.
        duration_ms.append((time.perf_counter() - started_at) * 1000)

    return duration_ms


def measure_stages(
    stages: list[tuple[str, Callable[[str], object]]],
    questions: list[str],
    repeat: int = DEFAULT_REPEAT,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, StageTiming]:
    """Đo MỌI tầng trên MỌI câu hỏi -> {tên tầng: StageTiming}.

    Ví dụ: measure_stages([("vector", run_vector)], ["câu 1", "câu 2"], repeat=3)
           -> {"vector": StageTiming(samples_ms=[...6 số...])}

    Đo TẤT CẢ tầng trong CÙNG một lần chạy vì latency phụ thuộc trạng thái máy (nhiệt độ CPU,
    cache Postgres, app khác đang chạy). So số tầng 4 đo lúc 14:00 với tầng 1 đo lúc 11:00 là
    so hai điều kiện khác nhau.

    Vòng NGOÀI là câu hỏi, vòng TRONG là tầng: mọi tầng của cùng một câu chia sẻ cùng trạng
    thái cache -> so sánh giữa các tầng công bằng nhất. Đảo lại thì tầng đo sau cùng luôn
    được hưởng cache ấm nhất, tức được ưu ái một cách có hệ thống.
    """
    timings = {name: StageTiming(name=name) for name, _ in stages}

    for index, question in enumerate(questions, 1):
        # In mỗi câu hỏi một dòng thôi: 160 lượt mà in mỗi lượt thì chính việc in ra terminal
        # (I/O thật) cũng làm nhiễu số đo.
        print(f"  [{index}/{len(questions)}] {question[:45]}...")

        for name, run in stages:
            # `lambda: run(question)` an toàn vì được gọi NGAY trong cùng vòng lặp. Nếu có lúc
            # nào gom một list lambda lại rồi mới chạy sau, mọi lambda sẽ cùng thấy giá trị
            # CUỐI của `question` (late binding) — khi đó phải viết `lambda q=question: run(q)`.
            samples = time_call(lambda: run(question), repeat=repeat, warmup=warmup)
            timings[name].samples_ms.extend(samples)

    return timings


def render_latency_table(timings: dict[str, StageTiming], stage_order=STAGE_ORDER) -> str:
    """Bảng markdown cộng dồn + cột "tầng này THÊM bao nhiêu". Hàm thuần.

    Mong muốn:
        | tầng | trung vị (ms) | p95 (ms) | tầng này thêm (ms) |
        |---|---:|---:|---:|
        | vector | 73.0 | 79.0 | — |
        | + rerank (cross-encoder) | 1599.3 | 1813.0 | +1518.7 |

    Cột "thêm" mới là cột đáng đọc nhất: cột trung vị chỉ nói tổng thời gian (ai cũng đoán
    được là tăng dần), cột "thêm" trả lời câu hỏi thật — mảnh nào ĐẮT.

    Chênh lệch có thể ÂM (fusion rẻ tới mức chìm dưới nhiễu đo). Đừng "sửa" bằng trị tuyệt
    đối — số âm nhỏ là thông tin thật.
    """
    lines = [
        "| tầng | trung vị (ms) | p95 (ms) | tầng này thêm (ms) |",
        "|---|---:|---:|---:|",
    ]

    # Đi theo STAGE_ORDER, KHÔNG lặp theo dict: thứ tự cộng dồn là ý nghĩa của bảng, để nó
    # phụ thuộc thứ tự chèn vào dict là mời rắc rối.
    previous_median = None
    for name in stage_order:
        timing = timings.get(name)
        if timing is None:
            continue

        # Tầng ĐẦU không có tầng trước để trừ -> "—". In "0.0" là nói dối: nó có chi phí, chỉ
        # là không so được với ai.
        delta = "—" if previous_median is None else f"{timing.median_ms - previous_median:+.1f}"
        lines.append(
            f"| {name} | {timing.median_ms:.1f} | {timing.p95_ms:.1f} | {delta} |"
        )
        previous_median = timing.median_ms

    return "\n".join(lines)


def print_latency_verdict(timings: dict[str, StageTiming]) -> str:
    """Đọc bảng latency thành câu người-nói-được, kèm cảnh báo khi số VÔ LÝ. Trả về text.

    Hàm tổng kết đi CẢNH BÁO thay vì chỉ tóm tắt, vì số latency sai trông y hệt số đúng.
    Hai ca sai hay gặp nhất:
      (a) rerank thêm < 20ms -> gần như chắc chắn cross-encoder KHÔNG chạy thật (pool đang là
          3 thay vì 20, hoặc scorer giả bị truyền vào).
      (b) tầng đầu > 2000ms  -> đang đo cả thời gian LOAD MODEL, warm-up chưa ăn.

    ⚠️ Đọc kỹ điều kiện của cảnh báo (b) ở dưới trước khi tin nó — xem note T4 tuần 7.

    Lưu ý khi diễn giải: đắt hay rẻ phụ thuộc NGÂN SÁCH. RAG hỏi-đáp mà LLM đã mất 3-5 giây
    sinh câu trả lời thì +200ms là 5%; autocomplete ngân sách 50ms thì bất khả thi. Và con số
    không kèm CPU/GPU thì không so được với ai — cùng model, GPU nhanh hơn 10-30 lần.
    """
    first = timings.get(STAGE_VECTOR)
    fusion = timings.get(STAGE_FUSION)
    rerank = timings.get(STAGE_RERANK)
    lines = ["=== ĐỌC BẢNG LATENCY ==="]

    if fusion and rerank:
        added = rerank.median_ms - fusion.median_ms
        lines.append(
            f"  Rerank {DEFAULT_RERANK_POOL} ứng viên thêm {added:.0f} ms (trung vị), "
            f"đưa tổng từ {fusion.median_ms:.0f} ms lên {rerank.median_ms:.0f} ms."
        )
        lines.append(f"  p95 tổng: {rerank.p95_ms:.0f} ms.")

        if added < 20:
            lines.append("  ⚠️ Rerank thêm < 20ms — nghi cross-encoder KHÔNG chạy thật.")
            lines.append("     Kiểm: len(candidates) có ~20 không? scorer giả có lọt vào không?")

    if first and first.median_ms > 2000:
        lines.append(f"  ⚠️ Tầng đầu {first.median_ms:.0f} ms — đang đo cả thời gian LOAD")
        lines.append("     MODEL. Tăng warmup, hoặc gọi preload_models() trước khi đo.")

    lines.append("")
    lines.append("  📋 Chép vào note KÈM: số dòng bảng chunks · pool · top_k · CPU hay GPU")
    lines.append("     · đã warm-up · repeat=? — thiếu một mục là số này hết so được.")

    text = "\n".join(lines)
    print("\n" + text)
    return text


def preload_models() -> None:
    """Nạp SẴN cả hai model trước khi đo.

    Warm-up trong time_call đã bỏ lần chạy đầu, nhưng lần chạy ĐÓ vẫn tốn 5-10 giây và làm
    mình tưởng chương trình treo. Gọi tường minh ở đây để (1) biết chính xác lúc nào model
    sẵn sàng, (2) tách bạch chi phí KHỞI ĐỘNG khỏi chi phí XỬ LÝ — hai con số khác nhau và
    cả hai đều đáng ghi vào note.
    """
    from rerank import get_cross_encoder
    from search_pg import embed_query

    started_at = time.perf_counter()
    embed_query("khởi động")
    embed_seconds = time.perf_counter() - started_at

    started_at = time.perf_counter()
    # Không gọi rerank_hits([]) — nó trả [] ngay và KHÔNG chạm tới model, tức là warm-up giả.
    # Phải chấm một cặp thật thì trọng số mới được nạp và đồ thị mới được dựng.
    get_cross_encoder().predict([("khởi động", "một đoạn văn bản bất kỳ")])
    cross_seconds = time.perf_counter() - started_at

    print(f"🔥 Nạp model: bi-encoder {embed_seconds:.1f}s · cross-encoder {cross_seconds:.1f}s")
    print("   (đây là CHI PHÍ KHỞI ĐỘNG — ghi vào note, nó là lý do phải lazy load 1 lần "
          "chứ không load mỗi request)\n")


def build_stages(conn, candidate_pool: int, top_k: int) -> list[tuple[str, Callable[[str], object]]]:
    """Dựng 4 tầng CỘNG DỒN, mỗi tầng là một closure nhận câu hỏi.

    Cố ý KHÔNG gọi `hybrid_search()` cho tầng 2/3 mà viết lại từng mảnh: chỉ có cách đó mới
    chèn được điểm dừng vào GIỮA hybrid_search (sau keyword, trước fusion). Cái giá phải trả
    là hai đoạn code song song — sửa hybrid_search thì phải sửa cả đây. Đánh đổi có ý thức.
    """
    from keyword_search import DEFAULT_MODE, search_by_keyword
    from retriever import search_chunks      # search_chunks, KHÔNG phải retrieve
    from search_pg import embed_query

    def run_vector(question: str):
        query_vector = embed_query(question)
        return search_chunks(conn, query_vector, k=candidate_pool)

    def run_keyword(question: str):
        query_vector = embed_query(question)
        vector_hits = search_chunks(conn, query_vector, k=candidate_pool)
        keyword_hits = search_by_keyword(conn, question, k=candidate_pool, mode=DEFAULT_MODE)
        return vector_hits, keyword_hits

    def run_fusion(question: str):
        vector_hits, keyword_hits = run_keyword(question)
        ranked_ids_by_branch = {
            BRANCH_VECTOR: [h.id for h in vector_hits],
            BRANCH_KEYWORD: [h.id for h in keyword_hits],
        }
        fused = fuse_rankings(ranked_ids_by_branch, k_constant=RRF_K_CONSTANT,
                              top_k=candidate_pool)
        chunk_by_id = {h.id: h for h in keyword_hits}
        chunk_by_id.update({h.id: h for h in vector_hits})
        return [chunk_by_id[r.chunk_id] for r in fused if r.chunk_id in chunk_by_id]

    def run_rerank(question: str):
        candidates = run_fusion(question)
        return rerank_hits(question, candidates, top_k=top_k)

    return [
        (STAGE_VECTOR, run_vector),
        (STAGE_KEYWORD, run_keyword),
        (STAGE_FUSION, run_fusion),
        (STAGE_RERANK, run_rerank),
    ]


def self_check() -> None:
    """Assert bằng time.sleep — không cần DB, không cần model. Chạy khoảng 1 giây."""

    samples = time_call(lambda: time.sleep(0.02), repeat=3, warmup=1)
    assert len(samples) == 3, f"phải trả đúng `repeat` mẫu, đang có {len(samples)}"
    assert all(15 < s < 60 for s in samples), (
        f"mỗi mẫu phải quanh 20 MILI-GIÂY, đang là {samples}. "
        "Ra ~0.02 = quên nhân 1000. Ra ~20000 = nhân nhầm hai lần."
    )
    assert time_call(lambda: None, repeat=2, warmup=0) is not None

    # Đếm số lần gọi thật: warmup PHẢI chạy nhưng KHÔNG được nằm trong kết quả
    call_count = 0

    def counted():
        nonlocal call_count      # nonlocal: sửa biến của hàm BAO NGOÀI (self_check); nếu là
        call_count += 1          # biến module thì phải dùng `global`

    time_call(counted, repeat=3, warmup=2)
    assert call_count == 5, f"phải gọi warmup+repeat = 5 lần, đang gọi {call_count}"

    # Bộ giả có tầng rerank RẺ BẤT THƯỜNG (chỉ thêm ~10ms so với fusion) để kiểm rằng
    # print_latency_verdict biết NGHI NGỜ. Chạy thật thì con số này phải hàng trăm ms.
    fake_stages = [
        (STAGE_VECTOR, lambda q: time.sleep(0.005)),
        (STAGE_KEYWORD, lambda q: time.sleep(0.010)),
        (STAGE_FUSION, lambda q: time.sleep(0.010)),
        (STAGE_RERANK, lambda q: time.sleep(0.020)),
    ]
    timings = measure_stages(fake_stages, ["câu 1", "câu 2"], repeat=2, warmup=1)
    assert set(timings) == {s[0] for s in fake_stages}, "phải có đủ 4 tầng"
    assert len(timings[STAGE_VECTOR].samples_ms) == 4, "2 câu × repeat 2 = 4 mẫu mỗi tầng"
    assert timings[STAGE_RERANK].median_ms > timings[STAGE_VECTOR].median_ms, \
        "tầng cộng dồn sau phải chậm hơn tầng trước"

    table = render_latency_table(timings)
    assert table.count("\n") == 5, "2 dòng header + 4 tầng"
    assert "| — |" in table, "tầng đầu không có cột 'thêm', phải là dấu gạch"
    assert "+" in table, "các tầng sau phải hiện chênh lệch có dấu"

    verdict = print_latency_verdict(timings)
    assert "⚠️" in verdict, ("bộ giả có rerank chỉ thêm ~10ms (< ngưỡng 20ms) nên verdict "
                            "PHẢI cảnh báo 'nghi cross-encoder không chạy thật'")
    assert "📋" in verdict, "phải nhắc ghi điều kiện đo"

    print("\n✅ self_check: time_call + measure_stages + render_latency_table + verdict pass")
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

    repeat = parse_int_flag("--repeat", DEFAULT_REPEAT)
    candidate_pool = parse_int_flag("--pool", DEFAULT_RERANK_POOL)
    top_k = parse_int_flag("--k", DEFAULT_TOP_K)

    from questions import DEFAULT_QUESTIONS_PATH, load_questions
    from vector_ops import get_conn

    cases, corpus_note = load_questions(DEFAULT_QUESTIONS_PATH)
    questions = [c.question for c in cases]

    print(f"⏱  Đo 4 tầng cộng dồn · {len(questions)} câu · repeat={repeat} · "
          f"pool={candidate_pool} · top_k={top_k}\n")
    preload_models()

    with get_conn() as conn:
        stages = build_stages(conn, candidate_pool=candidate_pool, top_k=top_k)
        timings = measure_stages(stages, questions, repeat=repeat, warmup=DEFAULT_WARMUP)

    table = render_latency_table(timings)
    print("\n" + table)
    verdict = print_latency_verdict(timings)

    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Latency 4 tầng: vector -> +keyword -> +fusion -> +rerank\n\n")
            f.write(f"**Điều kiện đo**: {len(questions)} câu · repeat={repeat} · "
                    f"warmup={DEFAULT_WARMUP} · pool={candidate_pool} · top_k={top_k} · "
                    f"rrf_k={RRF_K_CONSTANT} · cross-encoder/ms-marco-MiniLM-L-6-v2 · CPU\n\n")
            f.write("**TODO điền tay**: số dòng bảng chunks = ? · máy = ?\n\n")
            f.write(f"Corpus: {corpus_note[:200]}...\n\n")
            f.write(table + "\n\n```\n" + verdict + "\n```\n")
        print(f"\n💾 Đã ghi {out_path}")


if __name__ == "__main__":
    main()

# Kỳ vọng khi chạy:
#   `--dry`   : in "pass" + bảng 4 dòng, khoảng 1 giây
#   chạy thật : cột "thêm" của fusion là số RẤT NHỎ (dưới ~2ms) — lớn hơn nghĩa là
#               fuse_rankings đang làm gì đó ngoài số học
#               cột "thêm" của rerank là hàng trăm ms — dưới 20ms = cross-encoder chưa chạy thật
