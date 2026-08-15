"""
metrics.py — Hai chỉ số gọn thay cho "cảm giác": MRR và hit@3 (+ latency p50)

    INPUT :  danh sách THỨ HẠNG của đoạn đúng (list[int | None]) + danh sách latency (ms)
    OUTPUT:  MRR · hit@3 · p50 latency  cho MỘT chế độ retrieval

              8 câu hỏi chạy qua 1 chế độ
                        |
        rank_of_source() -> [1, None, 3, 2, 7, 1]   (chỉ 6 câu answerable)
                        |
            +-----------+-----------+-----------+
            |           |           |           |
    reciprocal_rank  hit_at_k   median_ms   distinct_source_count
            |           |           |           |
           MRR        hit@3       p50 ms    tín hiệu cho 2 câu no_answer
            +-----------+-----------+-----------+
                        |
                  ModeScore (1 dòng trong bảng tổng kết)

Vì sao tách file riêng, không viết thẳng trong compare_modes.py:
    Toàn bộ file này là HÀM THUẦN — không DB, không model, không API. Sửa-chạy mất 0.2 giây
    thay vì 40 giây. Mà đây lại đúng là chỗ dễ sai nhất hôm nay: sai công thức MRR thì bảng
    vẫn ra số đẹp, vẫn không báo lỗi, và kết luận "chế độ nào đáng dùng" sẽ sai im lặng.

So với tuần 7 T2/T3:
    T2/T3: so bằng THỨ HẠNG của từng câu, mắt người tự đọc bảng
    hôm nay: gộp 6 thứ hạng thành MỘT con số để so ba chế độ trong một nhịp mắt
             ^^^ mảnh mới. Đây là mầm của golden dataset + LLM-as-judge tuần 8.

Chạy self-test (không cần DB, không cần model):
    python metrics.py
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# hit@3 chứ không hit@1: người dùng đọc được 3 đoạn không thấy mệt, và tầng sinh câu trả lời
# (tuần 5) vốn đã nhận k=3. Đo hit@1 là đo một thứ không ai dùng.
DEFAULT_HIT_K = 3

# Không tìm thấy đoạn đúng thì reciprocal rank = 0. Ghi thành hằng có tên để khi đọc lại
# `RR_WHEN_MISSING` mình nhớ ra đây là một QUYẾT ĐỊNH (coi trượt = 0), không phải mặc định.
RR_WHEN_MISSING = 0.0


@dataclass
class ModeScore:
    """Điểm tổng của MỘT chế độ retrieval trên cả bộ câu hỏi.

    answerable_ranks: thứ hạng của đoạn đúng, CHỈ của 6 câu answerable. None = trượt.
    latencies_ms    : latency của CẢ 8 câu (câu no_answer vẫn tốn thời gian như thường).
    available       : chế độ này có chạy được không. False khi code tầng dưới chưa xong.

    Vì sao answerable_ranks tách riêng khỏi 8 câu: 2 câu no_answer KHÔNG CÓ đoạn đúng, hạng
    của chúng luôn là None. Nhét vào MRR thì cả ba chế độ đều bị kéo xuống một lượng y hệt
    nhau -> MRR trần chỉ còn 0.75 và khoảng cách giữa ba chế độ bị nén lại. Số vẫn ra, không
    ai báo lỗi, và kết luận thì nhạt đi đúng ở chỗ cần rõ nhất.
    """

    mode: str
    answerable_ranks: list[int | None] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    available: bool = True
    unavailable_reason: str = ""

    @property
    def mrr(self) -> float:
        return mean_reciprocal_rank(self.answerable_ranks)

    @property
    def hit_rate(self) -> float:
        return hit_at_k(self.answerable_ranks, DEFAULT_HIT_K)

    @property
    def p50_ms(self) -> float:
        return median_ms(self.latencies_ms)


def reciprocal_rank(rank: int | None) -> float:
    """Thứ hạng của đoạn đúng -> điểm nghịch đảo. Hàm thuần.

    Ví dụ: 1 -> 1.0 · 2 -> 0.5 · 3 -> 0.333 · 10 -> 0.1 · None -> 0.0

    1/rank chứ không phải thang tuyến tính (1 - rank/k): 1/rank phạt rất nặng ở đầu bảng và
    gần như không phân biệt ở cuối, đúng cách người dùng hành xử — tụt 1->2 mất một nửa giá
    trị, tụt 9->10 chẳng ai để ý. Thang tuyến tính coi hai chuyện đó nặng như nhau.

    Trượt = 0.0 chứ KHÔNG bỏ khỏi danh sách: bỏ đi nghĩa là chế độ nào trượt nhiều thì được
    chấm trên ít câu hơn -> chế độ tệ nhất lại có MRR cao nhất.
    """

    if rank is None:
        return RR_WHEN_MISSING

    # rank là 1-BASED (rank_of_source dùng enumerate(hits, 1)). Lỡ truyền chỉ số 0-based thì
    # hạng 2 thành 1/1 = 1.0 — câu trả lời hạng nhì được chấm điểm tuyệt đối, KHÔNG báo lỗi.
    if rank < 1:
        raise ValueError(f"rank phải 1-based, nhận được {rank}")
    return 1.0 / rank


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Trung bình các reciprocal rank -> MỘT con số cho cả chế độ. Hàm thuần.

    Ví dụ: [1, None, 2] -> (1.0 + 0.0 + 0.5) / 3 = 0.5
           []           -> 0.0

    Đọc MRR thành lời: "trung bình, đoạn đúng nằm quanh hạng 1/MRR". 0.5 -> hạng 2.

    `ranks` PHẢI chỉ chứa câu answerable — xem docstring ModeScore. Việc lọc thuộc về người
    gọi (build_scores), nhắc lại ở đây vì ba tuần sau đọc lại chỉ còn cái tên hàm.
    """

    if not ranks:
        return 0.0

    return sum(reciprocal_rank(r) for r in ranks) / len(ranks)


def hit_at_k(ranks: list[int | None], k: int = DEFAULT_HIT_K) -> float:
    """Tỉ lệ câu có đoạn đúng nằm trong top-k. Hàm thuần.

    Ví dụ: ([1, 5, 3, None], k=3) -> 2/4 = 0.5

    Cần CẢ hit@k lẫn MRR vì chúng đo hai thứ: MRR nhạy với mọi thay đổi thứ hạng, hit@k chỉ
    quan tâm trong hay ngoài top-k. Rerank hay tạo ca MRR tăng (7->4) mà hit@3 đứng im — nhìn
    một mình MRR sẽ tưởng người dùng đã cảm nhận được cải thiện.
    """

    if not ranks:
        return 0.0

    # `<=` chứ không `<`: sai dấu bằng thì hit@3 lặng lẽ thành hit@2, mọi chế độ tụt cùng một
    # lượng nên bảng vẫn "hợp lý" — chỉ con số tuyệt đối đi vào README là sai.
    # Lọc None TRƯỚC khi so: `None <= 3` ném TypeError trong Python 3.
    hit_count = sum(1 for r in ranks if r is not None and r <= k)

    return hit_count / len(ranks)


def median_ms(samples: list[float]) -> float:
    """Trung vị latency (ms). Hàm thuần.

    Ví dụ: [120.0, 130.0, 4800.0] -> 130.0  (trung bình sẽ là 1683.3 — vô nghĩa)

    Trung vị vì lần chạy đầu của mỗi chế độ phải nạp model (5-8 giây): một mẫu 5000ms lẫn vào
    7 mẫu 130ms kéo trung bình lên ~740ms, sai gần 6 lần, và con số đó đi thẳng vào README.

    Giới hạn phải ghi kèm khi báo cáo: đây là p50 với n=8. Thứ làm người dùng khó chịu là
    ĐUÔI (p95), mà 8 mẫu thì không tính p95 được.
    """

    if not samples:
        return 0.0

    return statistics.median(samples)


def distinct_source_count(hits: list, top_k: int = DEFAULT_HIT_K) -> int:
    """Đếm số FILE NGUỒN khác nhau trong top_k kết quả. Hàm thuần.

    Ví dụ: top-3 đều từ 'Frontend Interview Prep.docx'        -> 1  (kho đồng thuận)
           top-3 từ 3 file khác nhau                          -> 3  (kho không có gì để nói)

    Đây là cách đo 2 câu no_answer mà không nói dối: ở tầng retrieval chưa có câu trả lời nào
    được sinh ra nên KHÔNG THỂ đo "có bịa không" — bịa là việc của LLM. Thứ đo được là
    pipeline có đưa ra TÍN HIỆU để tầng app từ chối trả lời hay không.

    Vì sao đếm nguồn: câu có trong tài liệu thì các đoạn liên quan cụm lại trong 1-2 file;
    câu không có thì mỗi file góp một đoạn hao hao -> top-3 tán loạn. Đếm được vì nó là số
    nguyên — so ĐIỂM giữa ba chế độ thì không, cosine/ts_rank/logit là ba thang khác nhau.

    Tín hiệu THÔ, n=2 câu. Không đủ để viết "phát hiện được câu không trả lời được".
    """

    top = hits[:top_k]
    sources = set()
    for hit in top:
        # Hai lớp: RerankedHit không có `.source`, nó bọc hit gốc trong `.hit`.
        source = getattr(hit, "source", None) or getattr(getattr(hit, "hit", None), "source", None)
        if source:
            sources.add(source.lower())

    return len(sources)


# ══════════════════════════════════════════════════════════════════════════════
# Self-test bằng dữ liệu giả — không cần DB, không cần model, không cần mạng
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FakeHit:
    """Hit giả chỉ có đúng thứ cần để test distinct_source_count."""
    source: str


def self_check() -> None:
    """Chạy hết các phép thử nhỏ, in ✅/❌ từng dòng."""
    checks: list[tuple[str, bool]] = []

    def check(label: str, actual, expected) -> None:
        ok = actual == expected or (
            isinstance(actual, float) and isinstance(expected, float)
            and abs(actual - expected) < 1e-9
        )
        checks.append((f"{label}: {actual!r} (mong {expected!r})", ok))

    check("RR hạng 1", reciprocal_rank(1), 1.0)
    check("RR hạng 2", reciprocal_rank(2), 0.5)
    check("RR trượt", reciprocal_rank(None), 0.0)

    check("MRR [1, None, 2]", mean_reciprocal_rank([1, None, 2]), 0.5)
    check("MRR rỗng", mean_reciprocal_rank([]), 0.0)
    check("MRR toàn trượt", mean_reciprocal_rank([None, None]), 0.0)

    check("hit@3 biên (hạng 3 tính là TRÚNG)", hit_at_k([3], 3), 1.0)
    check("hit@3 [1,5,3,None]", hit_at_k([1, 5, 3, None], 3), 0.5)
    check("hit@3 rỗng", hit_at_k([], 3), 0.0)

    check("median bỏ qua ngoại lai", median_ms([120.0, 130.0, 4800.0]), 130.0)
    check("median rỗng", median_ms([]), 0.0)

    same_source = [FakeHit("frontend.docx"), FakeHit("frontend.docx"), FakeHit("frontend.docx")]
    spread = [FakeHit("a.md"), FakeHit("b.pdf"), FakeHit("c.docx")]
    check("nguồn tập trung", distinct_source_count(same_source), 1)
    check("nguồn tán loạn", distinct_source_count(spread), 3)
    check("hits rỗng", distinct_source_count([]), 0)

    # Ca đáng giá nhất: MRR tăng mà hit@3 đứng im — đúng thứ rerank hay tạo ra.
    before = [7, 2, 5]
    after = [4, 1, 5]
    moved = mean_reciprocal_rank(after) > mean_reciprocal_rank(before)
    flat = hit_at_k(after, 3) == hit_at_k(before, 3)
    checks.append(("MRR tăng nhưng hit@3 đứng im (ca kinh điển của rerank)", moved and flat))

    for label, ok in checks:
        print(f"  {'✅' if ok else '❌'} {label}")

    failed = [label for label, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} phép thử đạt")
    if failed:
        print("❌ Chưa xong:", failed[0])

    # ✅ ĐẠT khi:
    #   1. Cả 15 dòng đều ✅ (chạy `python metrics.py`, không cần bật Docker/Postgres).
    #   2. reciprocal_rank(0) ném ValueError chứ không trả về số — tự thử trên REPL.
    #   3. Dòng cuối "MRR tăng nhưng hit@3 đứng im" ✅ — nghĩa là hai chỉ số THẬT SỰ đo hai
    #      thứ khác nhau. Nếu dòng này ❌ thì một trong hai hàm đang tính sai.


if __name__ == "__main__":
    print("🧪 metrics.py — self-test bằng dữ liệu giả\n")
    self_check()
