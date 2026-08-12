"""
rrf.py — Reciprocal Rank Fusion: hợp nhất 2 bảng xếp hạng KHÔNG CÙNG THANG ĐO

    INPUT :  nhiều danh sách id đã xếp hạng (mỗi nhánh một danh sách)
    OUTPUT:  một danh sách id duy nhất, kèm điểm RRF và thứ hạng gốc ở từng nhánh

        vector: [17, 42,  8, 91, ...]        keyword: [42,  5, 17, 63, ...]
                 |   |    |   |                        |   |   |   |
                 └───┴────┴───┴──── rank 1,2,3,4 ──────┴───┴───┴───┘
                                       |
                            score(id) = Σ  1 / (k_constant + rank)
                                     nhánh nào có id đó
                                       |
                        sắp giảm dần -> [42, 17, 5, 8, ...]
                                          ^^ id 42 lên đầu vì XUẤT HIỆN Ở CẢ HAI
                                             nhánh, dù không phải hạng 1 ở nhánh nào

So với tuần 7 T2 (compare_branches.py):
    T2:      hai bảng SONG SONG, chỉ để NHÌN ai thắng ở đâu
    hôm nay: hai bảng -> MỘT bảng dùng được thật
                         ^^^^ mảnh mới nằm gọn trong file này, và nó KHÔNG chạm DB
                              một dòng nào — đó là lý do nó test được trong 1 giây

VÌ SAO KHÔNG CỘNG THẲNG ĐIỂM ĐƯỢC:
    similarity (cosine) nằm trong [-1, 1] và có ý nghĩa TUYỆT ĐỐI: 0.62 luôn là "khá giống",
    bất kể kho tài liệu nào.
    ts_rank KHÔNG có thang cố định: phụ thuộc số từ khớp, độ dài chunk, cờ chuẩn hoá, trọng
    số setweight. Trong kho này điểm ts_rank cao nhất chỉ cỡ 0.0x — mà đó đã là kết quả TỐT
    NHẤT.
    => 0.62 + 0.08 = 0.70 là phép cộng giữa mét và kilôgam. Nhánh vector áp đảo mọi quyết
       định, nhánh keyword coi như không tồn tại — nhưng code vẫn chạy, bảng vẫn ra, không
       một dòng lỗi. Bug im lặng kinh điển.
    RRF né hoàn toàn chuyện đó: nó VỨT BỎ điểm số, chỉ giữ THỨ HẠNG. Hạng 1 là hạng 1, ở
    nhánh nào cũng vậy — đơn vị chung có sẵn, không cần chuẩn hoá gì cả.

Self-test bằng dữ liệu giả — không cần DB, không cần model, chạy tức thì:
    python rrf.py
    python rrf.py --k-effect        # xem k_constant lớn/nhỏ đổi kết quả thế nào
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Tên nhánh. Dùng hằng chứ không gõ chuỗi "vector" rải rác: gõ nhầm ở một chỗ thì dict lookup
# trả None, điểm RRF thiếu một nhánh, và KHÔNG có lỗi nào được ném ra.
BRANCH_VECTOR = "vector"
BRANCH_KEYWORD = "keyword"

# Hằng k của RRF (Cormack 2009). Nó làm PHẲNG chênh lệch giữa các hạng đầu:
#     k=60 : hạng 1 -> 1/61 = 0.016393 · hạng 2 -> 1/62 = 0.016129  (chênh 1.61%)
#     k=1  : hạng 1 -> 1/2  = 0.5      · hạng 2 -> 1/3  = 0.333     (chênh 33.33%)
# k lớn = "không tin tuyệt đối vào thứ hạng của từng nhánh, ai được NHIỀU nhánh tiến cử thì
# thắng". k nhỏ = "hạng 1 của một nhánh là thiêng liêng". Hai nhánh ở repo này lệch chất
# lượng (keyword chỉ tìm ra tài liệu đúng ở 4/8 câu) nên k lớn an toàn hơn: nó không cho
# nhánh yếu quyền phủ quyết chỉ vì nó tình cờ xếp cái gì đó hạng 1.
RRF_K_CONSTANT = 60

# Lấy bao nhiêu ứng viên từ MỖI nhánh trước khi fusion. Phải LỚN HƠN top-k cuối cùng: chunk
# hạng 12 ở CẢ HAI nhánh thường tốt hơn chunk hạng 1 ở đúng một nhánh, nhưng pool hẹp thì nó
# không bao giờ vào được vòng trong. Lấy hẹp là tự tay vứt đúng cái mà hybrid sinh ra để cứu.
DEFAULT_CANDIDATE_POOL = 20

DEFAULT_TOP_K = 5


@dataclass
class FusionResult:
    """1 chunk sau khi hợp nhất.

    chunk_id  : id trong bảng chunks
    rrf_score : tổng Σ 1/(k_constant + rank) trên các nhánh CÓ chứa id này
    ranks     : {'vector': 3, 'keyword': None} — thứ hạng gốc 1-based ở từng nhánh,
                None nghĩa là nhánh đó KHÔNG tiến cử chunk này trong pool của nó.

    Giữ nguyên `ranks` thay vì chỉ trả điểm vì đây là thứ duy nhất cho phép debug "vì sao
    chunk này lên top". Bỏ đi thì hybrid thành hộp đen tự tay dựng.
    """

    chunk_id: int
    rrf_score: float
    ranks: dict[str, int | None] = field(default_factory=dict)

    def rank_in(self, branch: str) -> int | None:
        """Thứ hạng ở một nhánh, None nếu nhánh đó không tiến cử."""
        return self.ranks.get(branch)

    def found_in_branches(self) -> int:
        """Số nhánh có tiến cử chunk này. 2 = cả hai nhánh đồng ý."""
        return sum(1 for r in self.ranks.values() if r is not None)

    def explain(self) -> str:
        """Một dòng người-đọc-được: 'v=3 k=1 (2 nhánh)' hoặc 'v=7 k=— (1 nhánh)'."""
        parts = []
        for branch, rank in self.ranks.items():
            parts.append(f"{branch[0]}={rank if rank is not None else '—'}")
        return f"{' '.join(parts)} ({self.found_in_branches()} nhánh)"


def reciprocal_rank_score(rank: int, k_constant: int = RRF_K_CONSTANT) -> float:
    """Điểm RRF của MỘT thứ hạng ở MỘT nhánh: 1 / (k_constant + rank).

    Ví dụ: reciprocal_rank_score(1)  -> 0.016393...   (1/61)
           reciprocal_rank_score(2)  -> 0.016129...   (1/62)
           reciprocal_rank_score(1, k_constant=1) -> 0.5

    Vì sao có k_constant trong mẫu số mà không phải 1/rank trần trụi: 1/rank cho hạng 1 = 1.0
    và hạng 2 = 0.5 — chênh gấp đôi, nên một chunk hạng 1 ở nhánh yếu đè bẹp một chunk hạng 2
    ở CẢ HAI nhánh. k=60 kéo mọi hạng đầu về gần nhau, "được nhiều nhánh tiến cử" mới thắng
    được "hạng 1 của một nhánh" — đúng thứ mình muốn mua khi làm hybrid.
    """
    # rank là 1-based. Guard này chặn hai thứ: k_constant=0 gây ZeroDivisionError, và quan
    # trọng hơn là ai đó "chữa" rank=None bằng `rank or 0` — khi đó chunk KHÔNG được nhánh
    # nào tiến cử lại nhận 1/60, tức ĐIỂM CAO NHẤT có thể. Chặn ồn ào còn hơn sai im lặng.
    if rank < 1:
        raise ValueError(f"rank phải >= 1 (1-based), nhận được {rank}")

    # 1.0 chứ không 1: viết rõ ý định là chia thực. Gõ nhầm `//` thì mọi điểm thành 0, mọi
    # chunk hoà nhau, thứ tự cuối cùng biến thành thứ tự dict — chạy được, ra kết quả, sai
    # hoàn toàn, không một dòng lỗi.
    return 1.0 / (k_constant + rank)


def fuse_rankings(
    ranked_ids_by_branch: dict[str, list[int]],
    k_constant: int = RRF_K_CONSTANT,
    top_k: int = DEFAULT_TOP_K,
) -> list[FusionResult]:
    """Nhiều danh sách id đã xếp hạng -> một danh sách FusionResult, sắp theo rrf_score giảm.

    Ví dụ: fuse_rankings({'vector': [17, 42, 8], 'keyword': [42, 5, 17]}, top_k=3)
           -> [FusionResult(42, ranks={'vector': 2, 'keyword': 1}),
               FusionResult(17, ranks={'vector': 1, 'keyword': 3}),
               FusionResult(8,  ranks={'vector': 3, 'keyword': None})]

    Hàm nhận list[int] chứ không nhận list[KeywordHit]/list[RetrievedChunk] để nó THUẦN và
    test được bằng 6 con số trong 1 giây, không cần DB không cần model. Đây là hàm chứa toàn
    bộ phần đáng sai của tầng fusion nên nó phải là hàm dễ chạy nhất; việc ghép ngược
    id -> nội dung chunk là chuyện của hybrid_search.py.
    """
    # Ghi lại hạng TỐT NHẤT của từng id ở từng nhánh. `if branch not in row` chống id TRÙNG
    # trong cùng một nhánh (chunk bị ingest 2 lần, hoặc SQL join nhân đôi dòng): không có nó
    # thì chunk đó được cộng điểm HAI LẦN và leo lên top một cách bí ẩn.
    rank_table: dict[int, dict[str, int | None]] = {}
    for branch, ids in ranked_ids_by_branch.items():
        for rank, chunk_id in enumerate(ids, 1):
            row = rank_table.setdefault(chunk_id, {})
            if branch not in row:
                row[branch] = rank

    # Điền None cho các nhánh không tiến cử, để MỌI hàng có cùng bộ khoá. Thiếu khoá thì chỗ
    # in bảng phải `.get()` rải rác, và một ngày nào đó ai đó dùng `ranks[branch]` -> KeyError
    # giữa lúc đang chạy cả bộ câu hỏi.
    all_branches = list(ranked_ids_by_branch.keys())
    for row in rank_table.values():
        for branch in all_branches:
            row.setdefault(branch, None)

    results = []
    for chunk_id, row in rank_table.items():
        # `if rank is not None` là BỎ QUA nhánh vắng mặt, không thay bằng 0. Gán 0 là khẳng
        # định "nhánh kia đã xét và cho 0 điểm" — sai: nhánh kia CHƯA XÉT, chunk nằm ngoài
        # pool của nó. "Vắng mặt" khác bản chất với "điểm thấp".
        score = sum(
            reciprocal_rank_score(rank, k_constant)
            for rank in row.values()
            if rank is not None
        )
        results.append(FusionResult(chunk_id=chunk_id, rrf_score=score, ranks=row))

    # Tiêu chí phụ `chunk_id` là tie-breaker TẤT ĐỊNH: hai chunk cùng rrf_score (rất dễ xảy
    # ra) mà không có nó thì thứ tự phụ thuộc thứ tự dict, tức phụ thuộc thứ tự Postgres trả
    # dòng — chạy lại có thể ra khác. Đo mà không lặp lại được thì không phải đo.
    # Và SẮP XONG MỚI CẮT: cắt trước là trả về top_k chunk ngẫu nhiên trong pool.
    results.sort(key=lambda r: (-r.rrf_score, r.chunk_id))
    return results[:top_k]


def min_max_normalize(scores: list[float]) -> list[float]:
    """Kéo một list điểm bất kỳ về [0, 1]. Mảnh của CÁCH LÀM THAY THẾ (weighted_fusion).

    Ví dụ: min_max_normalize([0.62, 0.41, 0.20]) -> [1.0, 0.5, 0.0]
           min_max_normalize([0.5, 0.5, 0.5])    -> [0.5, 0.5, 0.5]   (mọi điểm bằng nhau)
           min_max_normalize([])                 -> []

    Nhược điểm chí mạng, và là lý do RRF thắng ở đây: chuẩn hoá PHỤ THUỘC BATCH. Điểm cao
    nhất trong pool luôn thành 1.0 — kể cả khi nó là 0.03 và mọi chunk đều vô quan. Đổi câu
    hỏi, đổi pool -> cùng một chunk nhận điểm khác hẳn, không so được giữa các truy vấn, và
    mọi ngưỡng lọc mất ý nghĩa.
    """
    # Guard rỗng TRƯỚC vì max([]) ném ValueError. Nhánh keyword trả 0 kết quả là chuyện xảy
    # ra thật (T2: 4/8 câu keyword trượt sạch).
    if not scores:
        return []

    lowest, highest = min(scores), max(scores)
    # Ca suy biến max == min KHÔNG hiếm: pool 20 chunk mà ts_rank bằng nhau hết là chuyện
    # thường khi câu hỏi chỉ khớp đúng một token. Trả 0.5 (giữa) trung thực hơn 1.0 — không
    # có thông tin phân biệt thì đừng nói dối rằng chúng đều hoàn hảo. Và tránh chia cho 0.
    if highest == lowest:
        return [0.5] * len(scores)

    span = highest - lowest
    return [(s - lowest) / span for s in scores]


def weighted_fusion(
    scored_by_branch: dict[str, list[tuple[int, float]]],
    weights: dict[str, float],
    top_k: int = DEFAULT_TOP_K,
) -> list[FusionResult]:
    """CÁCH THAY THẾ RRF: min-max normalize từng nhánh rồi cộng có trọng số.

    Ví dụ: weighted_fusion({'vector': [(17, 0.62), (42, 0.41)],
                            'keyword': [(42, 0.08), (5, 0.02)]},
                           weights={'vector': 0.7, 'keyword': 0.3})
           -> list FusionResult sắp giảm dần theo điểm đã trộn

    Hàm này KHÔNG dùng trong app — nó tồn tại để chạy thử cạnh RRF trên cùng dữ liệu và tự
    thấy nó lệ thuộc batch thế nào (`python rrf.py` in cả hai bảng). Biết cả hai cách mới trả
    lời được "vì sao anh chọn RRF" bằng so sánh, thay vì bằng "vì tutorial làm thế".

    Ưu điểm thật của weighted: chỉnh được trọng số theo LOẠI truy vấn (câu chứa mã lỗi thì
    tăng weight keyword) — RRF không làm được. Nhược: thêm 2 tham số phải tune, mà tune theo
    một bộ 8 câu là overfit vào đúng 8 câu đó.
    """
    # Normalize TỪNG NHÁNH RIÊNG, không normalize chung một rổ: trộn chung thì mọi điểm
    # ts_rank (0.0x) nằm bẹp ở đáy thang, sau chuẩn hoá vẫn ~0 -> nhánh keyword bị xoá sổ mà
    # bảng vẫn ra bình thường. Normalize riêng mới cho mỗi nhánh một thang [0,1] của chính nó.
    blended: dict[int, float] = {}
    rank_table: dict[int, dict[str, int | None]] = {}
    for branch, pairs in scored_by_branch.items():
        ids = [chunk_id for chunk_id, _ in pairs]
        raw_scores = [score for _, score in pairs]
        normalized = min_max_normalize(raw_scores)
        weight = weights.get(branch, 0.0)
        for rank, (chunk_id, norm) in enumerate(zip(ids, normalized), 1):
            blended[chunk_id] = blended.get(chunk_id, 0.0) + weight * norm
            rank_table.setdefault(chunk_id, {}).setdefault(branch, rank)

    for row in rank_table.values():
        for branch in scored_by_branch:
            row.setdefault(branch, None)

    results = [
        FusionResult(chunk_id=cid, rrf_score=score, ranks=rank_table[cid])
        for cid, score in blended.items()
    ]
    results.sort(key=lambda r: (-r.rrf_score, r.chunk_id))
    return results[:top_k]


def describe_k_effect(
    ranked_ids_by_branch: dict[str, list[int]],
    k_values: tuple[int, ...] = (1, 10, 60, 200),
    top_k: int = 5,
) -> str:
    """Bảng: cùng dữ liệu, đổi k_constant thì thứ tự cuối cùng đổi thế nào.

    Mong muốn:
        k_constant |  thứ tự top-5
        ---------- |  -------------
                 1 |  17, 42, 5, 8, 63
                60 |  42, 17, 5, 8, 63

    Biến câu chữ "k lớn thì ưu ái chunk được nhiều nhánh tiến cử" thành một bảng NHÌN THẤY
    thứ tự đảo.

    ⚠️ Nếu bảng KHÔNG đổi gì khi k chạy từ 1 tới 200 thì dữ liệu thử chưa có ca mâu thuẫn —
    sửa DỮ LIỆU THỬ, đừng sửa công thức. Muốn thấy đảo cần đúng một cấu hình: một chunk hạng
    1 ở MỘT nhánh đấu với một chunk hạng thấp ở CẢ HAI nhánh, ví dụ
        {'vector': [100, 11, 12, 13, 200], 'keyword': [21, 22, 23, 24, 200]}
    id 100 = 1/(k+1) · id 200 = 2/(k+5) -> giao điểm ở k = 3.
    """
    lines = ["  k_constant |  thứ tự top-%d" % top_k, "  ---------- |  " + "-" * 20]

    for k_constant in k_values:
        fused = fuse_rankings(ranked_ids_by_branch, k_constant=k_constant, top_k=top_k)
        order = ", ".join(str(r.chunk_id) for r in fused)
        # f"{x:>10}" căn phải trong 10 ký tự — cột thẳng hàng khi k có 1 hay 3 chữ số
        lines.append(f"  {k_constant:>10} |  {order}")

    return "\n".join(lines)


def self_check() -> None:
    """Assert bằng dữ liệu giả — không cần DB, không cần model."""

    # id 42: hạng 2 ở vector, hạng 1 ở keyword -> phải THẮNG id 17 (hạng 1 vector, hạng 3
    # keyword). Ca mâu thuẫn cố ý dựng ra để thấy RRF thật sự làm gì.
    ranked = {
        BRANCH_VECTOR: [17, 42, 8, 91, 63],
        BRANCH_KEYWORD: [42, 5, 17, 63, 88],
    }

    assert reciprocal_rank_score(1) > reciprocal_rank_score(2), "hạng nhỏ hơn phải điểm cao hơn"
    assert 0 < reciprocal_rank_score(1) < 1, "điểm phải nằm trong (0, 1) với k=60"
    gap = reciprocal_rank_score(1) - reciprocal_rank_score(2)
    assert gap < 0.001, f"k=60 phải làm hạng 1 và 2 gần nhau, chênh {gap} là chưa dùng k"
    try:
        reciprocal_rank_score(0)
    except ValueError:
        pass
    else:
        raise AssertionError("rank=0 phải ném ValueError, không được im lặng")

    fused = fuse_rankings(ranked, top_k=5)
    assert fused[0].chunk_id == 42, f"id 42 (có ở CẢ HAI nhánh) phải đứng đầu, đang là {fused[0].chunk_id}"
    assert fused[0].ranks[BRANCH_VECTOR] == 2 and fused[0].ranks[BRANCH_KEYWORD] == 1
    assert len(fused) == 5, "top_k phải cắt đúng"
    solo = [r for r in fused if r.chunk_id == 8]
    assert solo and solo[0].ranks[BRANCH_KEYWORD] is None, "nhánh vắng mặt phải là None, KHÔNG phải 0"
    assert all(fused[i].rrf_score >= fused[i + 1].rrf_score for i in range(len(fused) - 1)), \
        "phải sắp giảm dần theo rrf_score"

    # id trùng trong cùng một nhánh chỉ được tính MỘT lần, theo hạng tốt nhất
    duplicated = fuse_rankings({BRANCH_VECTOR: [7, 7, 9]}, top_k=3)
    seven = [r for r in duplicated if r.chunk_id == 7][0]
    assert seven.ranks[BRANCH_VECTOR] == 1, "id trùng phải giữ hạng TỐT NHẤT"
    assert abs(seven.rrf_score - reciprocal_rank_score(1)) < 1e-12, "id trùng không được cộng 2 lần"

    assert min_max_normalize([]) == []
    # So float bằng == là sai: (0.41-0.20)/(0.62-0.20) ra 0.49999999999999994, KHÔNG phải 0.5.
    # Luôn so bằng sai số cho phép, nếu không thì test báo đỏ trong khi code hoàn toàn đúng.
    assert all(
        abs(actual - expected) < 1e-9
        for actual, expected in zip(min_max_normalize([0.62, 0.41, 0.20]), [1.0, 0.5, 0.0])
    ), "chuẩn hoá phải ra [1.0, 0.5, 0.0] trong sai số float"
    assert min_max_normalize([0.5, 0.5, 0.5]) == [0.5, 0.5, 0.5], "max==min phải trả 0.5, không chia 0"

    weighted = weighted_fusion(
        {BRANCH_VECTOR: [(17, 0.62), (42, 0.41), (8, 0.20)],
         BRANCH_KEYWORD: [(42, 0.08), (5, 0.02), (17, 0.01)]},
        weights={BRANCH_VECTOR: 0.7, BRANCH_KEYWORD: 0.3},
        top_k=5,
    )
    assert weighted, "weighted_fusion phải trả kết quả"
    assert all(r.rrf_score >= 0 for r in weighted)

    print("✅ self_check: reciprocal_rank_score + fuse_rankings + min_max_normalize + weighted_fusion pass")
    print("\n--- RRF trên dữ liệu mâu thuẫn cố ý ---")
    for rank, result in enumerate(fused, 1):
        print(f"  {rank}. id={result.chunk_id:<4} rrf={result.rrf_score:.6f}  {result.explain()}")

    print("\n--- Cùng dữ liệu, trộn kiểu weighted (0.7 vector / 0.3 keyword) ---")
    for rank, result in enumerate(weighted, 1):
        print(f"  {rank}. id={result.chunk_id:<4} blended={result.rrf_score:.6f}  {result.explain()}")
    print("  ^ RRF cho id 42 hạng 1, weighted cho id 17 hạng 1 — CÙNG dữ liệu, KHÁC kết luận.")
    print("    Và id 8 (thấp nhất nhánh vector) bị min-max ép về đúng 0.000000: weighted xoá")
    print("    sạch thông tin ở đáy batch, RRF thì vẫn cho nó 1/(60+3).")


def main() -> None:
    ranked = {
        BRANCH_VECTOR: [17, 42, 8, 91, 63],
        BRANCH_KEYWORD: [42, 5, 17, 63, 88],
    }

    if "--k-effect" in sys.argv:
        print("Cùng một dữ liệu, đổi k_constant:\n")
        print(describe_k_effect(ranked))
        print("\n  Đọc bảng: k nhỏ ưu ái HẠNG 1 của một nhánh. k lớn ưu ái chunk được")
        print("  NHIỀU nhánh cùng tiến cử. Bảng không đổi = dữ liệu thử thiếu ca mâu thuẫn,")
        print("  xem docstring describe_k_effect để lấy bộ dữ liệu lật được thứ tự.")
        return

    self_check()


if __name__ == "__main__":
    main()
