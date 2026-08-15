"""
rerank.py — Tầng chấm lại cuối cùng bằng CROSS-ENCODER

    INPUT :  câu hỏi + list N chunk ứng viên (thường là output của hybrid_search)
    OUTPUT:  CÙNG list đó, đã xếp lại, mỗi phần tử kèm rerank_score + hạng TRƯỚC và SAU

           câu hỏi                    N chunk ứng viên (N=20)
              |                                |
              +--------> build_pairs <---------+
                              |
                   [(q, doc1), (q, doc2), ... (q, doc20)]
                              |
                        score_pairs          <-- MỘT lời gọi model, 20 cặp
                              |                  (không phải 20 lời gọi)
                     [1.2, -8.4, 5.9, ...]   <-- logit thô, KHÔNG phải xác suất
                              |
                        rerank_hits
                              |
                  sắp giảm dần theo score -> cắt top_k
                              |
                  RerankedHit(rank_before=11, rank_after=1)

BI-ENCODER vs CROSS-ENCODER — cả file này sinh ra từ đúng một khác biệt:

    bi-encoder (tuần 4-5, paraphrase-multilingual-MiniLM):
        encode(câu hỏi) -> vector          }  hai lần chạy model ĐỘC LẬP,
        encode(tài liệu) -> vector         }  rồi so hai vector bằng cosine
        => vector tài liệu TÍNH TRƯỚC ĐƯỢC, nhét vào index, truy vấn chỉ tốn 1 lần encode.
           Đó là lý do search 50k chunk vẫn nhanh.

    cross-encoder (ms-marco-MiniLM-L-6-v2):
        model đọc CẢ CẶP cùng lúc: [CLS] câu hỏi [SEP] tài liệu [SEP] -> 1 điểm liên quan
        => attention chạy CHÉO giữa từ của câu hỏi và từ của tài liệu, nên chính xác hơn
           hẳn. Cái giá: KHÔNG tính trước được gì cả. Điểm chỉ tồn tại khi đã biết câu hỏi.
           N tài liệu = N lần chạy model MỖI câu hỏi.

    => Vì sao không dùng cross-encoder cho cả kho: 50k chunk = 50k lần chạy model cho MỘT
       câu hỏi. Kiến trúc 2 tầng không phải tối ưu hoá cho vui — nó là điều kiện để
       cross-encoder dùng được.

⚠️ ĐIỂM CROSS-ENCODER KHÔNG PHẢI SIMILARITY:
    ms-marco-MiniLM-L-6-v2 xuất ra MỘT logit thô, khoảng chừng -11..+11, âm là bình thường.
    Nó KHÔNG chặn trên, KHÔNG phải xác suất, KHÔNG so được với cosine (0..1) hay ts_rank.
    Đây là loại điểm THỨ BA không cùng thang đo trong repo này (sau cosine và ts_rank).
    Cách dùng duy nhất an toàn: SẮP XẾP. Đừng cộng, đừng lấy trung bình, đừng đặt ngưỡng
    tuyệt đối kiểu "score > 0 là liên quan".

⚠️ GIỚI HẠN ĐÃ ĐO ĐƯỢC (12/08, xem python-knowledge/Tuan-07/T4):
    ms-marco-* huấn luyện trên MS MARCO — tiếng ANH. Bộ 8 câu hỏi của repo này TOÀN tiếng
    Việt. Kết quả đo: 0/6 ca cải thiện, 2 ca tụt hạng, 1 ca đoạn đúng bị đá khỏi top-3.
    Reranker ở đây đang làm hại chứ không giúp. Đây là giới hạn của MODEL với ngôn ngữ đầu
    vào, không phải lỗi của kiến trúc 2 tầng — muốn kiểm chứng thì đổi sang model đa ngữ
    (bge-reranker-v2-m3 / jina-reranker-v2-base-multilingual) rồi đo lại trên CÙNG bộ câu.

Self-test bằng dữ liệu giả + scorer giả — KHÔNG cần DB, KHÔNG cần model, KHÔNG tải 80MB:
    python rerank.py --dry
Chạy thật (tải model lần đầu ~80MB, cần mạng):
    python rerank.py "Hermes engine trong React Native là gì?"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "hybridRetrieval"))
sys.path.append(os.path.join(HERE, "..", "keywordSearch"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))
sys.path.append(os.path.join(HERE, "..", "..", "week5", "askCli"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# L-6 = 6 lớp transformer, ~80MB, chạy CPU được. Bản L-12 chính xác hơn và chậm gấp đôi.
# Đổi model là đổi điều kiện đo — muốn so hai model thì đo cả hai trong CÙNG một lần chạy.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Số ứng viên đưa vào rerank. Phải KHỚP candidate_pool của hybrid_search: rerank chỉ xếp lại
# thứ mình đưa cho nó, không đi tìm thêm. N=3 thì reranker vô dụng, N=200 thì latency ×10.
DEFAULT_RERANK_POOL = 20

DEFAULT_TOP_K = 3

# Cắt chunk trước khi đưa vào model. Cross-encoder có max_length 512 token cho CẢ CẶP
# (câu hỏi + tài liệu + token đặc biệt); quá thì model TỰ CẮT ÂM THẦM phần đuôi. Cắt tường
# minh ở đây để biết mình đang chấm cái gì.
# ⚠️ 900 ký tự tiếng Việt sinh ra nhiều token hơn 900 ký tự tiếng Anh (tokenizer tiếng Anh
#    tách dấu thanh thành nhiều subword) -> phần lớn cặp có thể đang chạy ở đúng trần 512
#    token, và đó là một phần lý do latency đo được cao. Giảm xuống 400 rồi đo lại là phép
#    thử rẻ nhất.
MAX_DOC_CHARS = 900

# Chấm bao nhiêu cặp một lượt. Lớn hơn = nhanh hơn nhưng tốn RAM. 32 an toàn trên máy 8GB.
BATCH_SIZE = 32


_cross_encoder = None


def get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL):
    """Load cross-encoder MỘT lần rồi cache.

    LAZY LOADING (bài học tuần 5, lý do ở đây còn mạnh hơn):
      · import `sentence_transformers` nằm TRONG hàm -> `--dry` và mọi self-test chạy tức
        thì, không phải chờ ~5 giây khởi động + ~400MB RAM.
      · cache module-level -> chạy 8 câu hỏi chỉ load model 1 lần. Thiếu cache thì mỗi câu
        tốn thêm 5 giây và bảng latency thành vô nghĩa.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # import lười, cố ý
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


@dataclass
class RerankedHit:
    """1 chunk sau khi rerank.

    hit         : object gốc (HybridHit / RetrievedChunk / KeywordHit) — GIỮ NGUYÊN, không
                  copy từng trường sang. Copy là tự tạo thêm một chỗ để lệch dữ liệu.
    rerank_score: logit thô của cross-encoder. Không phải xác suất, có thể âm.
    rank_before : hạng trong danh sách ĐẦU VÀO (1-based) — hạng do hybrid/RRF quyết định.
    rank_after  : hạng sau khi rerank (1-based).

    Giữ `rank_before` là bắt buộc: không có nó thì không trả lời được "reranker thật sự làm
    gì". Một chunk nhảy 11 -> 1 là bằng chứng; một danh sách đã sắp sẵn chỉ là một danh sách.
    """

    hit: object
    rerank_score: float
    rank_before: int
    rank_after: int

    @property
    def id(self) -> int:
        return getattr(self.hit, "id", -1)

    @property
    def label(self) -> str:
        """Nhãn người-đọc-được: ưu tiên title, không có thì rơi về source."""
        return getattr(self.hit, "title", None) or getattr(self.hit, "source", "?")

    def location(self) -> str:
        """Uỷ quyền cho hit gốc — HybridHit/KeywordHit/RetrievedChunk đều có location()."""
        locate = getattr(self.hit, "location", None)
        return locate() if callable(locate) else "—"

    def movement(self) -> int:
        """Số hạng đã DI CHUYỂN. Dương = leo lên, âm = tụt xuống, 0 = đứng yên."""
        return self.rank_before - self.rank_after

    def arrow(self) -> str:
        """Ký hiệu hướng di chuyển để nhìn bảng trong một nhịp mắt."""
        move = self.movement()
        if move > 0:
            return f"▲{move}"
        if move < 0:
            return f"▼{-move}"
        return "  ="


# Kiểu của hàm chấm điểm. Tách ra thành type để `score_pairs` nhận được scorer GIẢ trong
# self-test — nhờ vậy logic sắp xếp (phần dễ sai nhất) test được mà không cần tải 80MB model.
ScorerFn = Callable[[list[tuple[str, str]]], list[float]]


def build_pairs(question: str, hits: Sequence[object]) -> list[tuple[str, str]]:
    """Câu hỏi + N chunk -> N cặp (câu hỏi, văn bản tài liệu) đúng thứ tự đầu vào.

    Ví dụ: build_pairs("Hermes là gì?", [hit_a, hit_b])
           -> [("Hermes là gì?", "Hermes là JS engine do Meta..."), ...]

    THỨ TỰ phải giữ nguyên tuyệt đối: `score_pairs` trả về một list SỐ không kèm id, việc
    ghép điểm thứ i với chunk thứ i dựa hoàn toàn vào giả định hai list cùng thứ tự. Lỡ
    sort/lọc ở đây thì mọi chunk nhận điểm CỦA CHUNK KHÁC — bảng vẫn ra, hoàn toàn sai.
    Đó là lý do hàm này không có một câu `if` lọc nào.

    Quyết định thiết kế: nối `title` vào trước content. title chứa tên tài liệu, là tín hiệu
    thật cho câu hỏi có tên riêng — nhưng khi đọc kết quả phải nhớ là file có title trùng từ
    khoá sẽ được ưu ái toàn bộ.
    """
    clean_question = " ".join(question.split())

    def document_text(hit) -> str:
        title = getattr(hit, "title", None) or ""
        # " ".join(x.split()) bóp mọi khoảng trắng liên tiếp về 1 dấu cách: chunk từ PDF đầy
        # \n, để nguyên thì ngân sách 512 token bị đốt cho khoảng trắng.
        body = " ".join(hit.content.split())
        merged = f"{title}. {body}" if title else body
        return merged[:MAX_DOC_CHARS]

    # `hit.content` truy cập THẲNG (không getattr có default): truyền nhầm kiểu thì phải nổ
    # AttributeError ngay, chứ không lặng lẽ tạo ra 20 cặp rỗng cho model chấm rác.
    return [(clean_question, document_text(hit)) for hit in hits]


def score_pairs(pairs: list[tuple[str, str]], scorer: ScorerFn | None = None) -> list[float]:
    """N cặp -> N điểm float thuần Python, đúng thứ tự đầu vào.

    Ví dụ: score_pairs([("q", "doc liên quan"), ("q", "doc lạc đề")]) -> [5.91, -8.42]
           score_pairs(pairs, scorer=fake_scorer) -> không đụng tới model

    `scorer` là dependency injection: self_check chấm bằng hàm giả đếm từ khoá trùng và chạy
    trong 0.01 giây, không mạng, không 80MB. Pattern này sẽ dùng lại ở tuần 8 để mock
    LLM-as-judge.
    """
    if not pairs:
        return []

    if scorer is not None:
        return scorer(pairs)

    model = get_cross_encoder()
    # MỘT lời gọi cho CẢ list. Gọi predict trong vòng lặp (mỗi cặp một lời gọi) vẫn ra kết
    # quả đúng nhưng chậm gấp 5-10 lần vì mỗi lời gọi tự dựng batch + chuyển tensor + đồng bộ.
    raw_scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

    # predict trả np.ndarray của np.float32. Đổi sang float thuần NGAY tại biên giới model:
    # để np.float32 rò ra ngoài thì json.dump nổ "not JSON serializable" ở cuối buổi, sau khi
    # đã chạy xong hết.
    return [float(score) for score in raw_scores]


def rerank_hits(
    question: str,
    hits: Sequence[object],
    top_k: int = DEFAULT_TOP_K,
    scorer: ScorerFn | None = None,
) -> list[RerankedHit]:
    """Câu hỏi + N chunk -> top_k RerankedHit đã xếp lại, kèm hạng TRƯỚC và SAU.

    Ví dụ: rerank_hits("Hermes là gì?", hybrid_hits_20, top_k=3)
           -> [RerankedHit(rank_before=11, rank_after=1), ...]

    Ba thứ tự BẮT BUỘC trong hàm này, sai cái nào cũng là bug im lặng:
      1. chốt `rank_before` TRƯỚC khi sort — sau khi sort thì thứ tự cũ mất vĩnh viễn
      2. cắt `top_k` SAU khi sort — cắt trước là rerank chỉ được nhìn 3 ứng viên, chunk hạng
         11 (đúng thứ mà kiến trúc 2 tầng sinh ra để cứu) không bao giờ có cơ hội
      3. sắp GIẢM dần — điểm cross-encoder cao hơn là liên quan hơn, ngược với khoảng cách
         `<=>` của pgvector nơi NHỎ hơn là gần hơn. Repo này giờ có cả hai loại cạnh nhau.
    """
    if not hits:
        return []

    pairs = build_pairs(question, hits)
    scores = score_pairs(pairs, scorer=scorer)

    scored = [
        RerankedHit(hit=hit, rerank_score=score, rank_before=rank, rank_after=0)
        for rank, (hit, score) in enumerate(zip(hits, scores), 1)
    ]

    # zip dừng ở list NGẮN HƠN mà không báo gì — thiếu assert này thì vài chunk cuối biến mất
    # im lặng khi score_pairs trả thiếu phần tử.
    assert len(scores) == len(hits), f"lệch số lượng: {len(scores)} điểm / {len(hits)} chunk"

    # Khoá sắp xếp là TUPLE: giảm dần theo điểm (dấu trừ), tie-break tăng dần theo rank_before
    # -> hoà điểm thì giữ nguyên trật tự của hybrid, và chạy lại luôn ra kết quả y hệt.
    scored.sort(key=lambda r: (-r.rerank_score, r.rank_before))

    for new_rank, item in enumerate(scored, 1):
        item.rank_after = new_rank
    return scored[:top_k]


def format_rerank_table(reranked: list[RerankedHit], hits_before: Sequence[object]) -> str:
    """Bảng "trước -> sau" cho một câu hỏi. Hàm thuần — không chạm DB, không chạm model.

    Mong muốn:
        hạng cũ -> mới   điểm      chunk
          11    ->  1   ▲10   +5.912  id=142  Frontend Interview Prep · Hermes
        ⟶ leo cao nhất: id=142 (11 -> 1)
        ⟶ bị đá khỏi top-3: id=42 (hạng 2 cũ)

    Nhận thêm `hits_before` vì chunk BỊ ĐÁ RA khỏi top-k không còn nằm trong `reranked` —
    phải suy ra bằng cách so với danh sách gốc. Không có dòng đó thì không phân biệt được
    "rerank hiệu quả" với "rerank không làm gì cả".
    """
    if not reranked:
        return "  (0 kết quả — hybrid không trả về ứng viên nào, rerank không có gì để xếp)"

    lines = ["  cũ -> mới          điểm     chunk"]
    for item in reranked:
        snippet = " ".join(getattr(item.hit, "content", "").split())[:70]
        # `{:+.3f}` in kèm dấu: điểm âm là BÌNH THƯỜNG với ms-marco, thấy dấu thì khỏi hoảng
        # lên đi "sửa" cái không hỏng.
        lines.append(
            f"  {item.rank_before:>3} -> {item.rank_after:<3} {item.arrow():<5} "
            f"{item.rerank_score:+.3f}  id={item.id:<6} {item.label} · {item.location()}"
        )
        lines.append(f"       {snippet}...")

    climber = max(reranked, key=lambda r: r.movement())
    if climber.movement() > 0:
        lines.append(f"  ⟶ leo cao nhất: id={climber.id} "
                     f"({climber.rank_before} -> {climber.rank_after})")

    top_k = len(reranked)
    kept_ids = {r.id for r in reranked}     # set để tra O(1)
    dropped = [
        (rank, getattr(hit, "id", -1))
        for rank, hit in enumerate(hits_before[:top_k], 1)
        if getattr(hit, "id", -1) not in kept_ids
    ]

    if dropped:
        detail = ", ".join(f"id={cid} (hạng {rank} cũ)" for rank, cid in dropped)
        lines.append(f"  ⟶ bị đá khỏi top-{top_k}: {detail}")

    return "\n".join(lines)


@dataclass
class FakeHit:
    """Chunk giả cho self-test — đủ trường để rerank_hits và format_rerank_table dùng được."""

    id: int
    content: str
    title: str | None = None
    source: str = "fake.md"
    page: int | None = None

    def location(self) -> str:
        return f"trang {self.page}" if self.page is not None else "—"


def keyword_overlap_scorer(pairs: list[tuple[str, str]]) -> list[float]:
    """Scorer GIẢ: đếm số từ của câu hỏi xuất hiện trong tài liệu.

    Không mô phỏng cross-encoder — nó chỉ cần TẤT ĐỊNH và biết trước kết quả, để self_check
    khẳng định được "chunk nào phải lên hạng 1" mà không cần model.
    """
    scores = []
    for question, document in pairs:
        question_words = set(question.lower().split())
        document_words = set(document.lower().split())
        scores.append(float(len(question_words & document_words)))  # & = giao hai set
    return scores


def self_check() -> None:
    """Assert bằng chunk giả + scorer giả — không cần DB, không cần model, không cần mạng."""

    question = "Hermes engine trong React Native là gì"

    # Ca MÂU THUẪN cố ý: chunk đúng nhất nằm ở CUỐI danh sách đầu vào (hạng 4). Nếu rerank_hits
    # lỡ cắt top_k trước khi sắp xếp thì nó không bao giờ lên được hạng 1 -> test đỏ.
    hits = [
        FakeHit(1, "React Native dùng bridge để nói chuyện với native module.", "Frontend Interview Prep"),
        FakeHit(2, "Redux là thư viện quản lý state cho React.", "React Interview Questions"),
        FakeHit(3, "Node.js chạy trên V8 engine của Chrome.", "Backend-NodeJS", page=4),
        FakeHit(4, "Hermes engine trong React Native là gì: một JS engine do Meta viết.",
                "Frontend Interview Prep"),
    ]

    pairs = build_pairs(question, hits)
    assert len(pairs) == len(hits), "phải có đúng 1 cặp cho mỗi chunk"
    assert all(p[0] == " ".join(question.split()) for p in pairs), "câu hỏi phải giống nhau ở mọi cặp"
    assert "Frontend Interview Prep" in pairs[0][1], "title phải được nối vào văn bản tài liệu"
    assert "\n" not in pairs[0][1], "xuống dòng phải bị bóp về dấu cách"
    assert all(len(p[1]) <= MAX_DOC_CHARS for p in pairs), "tài liệu phải bị cắt ở MAX_DOC_CHARS"

    scores = score_pairs(pairs, scorer=keyword_overlap_scorer)
    assert len(scores) == len(pairs)
    assert type(scores[0]).__name__ == "float", f"phải là float thuần, đang là {type(scores[0]).__name__}"
    assert score_pairs([], scorer=keyword_overlap_scorer) == [], "pairs rỗng -> [] , không được nổ"

    reranked = rerank_hits(question, hits, top_k=3, scorer=keyword_overlap_scorer)
    assert len(reranked) == 3, "top_k phải cắt đúng"
    assert reranked[0].id == 4, (
        f"chunk id=4 (hạng 4 đầu vào, khớp nhiều từ nhất) PHẢI lên hạng 1, đang là id={reranked[0].id}. "
        "Sai ở đây gần như chắc chắn là: cắt top_k trước khi sort, hoặc sort nhầm chiều."
    )
    assert reranked[0].rank_before == 4 and reranked[0].rank_after == 1
    assert reranked[0].movement() == 3 and reranked[0].arrow() == "▲3"
    assert [r.rank_after for r in reranked] == [1, 2, 3], "rank_after phải liên tiếp từ 1"
    assert all(
        reranked[i].rerank_score >= reranked[i + 1].rerank_score for i in range(len(reranked) - 1)
    ), "phải sắp GIẢM dần theo điểm — điểm cross-encoder cao hơn là liên quan hơn"
    assert len({r.rank_before for r in reranked}) == 3, "rank_before không được trùng"
    assert rerank_hits(question, [], top_k=3, scorer=keyword_overlap_scorer) == []

    # Hoà điểm: hai chunk y hệt nhau -> phải giữ trật tự đầu vào, chạy 100 lần ra 100 kết quả giống nhau
    tied = [FakeHit(10, "một câu"), FakeHit(11, "một câu")]
    tied_result = rerank_hits("một câu", tied, top_k=2, scorer=keyword_overlap_scorer)
    assert [r.id for r in tied_result] == [10, 11], "hoà điểm phải giữ trật tự đầu vào (tie-break tất định)"

    table = format_rerank_table(reranked, hits)
    assert "None" not in table, "không được để chữ 'None' lọt vào bảng cho người đọc"
    assert "▲3" in table, "phải hiện mũi tên di chuyển"
    assert "id=2" in table or "hạng 2 cũ" in table, "phải chỉ ra chunk bị đá khỏi top-3"
    assert format_rerank_table([], hits).strip().startswith("(0 kết quả")

    print("✅ self_check: build_pairs + score_pairs + rerank_hits + format_rerank_table pass")
    print("   (chạy hoàn toàn bằng scorer giả — không tải model, không cần mạng)\n")
    print(table)


def main() -> None:
    if "--dry" in sys.argv or len(sys.argv) == 1:
        self_check()
        return

    from hybrid_search import hybrid_search  # import lười: kéo theo sentence_transformers
    from vector_ops import get_conn

    question = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
    print(f"❓ {question}")
    print(f"   pool={DEFAULT_RERANK_POOL} -> rerank -> top-{DEFAULT_TOP_K}\n")

    with get_conn() as conn:
        candidates = hybrid_search(conn, question, top_k=DEFAULT_RERANK_POOL,
                                   candidate_pool=DEFAULT_RERANK_POOL)
        print(f"  hybrid trả về {len(candidates)} ứng viên, đưa hết vào rerank\n")
        reranked = rerank_hits(question, candidates, top_k=DEFAULT_TOP_K)
        print(format_rerank_table(reranked, candidates))


if __name__ == "__main__":
    main()

# Kỳ vọng khi chạy:
#   `--dry`      : in "pass" + bảng có mũi tên ▲/▼, dưới 1 giây (lâu hơn = đang lỡ tải model)
#   chạy thật    : chunk chứa từ khoá của câu hỏi lên hạng 1, và CÓ chunk đổi hạng so với đầu vào
#   điểm quan sát: nằm khoảng -11..+11, số âm là bình thường
