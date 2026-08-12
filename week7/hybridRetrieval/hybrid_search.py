"""
hybrid_search.py — Nối RRF vào hai nhánh thật: vector (tuần 5) + keyword (tuần 7 T2)

    INPUT :  câu hỏi + top_k
    OUTPUT:  top_k chunk sau fusion, kèm ĐỦ BA CỘT: rank_vector · rank_keyword · rrf_score

                                  câu hỏi
                                     |
              +----------------------+----------------------+
              |                                             |
     embed_query -> search_chunks                  search_by_keyword
     (week5/askCli/retriever.py)              (week7/keywordSearch/keyword_search.py)
              |                                             |
        pool 20 ứng viên                             pool 20 ứng viên
              |                                             |
              +--------------> fuse_rankings <--------------+
                              (rrf.py, THUẦN)
                                     |
                          id + rank_v + rank_k + score
                                     |
                        ghép ngược id -> nội dung chunk
                                     |
                                 top_k HybridHit

So với tuần 7 T2:
    T2:      hai nhánh chạy song song, IN RA HAI BẢNG, người đọc tự so
    hôm nay: hai nhánh -> MỘT bảng xếp hạng dùng được
                          ^^^^ mảnh mới duy nhất là fuse_rankings; hai nhánh giữ NGUYÊN,
                               không sửa một dòng nào trong keyword_search.py hay retriever.py

⚠️ CHỖ DỄ SAI NHẤT CỦA FILE NÀY:
    week5/askCli/retriever.py có HAI hàm nhìn rất giống nhau:
        search_chunks(conn, query_vector, k)  -> trả top-k THÔ, không lọc     ✅ dùng cái này
        retrieve(question, k)                 -> có gọi apply_similarity_threshold()
                                                 và CÓ THỂ TRẢ VỀ [] khi top-1 < 0.35
    Dùng `retrieve` ở đây thì nhánh vector im lặng biến mất khỏi fusion với đúng những câu
    hỏi khó — tức là những câu mà hybrid sinh ra để cứu. Bảng vẫn ra, không lỗi, và kết luận
    sẽ là "hybrid không giúp gì" trong khi thật ra chưa bao giờ chạy hybrid.
    Ngưỡng similarity là quyết định của TẦNG APP (trả lời hay từ chối), không phải của tầng
    retrieval. Nó thuộc về SAU fusion, không phải trước.

Self-test bằng dữ liệu giả — không cần DB, không cần model:
    python hybrid_search.py --dry
Chạy thật (cần Postgres Up + đã ingest + đã chạy fulltext_schema.py):
    python hybrid_search.py "Hermes engine trong React Native là gì?"
    python hybrid_search.py "..." --pool 40 --k 5 --rrf-k 10
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "keywordSearch"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))
sys.path.append(os.path.join(HERE, "..", "..", "week5", "askCli"))

from keyword_search import DEFAULT_MODE, search_by_keyword  # noqa: E402
from rrf import (  # noqa: E402
    BRANCH_KEYWORD,
    BRANCH_VECTOR,
    DEFAULT_CANDIDATE_POOL,
    DEFAULT_TOP_K,
    RRF_K_CONSTANT,
    FusionResult,
    fuse_rankings,
)
from vector_ops import get_conn  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class HybridHit:
    """1 chunk sau fusion, đã ghép lại nội dung.

    Cố ý KHÔNG kế thừa RetrievedChunk hay KeywordHit: chunk này không còn `similarity` cũng
    không còn `score` — hai con số đó đã bị RRF vứt đi có chủ đích. Giữ lại một trường tên
    `similarity` ở đây là mời gọi chính mình đi so nó với thứ khác thang đo.
    """

    id: int
    source: str
    title: str | None
    page: int | None
    heading: str | None
    content: str
    rrf_score: float
    rank_vector: int | None
    rank_keyword: int | None

    def location(self) -> str:
        """Cùng luật với RetrievedChunk.location() tuần 5 và KeywordHit.location() T2."""
        if self.page is not None:
            return f"trang {self.page}"
        return self.heading or "—"

    def found_in_both(self) -> bool:
        """Cả hai nhánh cùng tiến cử — tín hiệu tin cậy mạnh nhất mà hybrid tạo ra."""
        return self.rank_vector is not None and self.rank_keyword is not None


def collect_candidates(
    conn,
    question: str,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    mode: str = DEFAULT_MODE,
) -> tuple[dict[str, list[int]], dict[int, object]]:
    """Chạy CẢ HAI nhánh, trả (danh sách id theo hạng của từng nhánh, bảng tra id -> chunk).

    Ví dụ: collect_candidates(conn, "Hermes engine là gì?", candidate_pool=20)
           -> ({'vector': [17, 42, ...], 'keyword': [42, 5, ...]}, {17: <chunk>, 42: <chunk>, ...})

    Trả về HAI thứ vì fuse_rankings cố tình chỉ ăn id (để thuần, test không cần DB), nhưng
    sau fusion vẫn cần nội dung để in ra. Bảng tra `dict[id -> chunk]` là cầu nối, và nó gộp
    chunk từ CẢ HAI nhánh nên id nào cũng tra được, kể cả id chỉ có ở một bên.

    Nhánh keyword trả 0 kết quả là CHUYỆN BÌNH THƯỜNG (đo ở T2: 4/8 câu keyword trượt sạch),
    không được để nó ném lỗi hay chặn nhánh còn lại — danh sách rỗng đi vào fuse_rankings là
    hợp lệ, kết quả khi đó đúng bằng vector-only.
    """
    # Import LƯỜI: hai module này kéo theo sentence_transformers (~5 giây khởi động, ~470MB
    # RAM). Để ở đầu file thì `--dry` (vốn không cần model) cũng phải chờ 5 giây mỗi lần sửa.
    from retriever import search_chunks  # search_chunks, KHÔNG phải retrieve — xem docstring module
    from search_pg import embed_query

    query_vector = embed_query(question)
    vector_hits = search_chunks(conn, query_vector, k=candidate_pool)

    keyword_hits = search_by_keyword(conn, question, k=candidate_pool, mode=mode)

    # KHÔNG sort lại — SQL đã sắp rồi. Sort lại theo điểm là quay về đúng cái bẫy "so hai
    # thang đo" mà cả file này sinh ra để tránh.
    ranked_ids_by_branch = {
        BRANCH_VECTOR: [h.id for h in vector_hits],
        BRANCH_KEYWORD: [h.id for h in keyword_hits],
    }

    # Đặt keyword trước rồi vector đè lên: hai bên trả cùng bộ metadata nên đè cái nào cũng
    # được, nhưng cố định thứ tự để kết quả tất định.
    chunk_by_id = {h.id: h for h in keyword_hits}
    chunk_by_id.update({h.id: h for h in vector_hits})

    return ranked_ids_by_branch, chunk_by_id


def hybrid_search(
    conn,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    k_constant: int = RRF_K_CONSTANT,
    mode: str = DEFAULT_MODE,
) -> list[HybridHit]:
    """Câu hỏi -> top_k HybridHit đã hợp nhất bằng RRF.

    Ví dụ: hybrid_search(conn, "Hermes engine là gì?", top_k=3) -> [HybridHit, HybridHit, HybridHit]

    Hàm này chỉ LẮP các mảnh: collect_candidates -> fuse_rankings -> ghép nội dung. Không có
    logic nào đáng sai ở đây, và đó là chủ ý: mọi thứ đáng sai đã nằm trong rrf.py (test được
    không cần DB). Hàm chạm DB thì càng đơn giản càng tốt.
    """
    ranked_ids_by_branch, chunk_by_id = collect_candidates(
        conn, question, candidate_pool=candidate_pool, mode=mode
    )

    fused: list[FusionResult] = fuse_rankings(
        ranked_ids_by_branch, k_constant=k_constant, top_k=top_k
    )

    hits: list[HybridHit] = []
    for result in fused:
        # `.get()` chứ không `[...]`: về lý thuyết id chỉ sinh ra từ hai nhánh nên luôn tra
        # được, nhưng nếu sau này thêm nhánh thứ ba mà quên gộp vào bảng tra thì KeyError sẽ
        # nổ giữa lúc đang chạy cả bộ câu hỏi. Bỏ qua thì phải THẤY được, không im lặng.
        chunk = chunk_by_id.get(result.chunk_id)
        if chunk is None:
            print(f"  ⚠️ id {result.chunk_id} có trong fusion nhưng không có trong "
                  f"bảng tra — bỏ qua. Kiểm lại collect_candidates.")
            continue
        # Gán theo TÊN từng trường, đừng bung `HybridHit(*chunk)`: RetrievedChunk và
        # KeywordHit đều có `doc_type` ở vị trí 3 mà HybridHit thì không -> bung theo thứ tự
        # là `title` và `page` hoán chỗ, im lặng.
        hits.append(HybridHit(
            id=chunk.id,
            source=chunk.source,
            title=getattr(chunk, "title", None),
            page=getattr(chunk, "page", None),
            heading=getattr(chunk, "heading", None),
            content=chunk.content,
            rrf_score=result.rrf_score,
            rank_vector=result.rank_in(BRANCH_VECTOR),
            rank_keyword=result.rank_in(BRANCH_KEYWORD),
        ))

    return hits


def format_hybrid_hits(hits: list[HybridHit], top_k: int) -> str:
    """Bảng kết quả có ĐỦ BA CỘT để debug được. Hàm thuần — không chạm DB.

    Mong muốn:
        1. id=142   rrf=0.032266  v=3  k=1   ✔cả hai   Frontend Interview Prep · Hermes
             Hermes là JS engine do Meta viết riêng cho React Native, tối ưu thời gian...
        2. id=17    rrf=0.016393  v=1  k=—             Top 50 Full Stack · trang 12

    BẮT BUỘC in cả rank_vector lẫn rank_keyword chứ không chỉ điểm cuối: đó là khác biệt giữa
    "hệ thống trả về chunk này" và "chunk này lên top vì keyword bắt được mã lỗi trong khi
    vector không thấy". Cột thứ hai là thứ trả lời được câu hỏi phỏng vấn, và là thứ cho phép
    sửa đúng chỗ khi kết quả tệ. Bảng chỉ có điểm cuối = hộp đen tự tay dựng.
    """
    if not hits:
        return "  (0 kết quả — CẢ HAI nhánh đều rỗng. Kiểm: DB đã ingest chưa, fulltext_schema.py đã chạy chưa)"

    # rank_keyword = None là chuyện thường; hiện chữ "None" trong bảng làm người đọc tưởng lỗi.
    def cell(rank: int | None) -> str:
        return "—" if rank is None else str(rank)

    lines = []
    for rank, hit in enumerate(hits[:top_k], 1):
        label = hit.title or hit.source
        # " ".join(text.split()) gộp mọi khoảng trắng liên tiếp thành một dấu cách. Chunk lấy
        # từ PDF thường đầy \n, in thẳng là bảng vỡ thành chục dòng.
        snippet = " ".join(hit.content.split())[:70]
        both = "✔cả hai" if hit.found_in_both() else "       "
        lines.append(
            f"  {rank}. id={hit.id:<6} rrf={hit.rrf_score:.6f}  "
            f"v={cell(hit.rank_vector):<3} k={cell(hit.rank_keyword):<3} {both}  "
            f"{label} · {hit.location()}"
        )
        lines.append(f"       {snippet}...")
    return "\n".join(lines)


def self_check() -> None:
    """Assert bằng chunk giả — không cần DB, không cần model, chạy tức thì."""

    fake_hits = [
        HybridHit(142, "docs/frontend.docx", "Frontend Interview Prep", None, "Hermes",
                  "Hermes là JS engine\ndo Meta viết riêng   cho React Native.",
                  0.032266, 3, 1),
        HybridHit(17, "docs/fullstack.pdf", "Top 50 Full Stack", 12, None,
                  "Chuỗi kết nối MongoDB mặc định dùng cổng 27017.", 0.016393, 1, None),
        HybridHit(88, "notes/rag.md", None, None, None,
                  "Một đoạn chỉ keyword tìm ra.", 0.016129, None, 2),
    ]

    table = format_hybrid_hits(fake_hits, top_k=3)
    assert "None" not in table, "không được để chữ 'None' lọt vào bảng cho người đọc"
    assert "—" in table, "rank vắng mặt phải hiện dấu gạch"
    assert "\n" in table and table.count("\n") == 5, "3 hit -> 6 dòng (mỗi hit 2 dòng)"
    assert "trang 12" in table, "chunk pdf phải hiện số trang"
    assert "Hermes" in table, "chunk md/docx phải hiện heading"
    assert "notes/rag.md" in table, "chunk không có title phải rơi về source"

    assert fake_hits[0].found_in_both() is True
    assert fake_hits[1].found_in_both() is False
    assert fake_hits[2].found_in_both() is False

    assert format_hybrid_hits([], top_k=3).strip().startswith("(0 kết quả")

    print("✅ self_check: format_hybrid_hits + HybridHit pass (không cần DB, không cần model)")
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
    candidate_pool = parse_int_flag("--pool", DEFAULT_CANDIDATE_POOL)
    k_constant = parse_int_flag("--rrf-k", RRF_K_CONSTANT)

    # Bỏ qua cả cờ lẫn GIÁ TRỊ của nó khi gom câu hỏi, nếu không thì "20" của `--pool 20` sẽ
    # bị nối vào câu hỏi.
    skip_next = False
    positional = []
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--"):
            skip_next = arg in ("--k", "--pool", "--rrf-k")
            continue
        positional.append(arg)

    question = " ".join(positional)
    if not question:
        raise SystemExit(
            'Dùng: python hybrid_search.py "câu hỏi" [--k 5] [--pool 20] [--rrf-k 60] | --dry'
        )

    print(f"❓ {question}")
    print(f"   pool={candidate_pool}/nhánh · rrf_k={k_constant} · top_k={top_k}\n")

    with get_conn() as conn:
        hits = hybrid_search(
            conn, question, top_k=top_k, candidate_pool=candidate_pool, k_constant=k_constant
        )
        print(format_hybrid_hits(hits, top_k))

        both = sum(1 for h in hits if h.found_in_both())
        print(f"\n📊 {both}/{len(hits)} chunk trong top-{top_k} được CẢ HAI nhánh tiến cử")
        if both == 0:
            print("   ⚠️ Không chunk nào được cả hai nhánh đồng ý. Hai nhánh đang nhìn hai")
            print("      hướng hoàn toàn khác nhau — hoặc nhánh keyword trả rỗng. Kiểm bằng:")
            print('      python ../keywordSearch/keyword_search.py "<câu hỏi>"')


if __name__ == "__main__":
    main()
