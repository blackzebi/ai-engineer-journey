"""
retriever.py — Lấy chunk kèm ĐẦY ĐỦ metadata + chặn câu trả lời bịa bằng ngưỡng similarity

Tuần 4 trích nguồn được tới mức "file nào". Tuần 5 (T2) đã nhét page/heading/title vào DB
rồi mà chưa ai dùng — file này lôi ra dùng, và thêm một lớp chống bịa thứ hai:

    tuần 4:  question ─embed─► search_top_k ──► (id, source, content, sim) ──► LLM
                                                        │
                                                        └─ trích nguồn tối đa: "README.md"

    tuần 5:  question ─embed─► search_chunks ─► RetrievedChunk(+title,+page,+heading,+doc_type)
                                    │                  │
                        WHERE doc_type = ?             ├─► apply_similarity_threshold
                        (--doc-type pdf)               │      └─ top-1 < 0.35 -> TRẢ VỀ RỖNG,
                                                       │         KHÔNG gọi LLM (tiết kiệm tiền,
                                                       │         và không cho nó cơ hội bịa)
                                                       └─► trích nguồn: "Báo cáo Q3 · trang 12"

Hai lớp chống bịa bổ sung cho nhau, khác nhau ở THỜI ĐIỂM:
    lớp 1 (tuần 4, prompt): "chỉ trả lời dựa trên đoạn trích" — chữa lúc SINH câu trả lời
    lớp 2 (hôm nay, ngưỡng): chặn TRƯỚC khi sinh — vì prompt tốt tới mấy, đưa cho model
                             3 đoạn hoàn toàn vô quan thì nó vẫn có xu hướng cố ghép đại

Tái dùng nguyên si:
    - week4/pgvector/vector_ops.py    : get_conn, to_pgvector
    - week4/ragPipeline/search_pg.py  : embed_query  (import LƯỜI, xem retrieve)
    - week4/ragLite/prompt_rag.py     : build_rag_prompt, SYSTEM_PROMPT (dùng ở stream_answer.py)
    - ./errors.py                     : AskError, ErrorKind

Về week5/indexTuning/filter_metadata.py::search_filtered — cố ý KHÔNG import, vì nó trả 6 cột
(thiếu title và heading). Thiếu title thì trích nguồn chỉ còn đường dẫn file thô; thiếu heading
thì docx/md/txt (page luôn NULL) mất sạch địa chỉ để chỉ. `search_chunks` dưới đây là bản 8 cột,
siêu tập của nó. Hai câu SQL gần trùng nhau ở hai file sẽ phân kỳ — dedup bằng commit riêng,
xem mục "Dedup với T3" trong guide.md.

Self-test bằng chunk giả — không cần DB, không cần model:
    python retriever.py
Chạy thật (cần Postgres Up + đã ingest):
    python retriever.py "embedding là gì?"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))

from errors import AskError, ErrorKind  # noqa: E402
from vector_ops import get_conn, to_pgvector  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Ngưỡng similarity: dưới mức này coi như "không tìm thấy".
# 0.35 là điểm khởi đầu cho paraphrase-multilingual-MiniLM-L12-v2, KHÔNG phải hằng số vũ trụ —
# mỗi model một thang điểm khác nhau. Cách chỉnh đúng: chạy 5 câu CÓ trong tài liệu và 5 câu
# CHẮC CHẮN KHÔNG có, xem điểm top-1 của hai nhóm, đặt ngưỡng vào khoảng trống giữa chúng.
# Đặt quá cao -> app im lặng với câu hỏi hợp lệ, hỏng kiểu đó còn khó phát hiện hơn là bịa.
MIN_SIMILARITY = 0.35

DEFAULT_TOP_K = 3


@dataclass
class RetrievedChunk:
    """1 chunk lấy từ DB, kèm đủ thứ cần để trích nguồn cho người đọc kiểm chứng.

    Là dataclass chứ không tuple 8 phần tử như tuần 4: `row[4]` là page hay heading? Đọc lại
    query mới biết. Với 4 cột còn chịu được, 8 cột thì mọi chỗ dùng đều thành câu đố.
    """

    id: int
    source: str
    doc_type: str | None
    title: str | None
    page: int | None
    heading: str | None
    content: str
    similarity: float

    def location(self) -> str:
        """Địa chỉ người-đọc-được: 'trang 12' | heading | '—' (khi cả hai đều NULL).

        page NULL với docx/md/txt là CHUYỆN BÌNH THƯỜNG (xem db_docs.py) — nghĩa là
        "không áp dụng", khác hẳn "quên điền". Không xử lý ở đây thì màn hình hiện 'trang None'.
        """
        if self.page is not None:
            return f"trang {self.page}"
        return self.heading or "—"

    def as_prompt_chunk(self) -> tuple[int, str, str, float]:
        """Đổi về đúng hình dạng (id, source, content, similarity) mà prompt_rag.py tuần 4 ăn.

        Mẹo tái dùng: nhét title + vị trí vào chỗ `source`, thế là format_context() của
        tuần 4 hiện "Báo cáo Q3 · trang 12" trong context -> model trích dẫn được tới TRANG
        mà KHÔNG phải sửa một dòng nào trong file tuần 4.
        """
        label = self.title or self.source
        return (self.id, f"{label} · {self.location()}", self.content, self.similarity)


def search_chunks(
    conn,
    query_vector: list[float],
    k: int = DEFAULT_TOP_K,
    doc_type: str | None = None,
) -> list[RetrievedChunk]:
    """top-k chunk gần nghĩa nhất, kèm metadata. doc_type=None nghĩa là không lọc.

    Ví dụ: search_chunks(conn, vector, k=3, doc_type="pdf") -> 3 RetrievedChunk, doc_type='pdf'

    Vẫn `ORDER BY embedding <=> %s` chứ không `ORDER BY similarity DESC`: hai câu ra cùng kết
    quả, nhưng chỉ câu đầu dùng được index HNSW (tạo ở T3). Sắp theo biểu thức `1 - (...)` là
    Postgres bỏ index, quét tuần tự.

    Lưu ý về lọc doc_type: kết hợp với index vector cho ra post-filter — Postgres lấy top-k
    theo vector TRƯỚC rồi mới bỏ dòng không khớp, nên xin k=5 có thể nhận về 2. Không phải bug
    và không chữa được bằng SQL khéo hơn. Xem số đo thật:
        python ../indexTuning/filter_metadata.py --doc-type pdf
    """
    # Thứ tự params phải khớp thứ tự dấu %s XUẤT HIỆN trong câu SQL, không phải thứ tự mình
    # nghĩ: (1) vector ở SELECT · (2) doc_type ở WHERE — chèn Ở GIỮA · (3) vector lần 2 ở
    # ORDER BY · (4) k. Đảo (1) và (2) mà psycopg cast được thì câu lệnh CHẠY BÌNH THƯỜNG và
    # trả về rác. Vì vậy append params đúng lúc lắp mảnh SQL tương ứng.
    query_literal = to_pgvector(query_vector)
    sql = ("SELECT id, source, doc_type, title, page, heading, content, "
           "1 - (embedding <=> %s) AS similarity FROM chunks")
    params: list = [query_literal]

    # Chỉ nối phần CẤU TRÚC vào chuỗi SQL; GIÁ TRỊ luôn đi qua params (tránh SQL injection).
    if doc_type:
        sql += " WHERE doc_type = %s"
        params.append(doc_type)

    sql += " ORDER BY embedding <=> %s LIMIT %s;"
    params += [query_literal, k]

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    # RetrievedChunk(*row) bung 8 phần tử theo THỨ TỰ — chỉ đúng khi thứ tự cột SELECT khớp
    # thứ tự field trong dataclass. Đổi một bên mà quên bên kia thì title và content hoán chỗ,
    # không báo lỗi. Sửa thì sửa cả hai.
    return [RetrievedChunk(*row) for row in rows]


def apply_similarity_threshold(
    chunks: list[RetrievedChunk],
    min_similarity: float = MIN_SIMILARITY,
) -> list[RetrievedChunk]:
    """Top-1 quá thấp -> trả list RỖNG (không đủ căn cứ). Ngược lại -> bỏ các chunk yếu.

    Ví dụ: [0.62, 0.41, 0.20] với ngưỡng 0.35 -> giữ 2 chunk đầu
           [0.28, 0.22, 0.19] với ngưỡng 0.35 -> trả []  (top-1 đã dưới ngưỡng)

    Vì sao xét TOP-1 riêng chứ không chỉ lọc từng chunk — hai tình huống khác hẳn bản chất:
      (a) top-1 = 0.62, phần sau = 0.20 -> CÓ tài liệu liên quan, chỉ là ít. Giữ chunk mạnh,
          bỏ chunk yếu (chunk yếu chỉ làm loãng context và tốn token).
      (b) top-1 = 0.28 -> tài liệu KHÔNG chứa câu trả lời. Trả [] để tầng trên nói
          "không tìm thấy" mà không tốn một đồng token nào.
    Chỉ lọc từng chunk thì (b) rơi vào cùng nhánh với (a), mất khả năng phân biệt.

    Giả định NGẦM: list đã sắp giảm dần theo similarity (đúng, vì SQL ORDER BY distance tăng).
    Ai đó sort lại ở giữa là hỏng im lặng.
    """
    # Guard rỗng trước: retrieve rỗng là chuyện thường xuyên (DB chưa ingest, lọc doc_type
    # không khớp), không có dòng này thì chunks[0] nổ IndexError.
    if not chunks:
        return []

    if chunks[0].similarity < min_similarity:
        return []

    return [ c for c in chunks if c.similarity >= min_similarity]


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """Danh sách nguồn đánh số, khớp với [1][2][3] mà model trích trong câu trả lời.

    Mong muốn:
        Nguồn tham khảo:
          [1] Báo cáo kỹ thuật 2026 · trang 12   (0.62)  bao-cao.pdf
          [2] Ghi chú RAG · Chunking             (0.41)  notes/rag.md

    In CẢ title lẫn source: title để nhận ra tài liệu, source để MỞ được đúng file mà kiểm
    chứng. Trích nguồn mà người ta không lần ra được bản gốc thì chỉ là trang trí.

    In cả điểm similarity vì nó là tín hiệu độ tin cậy — 0.61 và 0.36 cho hai mức tin khác
    hẳn nhau. Đây cũng là thứ giúp chỉnh MIN_SIMILARITY bằng số thật thay vì ngồi đoán.
    """
    # Trả "" khi rỗng và để chỗ gọi tự xử — in "Nguồn tham khảo:" rồi bỏ trống bên dưới
    # nhìn như bị lỗi.
    if not chunks:
        return ""

    lines = ["Nguồn tham khảo:"]
    # enumerate(chunks, 1): bắt đầu từ 1 cho khớp [1] model trích. Mặc định 0 là lệch một
    # nguồn, người đọc mở nhầm file.
    for i, chunk in enumerate(chunks, 1):
        label = chunk.title or chunk.source
        lines.append(
            f"  [{i}] {label} · {chunk.location()}"
            f"   ({chunk.similarity:.2f})  {chunk.source}"
        )
    return "\n".join(lines)


def retrieve(
    question: str,
    k: int = DEFAULT_TOP_K,
    doc_type: str | None = None,
) -> list[RetrievedChunk]:
    """Câu hỏi -> list chunk đã lọc ngưỡng. Ném AskError khi DB hỏng.

    Ví dụ: retrieve("embedding là gì?", k=3) -> [RetrievedChunk, RetrievedChunk]

    Đây là chỗ SỬA lỗi thiết kế của week4/ragLite/rag_qa.py::retrieve — bản tuần 4 bắt lỗi DB
    rồi `raise SystemExit` ngay tại chỗ. Hậu quả: dùng lại hàm đó trong Streamlit / FastAPI /
    pytest là nó giết luôn tiến trình, và test "DB hỏng thì báo đúng lỗi" không viết được vì
    process chết trước khi assert. Ở đây chỉ RAISE, main() mới quyết định in gì và exit.
    """
    # Import nặng đặt TRONG hàm: `from search_pg import embed_query` kéo theo
    # sentence_transformers (~5 giây khởi động). Để đầu module thì self-test bằng dữ liệu giả
    # cũng phải chờ 5 giây mỗi lần sửa. Vòng lặp sửa–chạy ngắn quan trọng hơn khối import gọn.
    from search_pg import embed_query
    query_vector = embed_query(question)

    try:
        with get_conn() as conn:
            chunks = search_chunks(conn, query_vector, k=k, doc_type=doc_type)
    except AskError:
        raise                       # đã phân loại rồi, bọc thêm lần nữa là sai nguyên nhân
    except Exception as exc:
        # Chỉ bọc ĐÚNG phần chạm DB, và luôn nhét type(exc).__name__ vào detail: bọc quá rộng
        # thì typo tên cột (ProgrammingError) bị báo thành "DB chưa Up", và mình đi restart
        # Docker 20 phút cho một lỗi chính tả.
        # `from exc` giữ nguyên traceback gốc — bỏ đi là mất dấu vết nguyên nhân thật.
        raise AskError(ErrorKind.DB_UNAVAILABLE,
                       f"{type(exc).__name__}: {exc}") from exc

    return apply_similarity_threshold(chunks)


if __name__ == "__main__":
    question = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))

    if not question:
        # Chế độ self-test: chunk giả, không DB, không model, chạy tức thì.
        fake_chunks = [
            RetrievedChunk(1, "docs/bao-cao.pdf", "pdf", "Báo cáo kỹ thuật 2026", 12, None,
                           "Hệ thống RAG gồm hai giai đoạn: truy xuất và sinh câu trả lời.", 0.62),
            RetrievedChunk(2, "notes/rag.md", "md", "Ghi chú RAG", None, "Chunking",
                           "Chunk 800 ký tự, overlap 100.", 0.41),
            RetrievedChunk(3, "notes/cu.md", "md", None, None, None,
                           "Một đoạn gần như vô quan.", 0.20),
        ]

        print("===== 1. location() + as_prompt_chunk() =====")
        for chunk in fake_chunks:
            print(f"  id={chunk.id}  location={chunk.location()!r}")
            print(f"      as_prompt_chunk -> {chunk.as_prompt_chunk()[1]!r}")

        print("\n===== 2. apply_similarity_threshold =====")
        scenarios = [
            ("có tài liệu liên quan", fake_chunks),
            ("toàn chunk yếu (top-1 dưới ngưỡng)", fake_chunks[2:]),
            ("retrieve rỗng", []),
        ]
        for label, data in scenarios:
            kept = apply_similarity_threshold(data)
            scores = [f"{c.similarity:.2f}" for c in kept]
            print(f"  {label:<38} -> giữ {len(kept)} chunk {scores}")

        print("\n===== 3. format_sources =====")
        print(format_sources(apply_similarity_threshold(fake_chunks)) or "(rỗng)")

        # ✅ ĐẠT khi:
        #   1. chunk pdf hiện "trang 12"; chunk md hiện "Chunking"; chunk không có gì hiện "—"
        #   2. as_prompt_chunk của chunk 3 dùng source làm nhãn (vì title = None)
        #   3. kịch bản 1 giữ 2 chunk (bỏ 0.20) · kịch bản 2 và 3 đều giữ 0 chunk
        #   4. Khối "Nguồn tham khảo" đánh số từ [1], KHÔNG có chữ "None" ở bất kỳ dòng nào
        raise SystemExit(0)

    # Chế độ thật: cần Postgres Up + đã chạy ingest_docs.py.
    doc_type_arg = None
    if "--doc-type" in sys.argv:
        doc_type_arg = sys.argv[sys.argv.index("--doc-type") + 1]

    results = retrieve(question, doc_type=doc_type_arg)
    print(f"❓ {question}  (doc_type={doc_type_arg or 'tất cả'})\n")
    if not results:
        print("Không tìm thấy đoạn nào đủ liên quan (top-1 dưới ngưỡng "
              f"{MIN_SIMILARITY}, hoặc DB chưa có dữ liệu).")
    else:
        print(format_sources(results))
        print(f"\n--- nội dung chunk [1] ---\n{results[0].content[:300]}...")
