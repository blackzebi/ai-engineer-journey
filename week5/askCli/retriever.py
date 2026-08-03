"""
retriever.py — Lấy chunk kèm ĐẦY ĐỦ metadata + chặn câu trả lời bịa bằng ngưỡng similarity

Tuần 4 trích nguồn được tới mức "file nào". Tuần 5 (T2) đã nhét page/heading/title vào DB
rồi mà chưa ai dùng — hôm nay lôi ra dùng, và thêm một lớp chống bịa thứ hai:

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

Hai lớp chống bịa, bổ sung cho nhau — cần hiểu rõ khác biệt để nói khi phỏng vấn:
    lớp 1 (tuần 4, prompt): "chỉ trả lời dựa trên đoạn trích" — chữa lúc SINH câu trả lời
    lớp 2 (hôm nay, ngưỡng): chặn TRƯỚC khi sinh — vì prompt tốt tới mấy, đưa cho model
                             3 đoạn hoàn toàn vô quan thì nó vẫn có xu hướng cố ghép đại

Tái dùng nguyên si, KHÔNG viết lại:
    - week4/pgvector/vector_ops.py    : get_conn, to_pgvector
    - week4/ragPipeline/search_pg.py  : embed_query  (import LƯỜI — xem get_query_vector)
    - week4/ragLite/prompt_rag.py     : build_rag_prompt, SYSTEM_PROMPT  (dùng ở stream_answer.py)
    - ./errors.py                     : AskError, ErrorKind

Về `week5/indexTuning/filter_metadata.py::search_filtered` (T3 — ĐÃ LÀM XONG):
    Hàm đó chạy được rồi, nhưng hôm nay vẫn KHÔNG import nó, vì nó trả **6 cột**
    (id, source, doc_type, content, page, similarity) — thiếu `title` và `heading`.
    Thiếu title thì trích nguồn chỉ hiện được đường dẫn file thô; thiếu heading thì
    docx/md/txt (page luôn NULL — xem db_docs.py) mất sạch địa chỉ để chỉ. Mà trích nguồn
    tới nơi tới chốn CHÍNH LÀ thứ hôm nay phải làm ra.

    `search_chunks` dưới đây là bản **8 cột** — cùng một câu SQL, chỉ khác danh sách cột,
    tức là SIÊU TẬP của `search_filtered`. Hệ quả: hai câu SQL gần trùng nhau đang nằm ở
    hai file, và chúng sẽ phân kỳ sau vài tuần (sửa cột ở file này quên file kia).
    ⇒ Đừng để quá hôm nay: xem mục **"Dedup với T3"** trong guide.md. Dọn bằng MỘT commit
      riêng, và làm SAU khi ask.py đã chạy được — đừng refactor giữa lúc đang làm tính năng.

📖 Mỗi hàm chia 2 khối: PHẦN 1 GIẢI THÍCH (đọc) · PHẦN 2 CODE CẦN VIẾT (gõ).

Self-test bằng chunk giả — KHÔNG cần DB, KHÔNG cần model:
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
# Đặt quá cao -> app im lặng với câu hỏi hợp lệ (hỏng còn khó phát hiện hơn là bịa).
MIN_SIMILARITY = 0.35

DEFAULT_TOP_K = 3


@dataclass
class RetrievedChunk:
    """1 chunk lấy từ DB, kèm đủ thứ cần để trích nguồn cho người đọc kiểm chứng.

    VIẾT SẴN — chỉ cần đọc hiểu.

    Vì sao là dataclass chứ không phải tuple 8 phần tử như tuần 4:
        `row[4]` là page hay heading? Đọc lại query mới biết. Với 4 cột còn chịu được,
        8 cột thì mọi chỗ dùng đều thành câu đố. `chunk.page` thì không cần đoán.
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

        page NULL với docx/md/txt là CHUYỆN BÌNH THƯỜNG (xem db_docs.py), không phải lỗi.
        Không xử lý ở đây thì màn hình hiện 'trang None'.
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


# ---------------------------------------------------------------------------
# TODO 1 — query lấy đủ metadata, có lọc doc_type tuỳ chọn
# ---------------------------------------------------------------------------
def search_chunks(
    conn,
    query_vector: list[float],
    k: int = DEFAULT_TOP_K,
    doc_type: str | None = None,
) -> list[RetrievedChunk]:
    """top-k chunk gần nghĩa nhất, kèm metadata. doc_type=None nghĩa là không lọc.

    Ví dụ: search_chunks(conn, vector, k=3, doc_type="pdf") -> 3 RetrievedChunk, doc_type='pdf'
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Vì sao vẫn `ORDER BY embedding <=> %s` chứ không `ORDER BY similarity DESC`?
    #   Hai câu ra cùng kết quả, nhưng chỉ câu đầu dùng được index HNSW (đã tạo ở T3 sáng,
    #   xem hnsw_index.py). Sắp theo biểu thức `1 - (...)` là Postgres bỏ index, quét tuần tự.
    #   Với 5k dòng chưa thấy gì; 500k dòng thì đây là khác biệt giữa 3ms và 2 giây.
    #
    # ▸ Bẫy CHẾT NGƯỜI — thứ tự params phải khớp thứ tự dấu %s XUẤT HIỆN trong câu SQL,
    #   không phải thứ tự mình nghĩ trong đầu. Câu này có 4 chỗ %s:
    #       SELECT ... 1 - (embedding <=> %s)   <- (1) vector
    #       WHERE doc_type = %s                 <- (2) doc_type — CHÈN Ở GIỮA
    #       ORDER BY embedding <=> %s           <- (3) vector (lần 2!)
    #       LIMIT %s                            <- (4) k
    #   Đảo (1) và (2): nếu psycopg cast được thì câu lệnh CHẠY BÌNH THƯỜNG và trả về rác.
    #   -> build list params theo đúng trình tự lắp SQL, đừng viết cả cụm rồi sửa sau.
    #
    # ▸ Bẫy 2 — chỉ nối phần CẤU TRÚC ("WHERE doc_type = %s") vào chuỗi SQL. GIÁ TRỊ luôn
    #   đi qua params. f-string giá trị vào SQL = SQL injection, kể cả khi "chỉ mình dùng".
    #
    # ▸ Bẫy 3 — lọc doc_type + index vector = post-filter: Postgres lấy top-k theo vector
    #   TRƯỚC rồi mới bỏ dòng không khớp -> xin k=5 có thể nhận về 2. Không phải bug của
    #   câu SQL này, và cũng KHÔNG chữa được bằng cách viết SQL khéo hơn.
    #   T3 hôm qua đã đo được hiện tượng đó thật rồi — chạy lại một lần để nhìn tận mắt
    #   thay vì tin lời người khác:
    #       python ../indexTuning/filter_metadata.py --doc-type pdf
    #   `inspect_filter_plan` in ra filter_position = 'post' cùng execution_ms của hai câu
    #   (có WHERE / không WHERE). Dán con số đó vào note T4: đó là câu trả lời CÓ SỐ LIỆU
    #   cho "vì sao --doc-type pdf đôi khi trả về ít hơn --k", và là chất liệu phỏng vấn tốt.
    #
    # ▸ Nhắc cú pháp: `cur.fetchall()` trả list tuple THÔ theo đúng thứ tự cột trong SELECT.
    #   Đổi thứ tự cột ở SELECT mà quên đổi thứ tự unpack -> title và content hoán chỗ cho
    #   nhau, không báo lỗi. Viết SELECT và dòng unpack sát nhau, sửa thì sửa cả hai.

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — literal vector + phần SELECT (params mở đầu bằng vector):
    #     query_literal = to_pgvector(query_vector)
    #     sql = ("SELECT id, source, doc_type, title, page, heading, content, "
    #            "1 - (embedding <=> %s) AS similarity FROM chunks")
    #     params: list = [query_literal]
    #
    # Bước 2 — WHERE tuỳ chọn (append đúng vị trí %s):
    #     if doc_type:
    #         sql += " WHERE doc_type = %s"
    #         params.append(doc_type)
    #
    # Bước 3 — ORDER BY + LIMIT:
    #     sql += " ORDER BY embedding <=> %s LIMIT %s;"
    #     params += [query_literal, k]
    #
    # Bước 4 — chạy, đổi tuple thô thành RetrievedChunk:
    #     with conn.cursor() as cur:
    #         cur.execute(sql, tuple(params))
    #         rows = cur.fetchall()
    #     return [RetrievedChunk(*row) for row in rows]
    #     # RetrievedChunk(*row): bung 8 phần tử vào 8 tham số THEO THỨ TỰ. Chỉ đúng khi
    #     # thứ tự cột SELECT khớp thứ tự field trong dataclass — xem Bẫy 3 ở trên.
    #
    # ✅ Kiểm tra nhanh: gọi với doc_type="pdf" -> MỌI phần tử phải có .doc_type == 'pdf'.
    #    Nhận về ít hơn k dòng chính là post-filter, ghi vào note chứ đừng sửa vội.
    query_literal = to_pgvector(query_vector)
    sql = ("SELECT id, source, doc_type, title, page, heading, content, "
           "1 - (embedding <=> %s) AS similarity FROM chunks")
    params: list = [query_literal]

    if doc_type:
        sql += " WHERE doc_type = %s"
        params.append(doc_type)

    sql += " ORDER BY embedding <=> %s LIMIT %s;"
    params += [query_literal, k]

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return [RetrievedChunk(*row) for row in rows]


# ---------------------------------------------------------------------------
# TODO 2 — lớp chống bịa thứ hai: ngưỡng similarity
# ---------------------------------------------------------------------------
def apply_similarity_threshold(
    chunks: list[RetrievedChunk],
    min_similarity: float = MIN_SIMILARITY,
) -> list[RetrievedChunk]:
    """Top-1 quá thấp -> trả list RỖNG (không đủ căn cứ). Ngược lại -> bỏ các chunk yếu.

    Ví dụ: [0.62, 0.41, 0.20] với ngưỡng 0.35 -> giữ 2 chunk đầu
           [0.28, 0.22, 0.19] với ngưỡng 0.35 -> trả []  (top-1 đã dưới ngưỡng)
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Vì sao xét TOP-1 riêng chứ không chỉ lọc từng chunk?
    #   Hai tình huống khác hẳn nhau về bản chất:
    #     (a) top-1 = 0.62, các chunk sau = 0.20 -> CÓ tài liệu liên quan, chỉ là ít.
    #         Giữ chunk mạnh, bỏ chunk yếu — chunk yếu chỉ làm loãng context và tốn token.
    #     (b) top-1 = 0.28 -> tài liệu KHÔNG chứa câu trả lời. Đưa gì cho model cũng vô ích.
    #         Trả [] để tầng trên nói "không tìm thấy" mà không tốn một đồng token nào.
    #   Chỉ lọc từng chunk thì (b) rơi vào cùng nhánh với (a) và mình mất khả năng phân biệt.
    #
    # ▸ Bẫy 1 — `chunks[0]` khi chunks rỗng -> IndexError. retrieve rỗng là chuyện THƯỜNG
    #   XUYÊN (DB chưa ingest, lọc doc_type không khớp), phải guard ngay dòng đầu.
    #
    # ▸ Bẫy 2 — hàm này giả định list đã sắp giảm dần theo similarity. Đúng, vì SQL
    #   ORDER BY distance tăng dần. Nhưng đó là giả định NGẦM giữa 2 hàm — ai đó sort lại
    #   ở giữa là hỏng im lặng. Viết giả định đó vào docstring (đã viết) là cách rẻ nhất
    #   để nó không biến mất khỏi trí nhớ.
    #
    # ▸ Bẫy 3 — đừng in cảnh báo bên trong hàm này. Hàm lọc thì chỉ lọc; ai gọi nó sẽ
    #   quyết định in gì. Trộn logic với I/O là thứ làm hàm không test được.
    #
    # ▸ Nhắc cú pháp: list comprehension có điều kiện — [c for c in chunks if c.similarity >= x]

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — guard rỗng (chặn Bẫy 1):
    #     if not chunks:
    #         return []
    #
    # Bước 2 — top-1 dưới ngưỡng thì bỏ TẤT CẢ:
    #     if chunks[0].similarity < min_similarity:
    #         return []
    #
    # Bước 3 — còn lại thì chỉ giữ chunk đạt ngưỡng:
    #     return [c for c in chunks if c.similarity >= min_similarity]
    #
    # ✅ Kiểm tra nhanh: self-test bên dưới có 3 kịch bản. Hỏi một câu hoàn toàn ngoài lề
    #    ("giá vàng hôm nay bao nhiêu?") trên dữ liệu thật -> phải ra "không tìm thấy"
    #    và log phải cho thấy KHÔNG có lời gọi LLM nào.
    if not chunks:
        return []

    if chunks[0].similarity < min_similarity:
        return []

    return [ c for c in chunks if c.similarity >= min_similarity]


# ---------------------------------------------------------------------------
# TODO 3 — khối trích nguồn hiện dưới câu trả lời
# ---------------------------------------------------------------------------
def format_sources(chunks: list[RetrievedChunk]) -> str:
    """Danh sách nguồn đánh số, khớp với [1][2][3] mà model trích trong câu trả lời.

    Mong muốn:
        Nguồn tham khảo:
          [1] Báo cáo kỹ thuật 2026 · trang 12   (0.62)  bao-cao.pdf
          [2] Ghi chú RAG · Chunking             (0.41)  notes/rag.md
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Vì sao in CẢ title lẫn source (tên file)?
    #   title để người đọc nhận ra tài liệu; source để họ MỞ được đúng file mà kiểm chứng.
    #   Trích nguồn mà người ta không lần ra được bản gốc thì chỉ là trang trí.
    #
    # ▸ Vì sao in điểm similarity ra màn hình cho người dùng thấy?
    #   Nó là tín hiệu độ tin cậy. 0.61 và 0.36 cho ra hai mức tin khác hẳn nhau, và người
    #   dùng có quyền biết. Đây cũng là thứ giúp mình chỉnh MIN_SIMILARITY: nhìn điểm thật
    #   của những câu hỏi thật, không ngồi đoán.
    #
    # ▸ Bẫy 1 — hàm này bị gọi với list rỗng (retrieve không ra gì). Trả "" và để chỗ gọi
    #   tự xử, đừng in "Nguồn tham khảo:" rồi bỏ trống bên dưới — nhìn như bị lỗi.
    #
    # ▸ Bẫy 2 — title có thể None (dữ liệu ingest trước khi migrate). `or self.source` là
    #   fallback, đã có sẵn trong as_prompt_chunk(); ở đây cũng cần y vậy.
    #
    # ▸ Nhắc cú pháp: `enumerate(chunks, 1)` — tham số thứ hai là số BẮT ĐẦU. Mặc định là 0,
    #   in ra "[0]" trong khi model trích "[1]" -> lệch một, người đọc mở nhầm nguồn.

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — guard rỗng:
    #     if not chunks:
    #         return ""
    #
    # Bước 2 — dựng từng dòng:
    #     lines = ["Nguồn tham khảo:"]
    #     for i, chunk in enumerate(chunks, 1):
    #         label = chunk.title or chunk.source
    #         lines.append(
    #             f"  [{i}] {label} · {chunk.location()}"
    #             f"   ({chunk.similarity:.2f})  {chunk.source}"
    #         )
    #
    # Bước 3 — nối lại:
    #     return "\n".join(lines)
    #
    # ✅ Kiểm tra nhanh: cho 1 chunk pdf (có page) và 1 chunk md (page=None, có heading) —
    #    dòng md phải hiện heading, KHÔNG được hiện chữ "None" ở bất kỳ đâu.
    if not chunks:
        return ""

    lines = ["Nguồn tham khảo:"]
    for i, chunk in enumerate(chunks, 1):
        label = chunk.title or chunk.source
        lines.append(
            f"  [{i}] {label} · {chunk.location()}"
            f"   ({chunk.similarity:.2f})  {chunk.source}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TODO 4 — ráp thành một lời gọi duy nhất cho ask.py
# ---------------------------------------------------------------------------
def retrieve(
    question: str,
    k: int = DEFAULT_TOP_K,
    doc_type: str | None = None,
) -> list[RetrievedChunk]:
    """Câu hỏi -> list chunk đã lọc ngưỡng. Ném AskError khi DB hỏng.

    Ví dụ: retrieve("embedding là gì?", k=3) -> [RetrievedChunk, RetrievedChunk]
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Đây chính là chỗ SỬA lỗi thiết kế của week4/ragLite/rag_qa.py::retrieve.
    #   Bản tuần 4 bắt lỗi DB rồi `raise SystemExit(...)` ngay tại chỗ. Hậu quả:
    #   dùng lại hàm đó trong Streamlit / FastAPI / pytest là nó giết luôn tiến trình,
    #   và test "DB hỏng thì báo đúng lỗi" không viết được vì process chết trước khi assert.
    #   Hôm nay: ném AskError(DB_UNAVAILABLE) — main() sẽ quyết định in gì và exit bao nhiêu.
    #
    # ▸ Bẫy 1 — `except Exception` bọc quanh cả cụm sẽ nuốt luôn lỗi lập trình của chính
    #   mình (typo tên cột -> psycopg.ProgrammingError -> báo nhầm thành "DB chưa Up",
    #   và mình đi restart Docker suốt 20 phút). Chỉ bọc ĐÚNG lời gọi có thể lỗi mạng,
    #   và luôn nhét `type(exc).__name__` vào detail để nhìn phát biết mình đoán sai chỗ nào.
    #
    # ▸ Bẫy 2 — import nặng đặt trong hàm, không đặt đầu file: `from search_pg import
    #   embed_query` kéo theo sentence_transformers (~5 giây khởi động). Để ở đầu module thì
    #   `python retriever.py` chỉ chạy self-test dữ liệu giả cũng phải chờ 5 giây mỗi lần sửa.
    #   Vòng lặp sửa–chạy ngắn quan trọng hơn sự gọn gàng của khối import.
    #
    # ▸ Nhắc cú pháp: `with get_conn() as conn:` tự đóng connection kể cả khi có exception.
    #   Đừng conn.close() thủ công — lỗi giữa chừng là rò connection.

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — embed câu hỏi (import lười, xem Bẫy 2):
    #     from search_pg import embed_query      # noqa: E402 — cố ý import trong hàm
    #     query_vector = embed_query(question)
    #
    # Bước 2 — truy vấn, chỉ bọc try quanh phần chạm DB:
    #     try:
    #         with get_conn() as conn:
    #             chunks = search_chunks(conn, query_vector, k=k, doc_type=doc_type)
    #     except AskError:
    #         raise                              # đã phân loại rồi, đừng bọc thêm lần nữa
    #     except Exception as exc:
    #         raise AskError(ErrorKind.DB_UNAVAILABLE,
    #                        f"{type(exc).__name__}: {exc}") from exc
    #     # `from exc` giữ nguyên traceback gốc — bỏ đi là mất dấu vết nguyên nhân thật
    #
    # Bước 3 — lọc ngưỡng rồi trả:
    #     return apply_similarity_threshold(chunks)
    #
    # ✅ Kiểm tra nhanh: tắt Docker (`docker compose stop`) rồi chạy `python ask.py "test"` —
    #    phải ra 3 dòng gọn gàng có chữ "docker compose up -d", KHÔNG có chữ "Traceback".

    from search_pg import embed_query
    query_vector = embed_query(question)

    try:
        with get_conn() as conn:
            chunks = search_chunks(conn, query_vector, k=k, doc_type=doc_type)
    except AskError:
        raise
    except Exception as exc:
        raise AskError(ErrorKind.DB_UNAVAILABLE,
                       f"{type(exc).__name__}: {exc}") from exc

    return apply_similarity_threshold(chunks)


if __name__ == "__main__":
    question = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))

    if not question:
        # ── Chế độ self-test: chunk giả, không DB, không model, chạy tức thì ──────────
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

    # ── Chế độ thật: cần Postgres Up + đã chạy ingest_docs.py ────────────────────────
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
