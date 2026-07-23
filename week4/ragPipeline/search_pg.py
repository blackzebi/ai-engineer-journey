"""
search_pg.py — Hỏi 1 câu -> embed -> top-5 chunk từ pgvector (T4 tuần 4)

Mảnh cuối của pipeline. Và là dịp **đối chiếu 2 cách làm cùng một việc**:

    tuần 3: vectors.json + cosine tự viết + sort()      (semanticSearch/semantic_search.py)
    hôm nay: bảng chunks + toán tử <=> + ORDER BY       (file này)

Cùng 1 câu hỏi — kết quả có giống nhau không? Vì sao? Đó là câu hỏi chính của buổi hôm nay,
không phải "chạy được hay chưa".

Code KHUNG — điền các chỗ `# TODO`. Gợi ý ở guide.md.

Chạy:
    python search_pg.py "làm sao gọi mạng bất đồng bộ trong Python?"
    python search_pg.py "..." --compare        # chạy kèm bản thuần Python để so
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pgvector"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from vector_ops import get_conn, search_top_k  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# PHẢI trùng model dùng ở ingest.py. Khác model = so vector khác không gian = kết quả rác.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load model 1 lần rồi cache (khởi tạo mất vài giây)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(question: str) -> list[float]:
    """Embed câu hỏi thành 1 vector.

    # TODO 1:
    #   return get_model().encode(question).tolist()
    #
    # Lưu ý: sentence-transformers KHÔNG có input_type (khác Voyage/OpenAI) — dùng chung
    # 1 model cho cả document lẫn query. Với Voyage thì phải phân biệt, đừng quen tay.
    """
    return get_model().encode(question).tolist()


def search(question: str, k: int = TOP_K) -> list[tuple[int, str, str, float]]:
    """Câu hỏi -> top-k chunk từ pgvector. Trả [(id, source, content, similarity), ...].

    # TODO 2:
    #   vec = embed_query(question)
    #   with get_conn() as conn:
    #       return search_top_k(conn, vec, k=k)
    """
    vec = embed_query(question)
    with get_conn() as conn:
        return search_top_k(conn, vec, k=k)


def print_results(title: str, rows: list[tuple], elapsed: float) -> None:
    """In kết quả gọn gàng để dễ so 2 bên."""
    print(f"\n=== {title} ({elapsed*1000:.0f} ms) ===")
    for i, row in enumerate(rows, 1):
        # rows từ pgvector: (id, source, content, similarity)
        _id, source, content, sim = row
        snippet = " ".join(content.split())[:110]
        print(f"{i}. [{sim:.4f}] {source}\n   {snippet}...")


def compare_with_python(question: str, k: int = TOP_K) -> None:
    """Chạy lại cùng câu hỏi bằng semantic search thuần Python (tuần 3) để đối chiếu.

    # TODO 3 (làm sau khi TODO 1-2 chạy được):
    #   sys.path.append(os.path.join(os.path.dirname(__file__), "..", "semanticSearch"))
    #   from semantic_search import search as py_search
    #   t0 = time.perf_counter()
    #   rows = [(0, "corpus.txt", text, score) for text, score in py_search(question, top_k=k)]
    #   print_results("Thuần Python (vectors.json + cosine tự viết)", rows, time.perf_counter() - t0)
    #
    # ⚠️ Hai bên đang chạy trên 2 KHO KHÁC NHAU (corpus.txt ~20 dòng vs toàn bộ notes .md),
    #    nên đừng kỳ vọng top-5 trùng khít. Thứ đáng so là:
    #      - Thứ hạng tương đối có hợp lý như nhau không?
    #      - Điểm similarity 2 bên có cùng thang không? (cả hai đều là cosine -> phải cùng thang)
    #      - Tốc độ: bên nào nhanh hơn ở quy mô này? Còn ở quy mô 100k chunk thì sao?
    """
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "semanticSearch"))
    from semantic_search import search as py_search
    t0 = time.perf_counter()
    rows = [(0, "corpus.txt", text, score) for text, score in py_search(question, top_k=k)]
    print_results("Thuần Python (vectors.json + cosine tự viết)", rows, time.perf_counter() - t0)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit('Dùng: python search_pg.py "câu hỏi của bạn" [--compare]')

    question = " ".join(args)
    print(f"❓ {question}")

    t0 = time.perf_counter()
    rows = search(question)
    print_results("pgvector (ORDER BY embedding <=> query)", rows, time.perf_counter() - t0)

    if "--compare" in sys.argv:
        compare_with_python(question)


if __name__ == "__main__":
    main()
