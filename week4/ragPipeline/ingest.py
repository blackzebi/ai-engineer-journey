"""
ingest.py — Pipeline ingest: folder .md -> chunk -> embed -> INSERT pgvector (T4 tuần 4)

    INPUT :  1 folder chứa file .md (mặc định: kho notes ở D:\\Study)
    OUTPUT:  bảng `chunks` trong Postgres có N dòng, mỗi dòng kèm vector 384 chiều

Đây là chỗ **ghép 3 mảnh đã học rời rạc** thành 1 pipeline chạy được:
    chunker.py (hôm nay)  +  embed model (tuần 3)  +  bảng chunks (block sáng nay)

Chạy:
    python ingest.py                        # dùng NOTES_DIR mặc định
    python ingest.py "D:/Study/AI-Engineer-Study/python-knowledge"
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pgvector"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from chunker import chunk_folder  # noqa: E402
from vector_ops import (  # noqa: E402
    EMBED_DIM,
    create_chunks_table,
    get_conn,
    insert_chunks_batch,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# PHẢI trùng model tuần 3 (semanticSearch) — nếu không, vector trong DB và vector câu hỏi
# thuộc 2 "không gian nghĩa" khác nhau, so sánh ra kết quả rác mà không hề báo lỗi.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

NOTES_DIR = os.getenv(
    "NOTES_DIR",
    r"D:\Study\AI-Engineer-Study\python-knowledge",
)

BATCH_SIZE = 64  # embed theo lô cho nhanh, đừng encode từng chunk một


def embed_texts(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    """Embed 1 lô text -> list vector.

    Kiểm tra 1 lần cho chắc: len(vectors[0]) phải == EMBED_DIM (384).
    Lệch là do đổi model -> phải sửa cả EMBED_DIM lẫn tạo lại bảng.
    """
    vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    if len(vectors[0]) == EMBED_DIM:
        return vectors.tolist()
    else:
        raise SystemExit(f"chiều vectors phải bằng 384 theo embed_dim")


def ingest(folder: str, reset: bool = True) -> int:
    """Chạy trọn pipeline, trả về số chunk đã insert.

    Luồng:
        1. chunk_folder(folder)              -> [(source, content), ...]
        2. embed_texts([content, ...])       -> [vector, ...]
        3. create_chunks_table(conn, drop=reset)
        4. insert_chunks_batch(conn, [(source, content, vector), ...])

    Vì sao reset=True mặc định: chạy ingest 2 lần mà không xoá -> mỗi chunk có 2 bản,
    top-5 trả về toàn cặp trùng nhau. (Cách "xịn" hơn ở tuần sau: UNIQUE(source, content)
    hoặc xoá theo source trước khi insert lại — để dành khi làm incremental update.)
    """
    pairs = chunk_folder(folder)
    if not pairs:
        raise SystemExit(f"Không tìm thấy file .md nào trong {folder}")
    print(f"📄 {len(pairs)} chunk — bắt đầu embed (lần đầu sẽ tải model ~470MB)")

    model = SentenceTransformer(MODEL_NAME)
    vectors = embed_texts([content for _src, content in pairs], model)
    assert len(vectors[0]) == EMBED_DIM, f"Model trả {len(vectors[0])} chiều, bảng khai {EMBED_DIM}"

    rows = [(src, content, vec) for (src, content), vec in zip(pairs, vectors)]
    with get_conn() as conn:
        create_chunks_table(conn, drop=reset)
        n = insert_chunks_batch(conn, rows)
    return n


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else NOTES_DIR
    if not os.path.isdir(folder):
        raise SystemExit(f"❌ Không thấy folder: {folder}\n   Truyền đường dẫn khác: python ingest.py <folder>")

    print(f"📂 Ingest từ: {folder}")
    n = ingest(folder)
    print(f"\n🎉 Đã ingest {n} chunk vào bảng chunks.")
    print("   Kiểm tra: docker exec -it rag-pg psql -U postgres -d study_rag "
          '-c "SELECT count(*), count(distinct source) FROM chunks;"')
    print("   Tiếp theo: python search_pg.py \"câu hỏi của bạn\"")


if __name__ == "__main__":
    main()
