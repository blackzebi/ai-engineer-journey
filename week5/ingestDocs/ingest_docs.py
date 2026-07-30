"""
ingest_docs.py — Pipeline ingest đa định dạng vào pgvector

    INPUT :  1 folder chứa .pdf / .docx / .md / .txt
    OUTPUT:  bảng `chunks` có N dòng = vector + content + doc_type + title + page/heading

Bản nâng cấp của `week4/ragPipeline/ingest.py`. Khác biệt cốt lõi ở chỗ metadata:

    tuần 4:  folder .md ──► chunk_folder() ──► [(source, content)] ──► embed ──► INSERT
                            └─ nối cả file thành 1 chuỗi -> MẤT trang/heading

    tuần 5:  folder mixed ─► load_any() ──► LoadedDoc[Block(text, page, heading)]
                                              └─ chunk TỪNG BLOCK, metadata đi kèm suốt đường
                            ──► embed ──► INSERT (…, doc_type, title, page, heading, idx)

Tái dùng nguyên si từ tuần 4:
    - week4/ragPipeline/chunker.py  : chunk_text()   (cửa sổ trượt + overlap)
    - week4/pgvector/vector_ops.py  : get_conn, EMBED_DIM

Chạy:
    python db_docs.py                                          # migrate schema trước
    python ingest_docs.py "D:/Study/tai-lieu-test"
    python ingest_docs.py "D:/Study/tai-lieu-test" --force     # bỏ cache hash, nạp lại hết
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "..", "week4", "pgvector"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from chunker import chunk_text  # noqa: E402
from db_docs import (  # noqa: E402
    create_documents_table,
    delete_document,
    file_hash,
    insert_chunks_batch,
    migrate_chunks_table,
    record_document,
    should_ingest,
    stats_by_doc_type,
)
from loaders import SUPPORTED_EXTS, LoadedDoc, load_any  # noqa: E402
from vector_ops import EMBED_DIM, get_conn  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# PHẢI trùng model đã dùng ở tuần 4. Khác model = vector cũ và vector mới thuộc 2 "không
# gian nghĩa" khác nhau -> so sánh ra kết quả rác mà KHÔNG hề báo lỗi.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64
MIN_CHUNK_CHARS = 50          # chunk ngắn hơn mức này thì bỏ (xem blocks_to_chunks)

DOCS_DIR = os.getenv("DOCS_DIR", r"D:\Study\tai-lieu-test")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy loading: load model 1 lần rồi cache (khởi tạo mất vài giây)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def collect_files(folder: str) -> list[str]:
    """Duyệt đệ quy, trả đường dẫn của mọi file có đuôi trong SUPPORTED_EXTS.

    Bỏ file ẩn ('.') và file tạm của Word ('~$bao-cao.docx' sinh ra khi mở file —
    python-docx đọc nó sẽ nổ).

    sorted() để mỗi lần chạy ra cùng thứ tự -> dễ so log giữa 2 lần chạy khi debug.
    """
    out = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.startswith((".", "~$")):
                continue
            if os.path.splitext(name)[1].lower() not in SUPPORTED_EXTS:
                continue
            out.append(os.path.join(root, name))
    return sorted(out)


def blocks_to_chunks(doc: LoadedDoc) -> list[tuple[str, int | None, str | None, int]]:
    """LoadedDoc -> [(content, page, heading, chunk_index), ...].

    Chunk TỪNG BLOCK thay vì nối cả tài liệu rồi cắt, vì 2 lý do:
      1. Nối hết là mất page/heading -> hỏng mục tiêu trích nguồn.
      2. Nối hết còn tạo "chunk lai": nửa đầu là cuối trang 5, nửa sau là đầu trang 6, hai
         chủ đề khác nhau -> vector là TRUNG BÌNH của 2 ý -> đúng cái bệnh mà chunking sinh
         ra để chữa.

    Cái giá phải trả: block ngắn (heading + 1 câu) tạo chunk tí hon, vector nhiễu.
    Đã chọn cách bỏ chunk < MIN_CHUNK_CHARS (cách khác: gộp block ngắn vào block kế tiếp).
    """
    out = []
    # idx khai báo NGOÀI vòng lặp block: chunk_index phải liên tục trên TOÀN tài liệu, vì
    # unique index là (source, chunk_index) — reset theo block thì trùng khoá ngay.
    idx = 0
    for block in doc.blocks:
        for piece in chunk_text(block.text):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            out.append((piece, block.page, block.heading, idx))
            # Tăng SAU append, trong cùng nhánh: đặt trước `continue` thì chunk bị bỏ vẫn
            # "ăn" một index -> dãy index thủng lỗ, nhìn DB tưởng mất dữ liệu.
            idx += 1
    return out


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed 1 lô text -> list vector.

    Giữ assert số chiều để bắt sớm trường hợp đổi MODEL_NAME mà quên đổi EMBED_DIM —
    không có nó thì lỗi chỉ hiện ra tận lúc INSERT, xa chỗ gây lỗi.
    """
    if not texts:
        return []
    vectors = get_model().encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    assert len(vectors[0]) == EMBED_DIM, \
        f"Model trả {len(vectors[0])} chiều, bảng khai {EMBED_DIM}"
    return vectors.tolist()


def ingest_one(conn, path: str, root: str, force: bool = False) -> tuple[str, int]:
    """Ingest 1 file. Trả (trạng_thái, số_chunk), trạng_thái ∈ {skip, new, changed, empty, error}.

    Thứ tự các bước có chủ đích: hash -> should_ingest -> RETURN SỚM nếu skip -> mới đọc
    file và embed. Cắt sớm ở đúng chỗ đắt tiền (embed) là lý do tồn tại của cache hash.

    try/except chỉ bọc quanh phần ĐỌC FILE: 1 PDF hỏng không được giết cả lượt chạy 200 file.
    Nhưng KHÔNG bọc lỗi DB — DB hỏng mà vẫn chạy tiếp 200 file là vô nghĩa, nên để nó nổ.
    """
    source = os.path.relpath(path, root).replace(os.sep, "/")
    fhash = file_hash(path)
    action = "changed" if force else should_ingest(conn, source, fhash)
    if action == "skip":
        return ("skip", 0)

    try:
        doc = load_any(path, root)
    except Exception as e:
        # Bắt Exception + in kèm lỗi gốc, không dùng `except:` trần (nó nuốt cả typo,
        # KeyError... rồi báo nhầm nguyên nhân).
        print(f"   ❌ {source}: {type(e).__name__}: {e}")
        return ("error", 0)

    if doc is None or doc.total_chars() == 0:
        print(f"   ⚠️  {source}: 0 ký tự (PDF scan? loader đọc hụt?)")
        return ("empty", 0)

    if action == "changed":
        delete_document(conn, source)

    pieces = blocks_to_chunks(doc)
    if not pieces:
        return ("empty", 0)

    vectors = embed_texts([p[0] for p in pieces])
    rows = [
        (source, content, vec, doc.doc_type, doc.title, page, heading, idx)
        for (content, page, heading, idx), vec in zip(pieces, vectors)
    ]
    insert_chunks_batch(conn, rows)

    # Ghi sổ SAU CÙNG — xem docstring record_document().
    record_document(conn, source, doc.doc_type, doc.title, fhash, len(rows))
    return (action, len(rows))


def ingest_folder(folder: str, force: bool = False) -> dict[str, int]:
    """Ingest cả folder, trả {'new': n, 'skip': n, 'changed': n, 'empty': n, 'error': n}.

    Mở 1 connection cho CẢ folder: mỗi lần connect tốn handshake TCP + xác thực.

    Gọi lại migration mỗi lần chạy là cố ý — 2 hàm đó idempotent, đổi lại người dùng không
    bao giờ gặp lỗi "chưa migrate".
    """
    files = collect_files(folder)
    stats = {"new": 0, "skip": 0, "changed": 0, "empty": 0, "error": 0}
    if not files:
        raise SystemExit(f"Không tìm thấy file nào hỗ trợ trong {folder}")

    with get_conn() as conn:
        migrate_chunks_table(conn)
        create_documents_table(conn)
        # In tiến độ từng file, không chỉ in tổng kết cuối: embed 200 file mất vài phút,
        # màn hình đứng im thì không biết đang chạy hay đã treo.
        for i, path in enumerate(files, 1):
            action, n = ingest_one(conn, path, folder, force=force)
            stats[action] += 1
            name = os.path.basename(path)
            print(f"[{i:2}/{len(files)}] {action:<8} {name:<40} {n or '—'} chunk")

        print("\n📊 Phân bố theo định dạng:")
        for doc_type, n_chunks, n_files in stats_by_doc_type(conn):
            print(f"   {doc_type or '(null)':<6} {n_chunks:>5} chunk / {n_files} file")

    return stats


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = args[0] if args else DOCS_DIR
    force = "--force" in sys.argv

    if not os.path.isdir(folder):
        raise SystemExit(
            f"❌ Không thấy folder: {folder}\n"
            f'   Dùng: python ingest_docs.py "<folder tài liệu>" [--force]'
        )

    print(f"📂 Ingest từ: {folder}" + ("  (--force: nạp lại tất cả)" if force else ""))
    stats = ingest_folder(folder, force=force)

    print("\n🎉 Xong:", ", ".join(f"{k}={v}" for k, v in stats.items()))
    print("   Kiểm chứng chống trùng: chạy lại đúng lệnh trên, phải ra skip=<tất cả>, new=0,")
    print("   và tổng chunk trong DB không đổi. 'Chạy không lỗi' không phải bằng chứng.")


if __name__ == "__main__":
    main()
