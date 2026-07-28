"""
chunker.py — Cắt file .md thành chunk (T4 tuần 4, block PROJECT)

Mảnh #1 của pipeline: [folder .md] --chunker--> [list chunk] --embed--> [INSERT pgvector]

Vì sao phải chunk: model embedding có giới hạn độ dài, và quan trọng hơn — 1 vector cho cả
file dài 3000 chữ là "trung bình cộng của mọi chủ đề" → search ra thứ chung chung, vô dụng.
Chunk nhỏ = mỗi vector đại diện 1 ý = retrieval chính xác.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Chiến lược hiện tại: cắt theo KÝ TỰ có overlap (đơn giản nhất, đủ dùng).
# Bước tiếp theo: cắt theo heading markdown / theo câu và so kết quả.
CHUNK_SIZE = 800      # ~ 1-2 đoạn văn. Nhỏ quá -> mất ngữ cảnh; to quá -> loãng nghĩa.
CHUNK_OVERLAP = 100   # phần chồng lấn giữa 2 chunk kề nhau, để câu bị cắt giữa không mất nghĩa.


def read_markdown_files(folder: str) -> list[tuple[str, str]]:
    """Duyệt đệ quy `folder`, trả về [(relative_path, nội dung), ...] cho mọi file .md.

    `relative_path` sẽ thành cột `source` trong DB → phải đọc được với người
    (vd 'Tuan-02_.../CN/README.md'), đừng lưu đường dẫn tuyệt đối.

    Bẫy: thiếu encoding="utf-8" -> Windows đọc tiếng Việt ra mojibake / UnicodeDecodeError.
    """
    results = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, folder).replace(os.sep, "/")
            with open(path, encoding="utf-8") as f:
                results.append((rel, f.read()))
    return results


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Cắt 1 chuỗi dài thành các chunk `chunk_size` ký tự, chồng lấn `overlap`.

    Cửa sổ trượt:
        chunk 1: [0        : 800]
        chunk 2: [700      : 1500]     <- lùi lại 100 ký tự = overlap
        chunk 3: [1400     : 2200]
        ...
    Bước nhảy = chunk_size - overlap.

    Bẫy CHẾT NGƯỜI: nếu overlap >= chunk_size thì step <= 0 -> range() lặp vô hạn / lỗi.
      -> thêm guard: if overlap >= chunk_size: raise ValueError(...)

    Cải tiến khả dĩ: cắt ưu tiên ở ranh giới đoạn ("\\n\\n") thay vì giữa từ,
    để chunk không đứt ngang câu.
    """
    text = text.strip()
    if not text:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start: start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
        if overlap >= chunk_size:
            raise ValueError("error infinity loop")
    return chunks


def chunk_folder(folder: str) -> list[tuple[str, str]]:
    """Ghép 2 hàm trên: folder -> [(source, chunk_content), ...] sẵn sàng đem đi embed."""
    out = []
    for rel_path, content in read_markdown_files(folder):
        for piece in chunk_text(content):
            out.append((rel_path, piece))
    return out


if __name__ == "__main__":
    # Chạy thử độc lập trước khi đụng tới DB:
    #     python chunker.py "D:/Study/AI-Engineer-Study/python-knowledge"
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = chunk_folder(folder)
    print(f"✅ {len(rows)} chunk từ folder {folder}")
    for source, piece in rows[:3]:
        print(f"\n--- {source} ({len(piece)} ký tự) ---\n{piece[:200]}...")
