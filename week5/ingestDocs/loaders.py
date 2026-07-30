"""
loaders.py — Đọc PDF / DOCX / TXT / MD ra (text, metadata)

Tuần 4 pipeline chỉ ăn được `.md`. Tuần 5 thay tầng ĐỌC FILE, phần sau giữ nguyên:

    tuần 4:  [.md] --open()------------> text --chunk--> embed --> pgvector
    tuần 5:  [.pdf/.docx/.txt/.md] --loaders.py--> Block(text, page, heading) --chunk--> ...

Mọi loader trả về CÙNG một hình dạng (`LoadedDoc`), nên `ingest_docs.py` không cần biết
file gốc là định dạng gì. Thêm .pptx/.html về sau chỉ tốn 1 hàm, không phải sửa pipeline.

Yêu cầu: pip install pypdf pdfplumber python-docx

Chạy thử 1 file (không cần DB, không cần model embedding):
    python loaders.py "D:/Study/tai-lieu-test/bao-cao.pdf"
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

from normalize import normalize_pdf_pages

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class Block:
    """Một mẩu text liền mạch, kèm ĐỊA CHỈ của nó trong tài liệu gốc.

    text    : nội dung
    page    : số trang (PDF). DOCX/MD/TXT không có khái niệm trang -> None
    heading : heading gần nhất phía trên (MD: '## ...', DOCX: style Heading N) -> None

    Ít nhất 1 trong 2 (page / heading) nên có giá trị — đó là thứ dùng để trích nguồn.
    Nối cả tài liệu thành 1 chuỗi rồi mới chunk sẽ VỨT BỎ thông tin này, và mất rồi thì
    không có cách nào gán lại.
    """
    text: str
    page: int | None = None
    heading: str | None = None


@dataclass
class LoadedDoc:
    """Một tài liệu đã đọc xong, chưa chunk, chưa embed.

    source   : đường dẫn TƯƠNG ĐỐI so với folder gốc (phải đọc được với người)
    doc_type : 'pdf' | 'docx' | 'md' | 'txt'
    title    : metadata thật -> heading đầu tiên -> tên file (3 tầng fallback)
    blocks   : danh sách Block
    """
    source: str
    doc_type: str
    title: str
    blocks: list[Block] = field(default_factory=list)

    def total_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".txt"}


def fallback_title(path: str) -> str:
    """Tên file bỏ đuôi — dùng khi tài liệu không khai title (đa số trường hợp thực tế)."""
    return os.path.splitext(os.path.basename(path))[0]


def load_pdf(path: str, source: str) -> LoadedDoc:
    """PDF -> LoadedDoc, MỖI TRANG là 1 Block (page = số trang thật, bắt đầu từ 1).

    normalize_pdf_pages() nhận cả LIST trang vì việc phát hiện header/footer lặp cần nhìn
    toàn bộ tài liệu — xử lý từng trang rời thì không thấy được.

    Bẫy: PDF scan (ảnh chụp) trả "" ở MỌI trang mà không báo lỗi -> total_chars() == 0.
    Bẫy: reader.metadata có thể là None, và .title có thể None hoặc "".
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    raw_pages = [(page.extract_text() or "") for page in reader.pages]
    clean_pages = normalize_pdf_pages(raw_pages)

    # Lọc trang rỗng LÚC TẠO (không lọc list trước rồi mới enumerate) để `i` vẫn là
    # số trang THẬT — người đọc phải mở được đúng trang đó để kiểm chứng.
    blocks = []
    for i, text in enumerate(clean_pages, 1):
        if text.strip():
            blocks.append(Block(text=text, page=i))

    meta = reader.metadata
    title = (meta.title if meta and meta.title else "") or fallback_title(path)

    return LoadedDoc(source=source, doc_type="pdf", title=title, blocks=blocks)


def load_docx(path: str, source: str) -> LoadedDoc:
    """DOCX -> LoadedDoc, gom các paragraph theo HEADING (page=None, heading='...').

    Gom theo heading chứ không phải mỗi paragraph 1 Block: 1 paragraph thường chỉ 1–2 câu
    -> chunk quá vụn, vector gần như vô nghĩa. Gom dưới cùng 1 heading = 1 khối cùng chủ đề.

    Bẫy: `p.style` có thể là None khi paragraph tham chiếu style không có trong styles.xml
    (hay gặp ở file convert từ Google Docs / web) -> `p.style.name` sẽ AttributeError.
    Giới hạn đã biết: doc.paragraphs KHÔNG bao gồm text trong table (doc.tables).
    """
    from docx import Document

    doc = Document(path)
    blocks: list[Block] = []
    buffer: list[str] = []
    current_heading: str | None = None
    first_heading: str | None = None

    def flush():
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(Block(text=text, heading=current_heading))
        buffer = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style_name = p.style.name if p.style else ""
        if style_name.startswith("Heading"):
            flush()
            current_heading = text
            if first_heading is None:
                first_heading = text
        else:
            buffer.append(text)
    flush()      # flush lần cuối: quên dòng này là mất section cuối, KHÔNG báo lỗi

    props = doc.core_properties
    title = (props.title or "") or (first_heading or "") or fallback_title(path)
    return LoadedDoc(source=source, doc_type="docx", title=title, blocks=blocks)


def load_text(path: str, source: str, doc_type: str) -> LoadedDoc:
    """.md -> Block theo heading markdown ; .txt -> 1 Block duy nhất. page luôn = None.

    .txt không có cấu trúc nào để dựa vào mà cắt -> để nguyên, chunker sẽ cắt theo cửa sổ trượt.

    Bẫy encoding: file người khác gửi có thể là utf-8-sig (BOM) hoặc cp1252. Khi phải
    fallback thì IN RA — im lặng fallback là công thức tạo dữ liệu bẩn mà không ai biết.
    """
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=enc) as f:
                text = f.read()
            if enc != "utf-8":
                print(f"   ⚠️  {source}: đọc bằng {enc} (không phải utf-8)")
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Không đọc được {path} với utf-8/utf-8-sig/cp1252")

    if doc_type != "md":
        blocks = [Block(text=text.strip())] if text.strip() else []
        return LoadedDoc(source, doc_type, fallback_title(path), blocks)

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
    blocks, buffer = [], []
    current_heading = first_heading = None

    def flush():
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(Block(text=text, heading=current_heading))
        buffer = []

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            current_heading = m.group(2).strip()
            if first_heading is None:
                first_heading = current_heading
        else:
            buffer.append(line)

    flush()
    title = (first_heading or "") or fallback_title(path)
    return LoadedDoc(source=source, doc_type=doc_type, title=title, blocks=blocks)


def load_any(path: str, root: str) -> LoadedDoc | None:
    """Nhìn đuôi file, gọi đúng loader. Trả None nếu định dạng không hỗ trợ."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        return None

    source = os.path.relpath(path, root).replace(os.sep, "/")
    if ext == ".pdf":
        return load_pdf(path, source)
    if ext == ".docx":
        return load_docx(path, source)
    return load_text(path, source, doc_type=ext.lstrip("."))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Dùng: python loaders.py "<đường dẫn file>"')

    target = sys.argv[1]
    doc = load_any(target, root=os.path.dirname(target) or ".")
    if doc is None:
        raise SystemExit(f"❌ Định dạng không hỗ trợ: {target}")

    print(f"📄 {doc.source}  ·  type={doc.doc_type}  ·  title={doc.title!r}")
    print(f"   {len(doc.blocks)} block  ·  {doc.total_chars()} ký tự")
    if doc.total_chars() == 0:
        print("   ⚠️  0 ký tự — PDF scan (cần OCR) hoặc loader đọc hụt.")
    for b in doc.blocks[:3]:
        addr = f"trang {b.page}" if b.page else (b.heading or "—")
        print(f"\n--- [{addr}] ({len(b.text)} ký tự) ---\n{b.text[:200]}...")
