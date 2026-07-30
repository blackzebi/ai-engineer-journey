"""
so_sanh_pdf.py — Công cụ tạm để so pypdf vs pdfplumber (T2 tuần 5, task 'so sánh')

File này KHÔNG thuộc pipeline. Nó là dụng cụ đo, chạy vài lần rồi có thể xoá (hoặc giữ
lại trong repo như bằng chứng "tôi có đo thật" — người đọc portfolio thích thứ này).

Chạy:
    python so_sanh_pdf.py "D:/Study/tai-lieu-test/bao-cao.pdf"
    python so_sanh_pdf.py "D:/Study/tai-lieu-test/bao-cao.pdf" 3     # xem trang 3
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHARS = 300      # số ký tự in ra để so bằng mắt


def doc_bang_pypdf(path: str, page_no: int) -> tuple[str, float]:
    """Trả (text của trang page_no, thời gian đọc CẢ file tính bằng giây)."""
    from pypdf import PdfReader

    t0 = time.perf_counter()
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    elapsed = time.perf_counter() - t0
    return pages[page_no - 1] if page_no <= len(pages) else "", elapsed


def doc_bang_pdfplumber(path: str, page_no: int, layout: bool = False) -> tuple[str, float]:
    """Trả (text của trang page_no, thời gian đọc CẢ file tính bằng giây).

    layout=True: pdfplumber cố giữ khoảng cách theo toạ độ gốc (chèn space để mô phỏng
    vị trí chữ trên giấy). Rất khác biệt với PDF nhiều cột / có bảng — thử cả 2 kiểu.
    """
    import pdfplumber

    t0 = time.perf_counter()
    with pdfplumber.open(path) as pdf:
        pages = [(p.extract_text(layout=layout) or "") for p in pdf.pages]
    elapsed = time.perf_counter() - t0
    return pages[page_no - 1] if page_no <= len(pages) else "", elapsed


def in_ket_qua(ten: str, text: str, elapsed: float) -> None:
    n_dong = len(text.splitlines())
    print(f"\n{'=' * 70}")
    print(f"  {ten}   ·   {len(text)} ký tự   ·   {n_dong} dòng   ·   {elapsed:.2f}s (cả file)")
    print("=" * 70)
    print(text[:CHARS] if text.strip() else "  ⚠️  (rỗng — trang ảnh / PDF scan?)")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Dùng: python so_sanh_pdf.py "<file.pdf>" [số_trang]')

    path = sys.argv[1]
    page_no = int(sys.argv[2]) if len(sys.argv) > 2 else 2   # mặc định trang 2, không phải bìa

    print(f"📄 {path}  ·  trang {page_no}  ·  {CHARS} ký tự đầu")

    in_ket_qua("pypdf", *doc_bang_pypdf(path, page_no))
    in_ket_qua("pdfplumber (mặc định)", *doc_bang_pdfplumber(path, page_no))
    in_ket_qua("pdfplumber (layout=True)", *doc_bang_pdfplumber(path, page_no, layout=True))

    # Bonus: pdfplumber có extract_tables() mà pypdf không có.
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        if page_no <= len(pdf.pages):
            tables = pdf.pages[page_no - 1].extract_tables()
            print(f"\n📊 pdfplumber tìm thấy {len(tables)} bảng ở trang {page_no}")
            if tables:
                print("   2 hàng đầu của bảng 1:", tables[0][:2])

    print("\n" + "-" * 70)
    print("Nhìn vào đâu để kết luận (ghi vào python-knowledge/Tuan-05/T2/README.md):")
    print("  1. Chữ có bị dính liền không?      'truyxuất' vs 'truy xuất'")
    print("  2. Thứ tự đọc có đúng không?       PDF 2 cột: cột trái xong mới sang phải,")
    print("                                     hay trộn lẫn từng dòng?")
    print("  3. Bảng ra thế nào?                thành text lộn xộn hay giữ được hàng/cột?")
    print("  4. Số ký tự lệch bao nhiêu?        lệch >10% = một bên đọc hụt nội dung")
    print("  5. Chậm hơn bao nhiêu lần?         nhân với số file thật sẽ ingest")
    print("\nKết luận cần có dạng: 'với <loại file này>, chọn <X> vì <lý do quan sát được>'")
    print("— không phải 'pdfplumber tốt hơn'.")


if __name__ == "__main__":
    main()
