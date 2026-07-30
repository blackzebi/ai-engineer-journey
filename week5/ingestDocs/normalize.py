"""
normalize.py — Làm sạch text trích từ PDF

Vì sao PDF cần làm sạch mà .md thì không:
    .md là text người viết cho người đọc. PDF là **mô tả cách VẼ chữ lên giấy** —
    extract_text() ghép các mảnh vẽ đó lại theo toạ độ, nên nhận về 4 loại rác:

        Hệ thống RAG bao gồm hai giai đoạn chính là truy      <- (1) ngắt dòng giữa câu
        xuất và sinh câu trả lời.
        Báo cáo kỹ thuật 2026                                 <- (2) header lặp ở MỌI trang
        12                                                    <- (3) số trang trần trụi
        micro-  service                                       <- (4) gạch nối cuối dòng

Ảnh hưởng tới chất lượng RAG:
    - Câu bị ngắt       -> embedding loãng nghĩa -> similarity tụt -> retrieval trượt
    - Header lặp 40 lần -> 40 chunk chứa cùng 1 chuỗi rác -> hỏi trúng chữ trong header là
                           top-k trả về toàn trang bìa. Rác LẶP LẠI nguy hiểm hơn rác ngẫu nhiên
    - Số trang trần     -> chunk vô nghĩa nhưng vẫn chiếm 1 slot trong top-k

Bỏ qua module này thì pipeline vẫn "chạy" — chỉ có chất lượng retrieval âm thầm tệ đi.

Chạy self-test bằng dữ liệu giả (không cần PDF, không cần DB):
    python normalize.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Ngưỡng coi 1 dòng là header/footer: xuất hiện ở >= 60% số trang.
# Không dùng 100%: trang bìa / mục lục thường không có header -> sẽ bắt trượt.
# Không dùng 30%: sẽ ăn nhầm dòng nội dung lặp lại hợp lệ (vd tên chương).
HEADER_FOOTER_RATIO = 0.6

# Chỉ dòng NGẮN mới là ứng viên header/footer. Câu văn dài lặp lại là chuyện khác.
MAX_HEADER_LEN = 80


def find_repeated_lines(pages: list[str], ratio: float = HEADER_FOOTER_RATIO) -> set[str]:
    """Tìm các dòng xuất hiện ở >= `ratio` số trang -> gần như chắc chắn là header/footer.

    Suy luận THỐNG KÊ trên chính tài liệu, không hard-code blacklist — vì header mỗi file
    mỗi khác, không thể biết trước. (Cùng ý tưởng với IDF trong BM25: thứ xuất hiện ở mọi
    nơi thì không mang thông tin phân biệt.)
    """
    # Quá ít trang thì mọi dòng đều "lặp ở 100% số trang" -> không guard là xoá sạch tài liệu.
    if len(pages) < 4:
        return set()

    counter = Counter()
    for page in pages:
        # set() cho từng trang -> mỗi trang chỉ đóng góp 1 phiếu. Không có set() thì 1 trang
        # lặp dòng 3 lần có thể tự đẩy mình qua ngưỡng.
        lines_in_page = {ln.strip() for ln in page.splitlines() if ln.strip()}
        counter.update(lines_in_page)

    nguong = len(pages) * ratio
    return {
        line for line, count in counter.items()
        if count >= nguong and len(line) <= MAX_HEADER_LEN
    }


def is_page_number(line: str) -> bool:
    """True nếu dòng CHỈ là số trang: '12', '- 12 -', 'Trang 12', 'Page 12 of 40'.

    Số trang thoát khỏi find_repeated_lines() vì mỗi trang một số khác nhau -> không lặp.
    Phải bắt bằng HÌNH DẠNG, không bắt bằng tần suất. Hai cơ chế bổ sung nhau.

    Ràng buộc "1–4 chữ số" + "phải là TOÀN BỘ dòng" để không xoá nhầm bảng biểu /
    danh sách đánh số.
    """
    line = line.strip()
    if not line:
        return False

    patterns = (
        r"^\d{1,4}$",                                        # 12
        r"^[-–—\s]*\d{1,4}[-–—\s]*$",                        # - 12 -   |   — 12 —
        r"^(trang|page)\s+\d{1,4}(\s*(/|of)\s*\d{1,4})?$",   # Trang 12 | Page 12 of 40
    )

    return any(re.match(p, line, re.IGNORECASE) for p in patterns)


def join_broken_lines(text: str) -> str:
    """Nối các dòng bị PDF ngắt giữa câu, GIỮ NGUYÊN ngắt đoạn thật.

    Vào:  "Hệ thống RAG gồm hai giai\\nđoạn chính.\\n\\nGiai đoạn một là truy xuất."
    Ra :  "Hệ thống RAG gồm hai giai đoạn chính.\\n\\nGiai đoạn một là truy xuất."

    Trong text thô có 2 loại xuống dòng trông y hệt nhau: ngắt dòng KỸ THUẬT (chữ hết chiều
    rộng trang, phải nối) và ngắt đoạn NGỮ NGHĨA (hết ý, phải giữ). Không có cách nào chắc
    chắn 100% -> dùng heuristic 3 điều kiện, xem nen_noi().

    Giới hạn đã biết: tiếng Việt nhiều dòng bắt đầu bằng chữ hoa giữa câu (tên riêng) nên
    điều kiện 2 sẽ bỏ sót vài chỗ. Mục tiêu là giảm ~80% nhiễu, không phải 100%.
    """
    # Gạch nối cuối dòng phải xử lý TRƯỚC vòng lặp: 'micro-\nservice' -> 'microservice'
    # (dính liền). Nếu vòng lặp nối bằng " " trước thì thành 'micro- service' — mất dấu vết.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    lines = text.split("\n")
    out: list[str] = []

    def nen_noi(a: str, b: str) -> bool:
        """Có nên nối dòng a với dòng b không — cả 3 điều kiện phải đúng."""
        a, b = a.rstrip(), b.lstrip()
        if not a or not b:
            return False                          # dòng trống = ngắt đoạn, giữ nguyên
        if a[-1] in '.!?:;)]"”':
            return False                          # 1. a đã kết thúc câu
        if not (b[0].islower() or b[0].isdigit()):
            return False                          # 2. b không bắt đầu bằng chữ thường/số
        if re.match(r"^([-*•]|\d+\.|#)", b):
            return False                          # 3. b là bullet hoặc heading
        return True

    # Mẹo: xây `out` rồi mỗi lần nối thì SỬA phần tử cuối. Cách này tự động xử lý được
    # chuỗi 3–4 dòng liên tiếp bị ngắt, vì out[-1] luôn là phiên bản đã nối mới nhất.
    for line in lines:
        if out and nen_noi(out[-1], line):
            # Nối bằng " " chứ không nối trần: "truy"+"xuất" = "truyxuất" -> embedding sai.
            out[-1] = out[-1].rstrip() + " " + line.lstrip()
        else:
            out.append(line)

    return "\n".join(out)


def collapse_whitespace(text: str) -> str:
    """Gom khoảng trắng thừa, bỏ dòng trống liên tiếp, strip từng dòng. GIỮ ngắt đoạn.

    ⚠️ KHÔNG dùng " ".join(text.split()) — nó xoá sạch mọi newline kể cả ngắt đoạn thật.
    Phải xử lý RIÊNG khoảng trắng ngang ([ \\t]) và khoảng trắng dọc (newline).
    """
    text = re.sub(r"[ \t]+", " ", text)                        # ngang; \s+ sẽ ăn cả \n
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)                     # giữ 1 dòng trống = ngắt đoạn
    return text.strip()


def normalize_pdf_pages(pages: list[str]) -> list[str]:
    """Nhận text thô của TỪNG TRANG, trả text đã làm sạch của từng trang.

    Nhận cả LIST trang chứ không normalize từng trang rời: phát hiện header/footer cần nhìn
    TOÀN BỘ tài liệu mới biết dòng nào lặp.

    THỨ TỰ BẮT BUỘC (đảo là hỏng):
        1. find_repeated_lines(pages)   — cần text THÔ, còn nguyên từng dòng
        2. lọc header/footer + số trang — theo từng dòng
        3. join_broken_lines            — sau khi rác đã biến mất; làm trước sẽ nối nhầm
                                          header dính vào câu văn
        4. collapse_whitespace          — dọn cuối cùng
    """
    repeated = find_repeated_lines(pages)
    out = []
    for raw in pages:
        kept = [
            ln for ln in raw.splitlines()
            if ln.strip() not in repeated and not is_page_number(ln.strip())
        ]
        out.append(collapse_whitespace(join_broken_lines("\n".join(kept))))
    return out


if __name__ == "__main__":
    # Self-test bằng dữ liệu giả mô phỏng đúng 4 loại rác — không cần PDF, sửa cực nhanh.
    fake_pages = [
        "Báo cáo kỹ thuật 2026\nHệ thống RAG bao gồm hai giai đoạn chính là truy\nxuất và sinh câu trả lời.\n1",
        "Báo cáo kỹ thuật 2026\nGiai đoạn truy xuất dùng micro-\nservice riêng để tính embedding.\n\nGiai đoạn sinh dùng LLM.\n2",
        "Báo cáo kỹ thuật 2026\nĐánh giá chất lượng cần một bộ dữ liệu vàng.\nTrang 3",
        "Báo cáo kỹ thuật 2026\nKết luận: pipeline chạy ổn định.\n4",
    ]
    for i, page in enumerate(normalize_pdf_pages(fake_pages), 1):
        print(f"--- trang {i} ---\n{page}\n")

    # Kỳ vọng:
    #   1. Không còn dòng "Báo cáo kỹ thuật 2026" ở bất kỳ trang nào
    #   2. Không còn "1" / "2" / "Trang 3" / "4" đứng riêng một dòng
    #   3. "truy\nxuất" -> "truy xuất" (có dấu cách)
    #      "micro-\nservice" -> "microservice" (không có dấu cách)
    #   4. Dòng trống giữa 2 đoạn ở trang 2 vẫn còn
