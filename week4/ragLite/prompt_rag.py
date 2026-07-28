"""
prompt_rag.py — Ghép retrieval vào prompt (T5 tuần 4, block AI CORE 09:00–11:00)

Đây là mảnh "G" (Generate) của RAG. Sáng nay pipeline retrieval (T4) đã cho ra top-k chunk.
Việc còn lại: **nhét chúng vào 1 prompt sao cho model trả lời DỰA TRÊN chunk, không bịa.**

Hai kỹ thuật cốt lõi của buổi hôm nay:
    1) Khuôn prompt "grounding": 'Dựa vào các đoạn sau… nếu không có thông tin thì nói KHÔNG BIẾT'
    2) Trích nguồn: mỗi câu trả lời kèm source + vị trí chunk để người đọc kiểm chứng

Vì sao khuôn (1) giảm hallucination (câu hỏi CHÍNH của buổi, hiểu rồi hãy code):
    - LLM mặc định là máy "đoán từ tiếp theo hợp lý nhất" → khi thiếu dữ kiện nó vẫn tự tin bịa.
    - Khuôn này làm 2 việc: (a) ĐÓNG KHUNG nguồn tri thức hợp lệ = chỉ các đoạn được cung cấp,
      (b) MỞ ĐƯỜNG THOÁT hợp lệ ("nói KHÔNG BIẾT") để model không bị ép phải bịa cho có.
    - Không có (b), model coi "phải trả lời" là ràng buộc mạnh hơn "phải đúng" → bịa.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Kiểu 1 chunk lấy từ pgvector: (id, source, content, similarity)
Chunk = tuple[int, str, str, float]


# System prompt: đặt "luật chơi" cho model. Tách riêng khỏi context để dễ chỉnh giọng.
SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi-đáp chỉ được trả lời DỰA TRÊN các đoạn trích được cung cấp. "
    "Nếu các đoạn trích không chứa thông tin để trả lời, hãy nói đúng câu: "
    "'Tôi không tìm thấy thông tin này trong tài liệu.' — tuyệt đối không suy đoán hay bịa. "
    "Khi trả lời, trích dẫn nguồn bằng số [1], [2]... khớp với danh sách đoạn."
)


def format_context(chunks: list[Chunk]) -> str:
    """Biến top-k chunk thành khối context đánh số [1], [2]... để model trích dẫn được.

    Định dạng mong muốn cho MỖI chunk:
        [1] (nguồn: Tuan-01_.../Async/README.md · similarity=0.83)
        <nội dung chunk>

    Vì sao đánh số + kèm nguồn NGAY trong context: để model chỉ việc "chép lại" [n] khi
    trích dẫn — dễ hơn nhiều so với bắt nó tự nhớ tên file dài. Đây là nền của trích nguồn.
    """
    blocks = []
    for i, (_id, source, content, sim) in enumerate(chunks, 1):
        blocks.append(f"[{i}] (nguồn: {source} · similarity={sim:.2f})\n{content.strip()}")
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, chunks: list[Chunk]) -> str:
    """Ghép context + câu hỏi thành user prompt cuối cùng gửi cho Claude.

    Khung khuyến nghị (giữ đúng thứ tự: context TRƯỚC, câu hỏi SAU):
        Dựa vào các đoạn trích dưới đây để trả lời câu hỏi.
        Nếu không có thông tin, hãy nói 'Tôi không tìm thấy thông tin này trong tài liệu.'

        === CÁC ĐOẠN TRÍCH ===
        {context}

        === CÂU HỎI ===
        {question}

        Trả lời ngắn gọn bằng tiếng Việt, kèm trích dẫn [số] cho mỗi ý.

    Bẫy: nếu chunks rỗng (retrieval không ra gì) thì context = "" → model vẫn nên trả
    'không tìm thấy'. Xử lý case rỗng ở rag_qa.py chứ không cần ở đây.
    """
    context = format_context(chunks)
    return (
        "Dựa vào các đoạn trích dưới đây để trả lời câu hỏi.\n"
        # Câu này phải khớp TỪNG KÝ TỰ (kể cả viết hoa) với NO_ANSWER trong rag_qa.py,
        # vì eval chấm bằng `NO_ANSWER in answer` — phân biệt hoa/thường.
        "Nếu không có thông tin, hãy nói 'Tôi không tìm thấy thông tin này trong tài liệu.'\n\n"
        f"=== CÁC ĐOẠN TRÍCH ===\n{context}\n\n"
        f"=== CÂU HỎI ===\n{question}\n\n"
        "Trả lời ngắn gọn bằng tiếng Việt, kèm trích dẫn [số] cho mỗi ý."
    )


def format_citations(chunks: list[Chunk]) -> str:
    """In danh sách nguồn để hiện DƯỚI câu trả lời (người đọc kiểm chứng được).

    Mong muốn:
        Nguồn tham khảo:
          [1] Tuan-01_.../Async/README.md            (similarity 0.83)
          [2] Tuan-03_.../T5/Embedding/README.md     (similarity 0.79)
    """
    lines = ["Nguồn tham khảo:"]
    for i, (_id, source, _content, sim) in enumerate(chunks, 1):
        lines.append(f" [{i}] {source} (similarity {sim:.2f})")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test nhanh KHÔNG cần DB/model — dùng chunk giả để xem prompt ráp ra có đúng hình không.
    fake_chunks: list[Chunk] = [
        (1, "Tuan-01_.../Async/README.md", "asyncio dùng event loop để chạy nhiều I/O song song...", 0.83),
        (2, "Tuan-03_.../T5/Embedding/README.md", "embedding là vector số biểu diễn nghĩa của text...", 0.79),
    ]
    print("===== build_rag_prompt =====")
    print(build_rag_prompt("Embedding là gì?", fake_chunks))
    print("\n===== format_citations =====")
    print(format_citations(fake_chunks))
