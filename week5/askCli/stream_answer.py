"""
stream_answer.py — Streaming câu trả lời + đo token + ước tính chi phí mỗi câu hỏi

Hai việc của file này, tưởng rời nhau nhưng dùng CHUNG một lời gọi API:

    tuần 4:  resp = client.messages.create(...)         # chờ ~6 giây, màn hình đứng im
             print(resp.content[0].text)                # rồi cả đoạn văn hiện ra một lúc

    tuần 5:  with client.messages.stream(...) as stream:
                 for text in stream.text_stream:        # chữ hiện ra sau ~0.4 giây
                     print(text, end="", flush=True)    #    └─ cùng tổng thời gian, khác hẳn
                 final = stream.get_final_message()     #       cảm giác chờ đợi
             final.usage  ──► input_tokens/output_tokens ──► estimate_cost ──► log JSONL

Vì sao streaming đổi hẳn cảm giác dùng (câu hỏi CHÍNH của block sáng, hiểu rồi hãy code):
    Tổng thời gian KHÔNG giảm — thậm chí nhích lên vài chục ms. Thứ giảm là **time to first
    token**: 6 giây xuống 0.4 giây. Người dùng chịu được chờ lâu nếu thấy có gì đó đang xảy ra;
    cái họ không chịu được là màn hình chết không biết treo hay đang chạy. Đây là bài học UX
    dùng được ở mọi chỗ, không riêng LLM.

Vì sao phải đo token và tiền ngay từ bây giờ:
    "Mỗi câu hỏi tốn khoảng 0.002 USD, chủ yếu là input vì context 3 chunk" — câu này nói
    được trong phỏng vấn tách hẳn người đã chạy thật với người mới đọc tutorial. Và nó bắt
    mình nhìn ra một sự thật: trong RAG, input token (context) thường ĐẮT hơn output.
    Tăng k từ 3 lên 10 là nhân ~3 lần tiền mỗi câu hỏi, đổi lại rất ít chất lượng.

Tái dùng nguyên si từ week4/ragLite/prompt_rag.py: SYSTEM_PROMPT, build_rag_prompt.

📖 Mỗi hàm chia 2 khối: PHẦN 1 GIẢI THÍCH (đọc) · PHẦN 2 CODE CẦN VIẾT (gõ).

Self-test bằng client GIẢ — KHÔNG cần API key, KHÔNG tốn tiền, không cần mạng:
    python stream_answer.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragLite"))

from errors import AskError, ErrorKind, retry_with_backoff  # noqa: E402
from prompt_rag import SYSTEM_PROMPT, build_rag_prompt  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Trùng model đã dùng ở tuần 4 để so sánh chi phí giữa hai bản có ý nghĩa.
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512

# USD cho 1 TRIỆU token. ⚠️ Giá thay đổi theo thời gian — mở trang pricing của Anthropic
# đối chiếu lại rồi sửa 2 số này. Con số sai còn tệ hơn không đo, vì mình sẽ tin nó.
PRICE_PER_MTOK_INPUT = 1.00
PRICE_PER_MTOK_OUTPUT = 5.00

LOG_PATH = os.path.join(HERE, "logs", "queries.jsonl")

NO_ANSWER = "Tôi không tìm thấy thông tin này trong tài liệu."


def get_client():
    """Khởi tạo Anthropic client, báo lỗi PHÂN LOẠI ĐƯỢC nếu thiếu key. VIẾT SẴN.

    Khác bản tuần 4 (rag_qa.py::get_client dùng `raise SystemExit`): ở đây ném AskError
    để main() xử lý tập trung — xem docstring retriever.retrieve.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise AskError(ErrorKind.MISSING_CONFIG, "chưa có ANTHROPIC_API_KEY")
    import anthropic

    return anthropic.Anthropic(timeout=30)


# ---------------------------------------------------------------------------
# TODO 1 — stream ra màn hình, đồng thời giữ lại text đầy đủ + usage
# ---------------------------------------------------------------------------
def stream_and_collect(client, question: str, chunks) -> tuple[str, dict]:
    """Gọi Claude ở chế độ stream. In dần ra màn hình, trả (text đầy đủ, usage).

    usage = {"input_tokens": 1234, "output_tokens": 210}
    Ví dụ: stream_and_collect(client, "RAG là gì?", chunks) -> ("RAG là...", {...})
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Vì sao vừa in NGAY vừa gom vào list, chứ không gom xong rồi in một lần?
    #   Gom xong mới in = quay lại đúng trải nghiệm của messages.create, streaming thành vô
    #   nghĩa. Nhưng bản thân text đầy đủ vẫn cần: để ghi log, để eval, để Streamlit dùng lại.
    #   -> làm cả hai trong cùng vòng lặp.
    #
    # ▸ Bẫy CHẾT NGƯỜI 1 — `print(text, end="")` KHÔNG có `flush=True` thì Python giữ chữ
    #   trong buffer và xả ra theo từng dòng. Kết quả: chữ hiện ra giật cục từng đoạn, hoặc
    #   im lìm rồi hiện hết một lúc — nghĩa là mất sạch lợi ích của streaming, mà code thì
    #   nhìn vẫn "đúng". Bug im lặng kinh điển của streaming trên terminal.
    #
    # ▸ Bẫy 2 — `"".join(parts)` chứ không `text += part` trong vòng lặp. String trong Python
    #   là immutable: mỗi lần += là copy lại toàn bộ chuỗi -> O(n²). Vài trăm mảnh thì chưa
    #   thấy gì, nhưng đây là phản xạ nên có sẵn.
    #
    # ▸ Bẫy 3 — `get_final_message()` phải gọi BÊN TRONG khối `with`. Ra khỏi `with` là
    #   stream đã đóng -> lỗi. Và phải duyệt HẾT text_stream trước khi gọi nó, nếu không
    #   usage chưa đầy đủ.
    #
    # ▸ Bẫy 4 — retry cả hàm stream là sai: lỗi ở giữa chừng thì phần chữ đã in ra màn hình
    #   không rút lại được, retry sẽ in tiếp lần hai và người dùng thấy câu trả lời trùng lặp.
    #   Đúng: chỉ retry ở tầng ngoài KHI CHƯA in gì (xem chỗ dùng retry_with_backoff bên dưới),
    #   hoặc chấp nhận không retry cho stream. Hôm nay chọn: retry quanh chỗ MỞ stream.
    #
    # ▸ Nhắc cú pháp: `with client.messages.stream(...) as stream:` — context manager, tự
    #   đóng kết nối. `stream.text_stream` là generator chỉ duyệt được MỘT LẦN; duyệt lần
    #   hai ra rỗng chứ không báo lỗi.

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — không có chunk thì đừng gọi API (tiết kiệm tiền + không cho model cơ hội bịa):
    #     if not chunks:
    #         print(NO_ANSWER)
    #         return NO_ANSWER, {"input_tokens": 0, "output_tokens": 0}
    #
    # Bước 2 — dựng prompt bằng đồ tuần 4, chunk đổi hình qua as_prompt_chunk():
    #     prompt_chunks = [c.as_prompt_chunk() for c in chunks]
    #     prompt = build_rag_prompt(question, prompt_chunks)
    #
    # Bước 3 — mở stream (bọc retry quanh CHỖ MỞ, xem Bẫy 4):
    #     parts: list[str] = []
    #     open_stream = lambda: client.messages.stream(      # lambda: hoãn lời gọi lại
    #         model=MODEL,
    #         max_tokens=MAX_TOKENS,
    #         system=SYSTEM_PROMPT,        # để ở system=, KHÔNG nhét vào messages
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     with retry_with_backoff(open_stream) as stream:
    #
    # Bước 4 — (trong with) vừa in vừa gom, nhớ flush:
    #         for text in stream.text_stream:
    #             print(text, end="", flush=True)
    #             parts.append(text)
    #         final_message = stream.get_final_message()     # vẫn TRONG with
    #     print()                                            # xuống dòng sau khi stream xong
    #
    # Bước 5 — lấy usage và trả về:
    #     usage = {
    #         "input_tokens": final_message.usage.input_tokens,
    #         "output_tokens": final_message.usage.output_tokens,
    #     }
    #     return "".join(parts), usage
    #
    # ✅ Kiểm tra nhanh: chạy thật một câu hỏi, chữ phải hiện ra TỪNG MẢNH chứ không phải
    #    cả đoạn một lúc. Nếu hiện một lúc -> gần như chắc chắn quên flush=True.
    if not chunks:
        print(NO_ANSWER)
        return NO_ANSWER, {"input_tokens": 0, "output_tokens": 0}

    prompt_chunks = [c.as_prompt_chunk() for c in chunks]
    prompt = build_rag_prompt(question, prompt_chunks)

    parts: list[str] = []
    open_stream = lambda: client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    with retry_with_backoff(open_stream) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            parts.append(text)
        final_message = stream.get_final_message()
    print()

    usage = {
        "input_tokens": final_message.usage.input_tokens,
        "output_tokens": final_message.usage.output_tokens,
    }
    return "".join(parts), usage



# ---------------------------------------------------------------------------
# TODO 2 — quy token ra tiền
# ---------------------------------------------------------------------------
def estimate_cost(input_tokens: int, output_tokens: int) -> dict:
    """Token -> chi phí ước tính USD, tách riêng phần input và output.

    Trả: {"input_usd": .., "output_usd": .., "total_usd": ..}
    Ví dụ: estimate_cost(2000, 300) -> input 0.002 USD, output 0.0015 USD
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Vì sao tách input/output chứ không gộp một con số?
    #   Vì hai phần này phản ứng với hai quyết định thiết kế khác nhau:
    #     input  <- k (số chunk) và chunk_size  -> tăng k là tăng thẳng phần này
    #     output <- max_tokens và độ dài câu trả lời
    #   Gộp lại thì thấy "tốn 0.003 USD" nhưng không biết siết chỗ nào. Tách ra thì thấy
    #   ngay "80% tiền nằm ở context" -> biết ngay nên chỉnh k, không phải chỉnh max_tokens.
    #
    # ▸ Bẫy — chia cho 1000 thay vì 1_000_000. Giá niêm yết là "per million tokens"; nhầm
    #   một dấu phẩy là con số lệch 1000 lần, mà nó vẫn in ra đẹp đẽ nên không ai nghi ngờ.
    #   Cách tự bắt: 2000 token input với giá 1 USD/triệu PHẢI ra 0.002 USD. Nhẩm được.
    #
    # ▸ Đây là ƯỚC TÍNH, không phải hoá đơn: chưa tính prompt caching, chưa tính batch
    #   discount, giá có thể đã đổi. Nói "ước tính" khi trình bày là chính xác và đủ dùng.
    #
    # ▸ Nhắc cú pháp: `1_000_000` — dấu gạch dưới trong số là cú pháp hợp lệ của Python,
    #   chỉ để người đọc đỡ đếm số 0. Dùng đi, nó chặn đúng cái bẫy ở trên.

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — quy đổi từng phần:
    #     input_usd = input_tokens / 1_000_000 * PRICE_PER_MTOK_INPUT
    #     output_usd = output_tokens / 1_000_000 * PRICE_PER_MTOK_OUTPUT
    #
    # Bước 2 — trả dict (round 6 chữ số: dưới mức đó là nhiễu, in ra chỉ rối mắt):
    #     return {
    #         "input_usd": round(input_usd, 6),
    #         "output_usd": round(output_usd, 6),
    #         "total_usd": round(input_usd + output_usd, 6),
    #     }
    #
    # ✅ Kiểm tra nhanh: estimate_cost(1_000_000, 0)["input_usd"] phải bằng đúng
    #    PRICE_PER_MTOK_INPUT. Phép thử này bắt được ngay lỗi chia nhầm 1000.
    input_usd = input_tokens / 1_000_000 * PRICE_PER_MTOK_INPUT
    output_usd = output_tokens / 1_000_000 * PRICE_PER_MTOK_OUTPUT

    return {
        "input_usd": round(input_usd, 6),
        "output_usd": round(output_usd, 6),
        "total_usd": round(input_usd + output_usd, 6)
    }


# ---------------------------------------------------------------------------
# TODO 3 — ghi lại mỗi câu hỏi để sau này còn đo được
# ---------------------------------------------------------------------------
def log_query(record: dict, path: str = LOG_PATH) -> None:
    """Ghi 1 dòng JSON vào file log (định dạng JSONL: mỗi dòng là 1 JSON độc lập).

    Ví dụ 1 dòng: {"at": "2026-07-29T11:20:03Z", "question": "...", "k": 3,
                   "n_chunks": 2, "top_similarity": 0.62, "input_tokens": 1840, ...}
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Vì sao JSONL (mỗi dòng 1 JSON) chứ không phải 1 mảng JSON lớn?
    #   File JSON mảng thì mỗi lần ghi phải đọc cả file, parse, append, ghi đè lại — chậm dần
    #   và mất sạch dữ liệu nếu crash giữa lúc ghi đè. JSONL chỉ append thêm một dòng, đọc thì
    #   duyệt từng dòng. Đây là định dạng chuẩn của log/dataset, và tuần 7 (golden dataset,
    #   LLM-as-judge) sẽ ăn thẳng file này làm đầu vào — hôm nay ghi là để tuần 7 có cái mà đo.
    #
    # ▸ Bẫy 1 — `json.dumps` mặc định escape tiếng Việt thành ạ... File log mở ra không
    #   đọc nổi bằng mắt. Phải `ensure_ascii=False`. Và `encoding="utf-8"` khi mở file, vì
    #   Windows mặc định cp1252 -> UnicodeEncodeError ngay dòng tiếng Việt đầu tiên.
    #
    # ▸ Bẫy 2 — mở file mode "w" là xoá sạch log cũ mỗi lần chạy. Phải là "a" (append).
    #
    # ▸ Bẫy 3 — thư mục logs/ chưa tồn tại -> FileNotFoundError ngay lần chạy đầu của người
    #   clone repo về. `os.makedirs(..., exist_ok=True)` giải quyết, và exist_ok=True để chạy
    #   lần thứ hai không nổ.
    #
    # ▸ Bẫy 4 — ĐỪNG ghi API key hay toàn văn chunk vào log. Log rồi vô tình commit là đúng
    #   cái lỗi mà task "config sạch" hôm nay muốn tránh. Log metadata, không log nội dung.
    #   Nhớ thêm `logs/` vào .gitignore (xem guide.md).
    #
    # ▸ Nhắc cú pháp: `datetime.now(timezone.utc).isoformat()` — có timezone. `datetime.now()`
    #   trần cho giờ máy không kèm múi giờ, so sánh log giữa 2 máy là sai giờ mà không biết.

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — đảm bảo thư mục tồn tại:
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #
    # Bước 2 — thêm mốc thời gian nếu chỗ gọi chưa đưa:
    #     record = {"at": datetime.now(timezone.utc).isoformat(), **record}
    #     # ** bung dict: khoá trùng thì cái SAU thắng -> chỗ gọi vẫn ghi đè "at" được
    #
    # Bước 3 — append 1 dòng:
    #     with open(path, "a", encoding="utf-8") as f:
    #         f.write(json.dumps(record, ensure_ascii=False) + "\n")
    #
    # ✅ Kiểm tra nhanh: hỏi 3 câu rồi mở logs/queries.jsonl — phải có đúng 3 dòng, tiếng Việt
    #    đọc được bằng mắt, và KHÔNG dòng nào chứa chuỗi bắt đầu bằng "sk-ant".
    os.makedirs(os.path.dirname(path), exist_ok=True)

    record = {"at": datetime.now(timezone.utc).isoformat(), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_cost_line(usage: dict, cost: dict, elapsed_seconds: float, n_chunks: int) -> None:
    """In dòng tổng kết dưới câu trả lời. VIẾT SẴN — chỉ cần đọc.

    Ví dụ: ⏱ 3.2s · 2 chunk · 1840 in / 210 out token · ~0.003050 USD
    """
    print(
        f"\n⏱  {elapsed_seconds:.1f}s · {n_chunks} chunk · "
        f"{usage['input_tokens']} in / {usage['output_tokens']} out token · "
        f"~{cost['total_usd']:.6f} USD "
        f"(in {cost['input_usd']:.6f} + out {cost['output_usd']:.6f})"
    )


if __name__ == "__main__":
    # ── Client GIẢ: mô phỏng đúng hình dạng API thật, không tốn tiền, không cần mạng ────
    class FakeUsage:
        input_tokens = 1840
        output_tokens = 210

    class FakeFinalMessage:
        usage = FakeUsage()

    class FakeStream:
        """Bắt chước đủ 3 thứ mà stream_and_collect dùng: with, text_stream, get_final_message."""

        text_stream = iter(
            ["RAG ", "gồm hai ", "giai đoạn: ", "truy xuất ", "và sinh ", "câu trả lời [1]."]
        )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_final_message(self):
            return FakeFinalMessage()

    class FakeMessages:
        def stream(self, **kwargs):
            print(f"   (fake API: model={kwargs.get('model')}, "
                  f"prompt {len(kwargs['messages'][0]['content'])} ký tự)")
            return FakeStream()

    class FakeClient:
        messages = FakeMessages()

    class FakeChunk:
        """Đủ để build_rag_prompt chạy — chỉ cần as_prompt_chunk()."""

        def __init__(self, id_, source, content, similarity):
            self.id = id_
            self.source = source
            self.content = content
            self.similarity = similarity

        def as_prompt_chunk(self):
            return (self.id, self.source, self.content, self.similarity)

    fake_chunks = [
        FakeChunk(1, "Báo cáo kỹ thuật 2026 · trang 12", "RAG gồm truy xuất và sinh.", 0.62),
        FakeChunk(2, "Ghi chú RAG · Chunking", "Chunk 800 ký tự, overlap 100.", 0.41),
    ]

    print("===== 1. stream_and_collect (client giả) =====")
    started_at = time.perf_counter()
    text, usage = stream_and_collect(FakeClient(), "RAG là gì?", fake_chunks)
    elapsed = time.perf_counter() - started_at
    print(f"   text gom được: {text!r}")
    print(f"   usage: {usage}")

    print("\n===== 2. estimate_cost =====")
    cost = estimate_cost(usage["input_tokens"], usage["output_tokens"])
    print(f"   {cost}")
    print(f"   phép thử bắt lỗi chia nhầm: estimate_cost(1_000_000, 0)['input_usd'] = "
          f"{estimate_cost(1_000_000, 0)['input_usd']}  (phải = {PRICE_PER_MTOK_INPUT})")
    print_cost_line(usage, cost, elapsed, len(fake_chunks))

    print("\n===== 3. stream_and_collect với 0 chunk (không được gọi API) =====")
    text_empty, usage_empty = stream_and_collect(FakeClient(), "câu hỏi ngoài lề", [])
    print(f"   usage: {usage_empty}")

    print("\n===== 4. log_query =====")
    log_query({
        "question": "RAG là gì?",
        "k": 3,
        "n_chunks": len(fake_chunks),
        "top_similarity": 0.62,
        **usage,
        **cost,
        "elapsed_seconds": round(elapsed, 2),
    })
    print(f"   đã ghi -> {LOG_PATH}")
    with open(LOG_PATH, encoding="utf-8") as f:
        print("   dòng cuối:", f.readlines()[-1].strip())

    # ✅ ĐẠT khi:
    #   1. Chữ ở phần 1 hiện ra TỪNG MẢNH (không phải cả câu một lúc) — nếu một lúc: thiếu flush
    #   2. text gom được ghép đủ 6 mảnh thành 1 câu hoàn chỉnh
    #   3. estimate_cost(1_000_000, 0)['input_usd'] == PRICE_PER_MTOK_INPUT (chặn lỗi /1000)
    #   4. 0 chunk -> in đúng câu NO_ANSWER, usage cả hai đều = 0, KHÔNG có dòng "(fake API...)"
    #   5. logs/queries.jsonl đọc được tiếng Việt bằng mắt; chạy lại 2 lần -> file có 2 dòng
