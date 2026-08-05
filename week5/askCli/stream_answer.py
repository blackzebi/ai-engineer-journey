"""
stream_answer.py — Streaming câu trả lời + đo token + ước tính chi phí mỗi câu hỏi

Hai việc, dùng CHUNG một lời gọi API:

    tuần 4:  resp = client.messages.create(...)         # chờ ~6 giây, màn hình đứng im
             print(resp.content[0].text)                # rồi cả đoạn văn hiện ra một lúc

    tuần 5:  with client.messages.stream(...) as stream:
                 for text in stream.text_stream:        # chữ hiện ra sau ~0.4 giây
                     print(text, end="", flush=True)    #    └─ cùng tổng thời gian, khác hẳn
                 final = stream.get_final_message()     #       cảm giác chờ đợi
             final.usage  ──► input_tokens/output_tokens ──► estimate_cost ──► log JSONL

Vì sao streaming đổi hẳn cảm giác dùng:
    Tổng thời gian KHÔNG giảm — thậm chí nhích lên vài chục ms. Thứ giảm là time to first
    token: 6 giây xuống 0.4 giây. Người dùng chịu được chờ lâu nếu thấy có gì đó đang xảy ra;
    cái họ không chịu được là màn hình chết không biết treo hay đang chạy. Streaming là kỹ
    thuật UX, không phải kỹ thuật hiệu năng.

Vì sao đo token và tiền ngay từ bây giờ:
    Nó bắt mình nhìn ra một sự thật: trong RAG, input token (context) thường ĐẮT hơn output,
    dù đơn giá input rẻ hơn. Đo thật ở đây: 835 in / 106 out -> 61% chi phí nằm ở input.
    Kéo theo: tăng k từ 3 lên 10 là nhân ~3 lần tiền mỗi câu hỏi, đổi lại rất ít chất lượng.

Tái dùng nguyên si từ week4/ragLite/prompt_rag.py: SYSTEM_PROMPT, build_rag_prompt.

Self-test bằng client GIẢ — không cần API key, không tốn tiền, không cần mạng:
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
    """Khởi tạo Anthropic client, báo lỗi PHÂN LOẠI ĐƯỢC nếu thiếu key.

    Khác bản tuần 4 (rag_qa.py::get_client dùng `raise SystemExit`): ở đây ném AskError
    để main() xử lý tập trung — xem docstring retriever.retrieve.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise AskError(ErrorKind.MISSING_CONFIG, "chưa có ANTHROPIC_API_KEY")
    import anthropic

    return anthropic.Anthropic(timeout=30)


def stream_and_collect(client, question: str, chunks) -> tuple[str, dict]:
    """Gọi Claude ở chế độ stream. In dần ra màn hình, trả (text đầy đủ, usage).

    usage = {"input_tokens": 1234, "output_tokens": 210}
    Ví dụ: stream_and_collect(client, "RAG là gì?", chunks) -> ("RAG là...", {...})

    Vừa in NGAY vừa gom vào list trong cùng vòng lặp: gom xong mới in là quay lại đúng trải
    nghiệm của messages.create. Nhưng text đầy đủ vẫn cần — để ghi log, để eval, để Streamlit
    dùng lại.

    stream.text_stream là generator chỉ duyệt được MỘT LẦN; duyệt lần hai ra rỗng chứ không
    báo lỗi. get_final_message() phải gọi BÊN TRONG khối `with` và SAU khi duyệt hết
    text_stream, nếu không usage chưa đầy đủ.
    """
    # Không có chunk thì đừng gọi API: tiết kiệm tiền và không cho model cơ hội bịa.
    if not chunks:
        print(NO_ANSWER)
        return NO_ANSWER, {"input_tokens": 0, "output_tokens": 0}

    prompt_chunks = [c.as_prompt_chunk() for c in chunks]
    prompt = build_rag_prompt(question, prompt_chunks)

    parts: list[str] = []
    open_stream = lambda: client.messages.stream(      # noqa: E731 — lambda để hoãn lời gọi
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,        # để ở system=, KHÔNG nhét vào messages
        messages=[{"role": "user", "content": prompt}],
    )
    with retry_with_backoff(open_stream) as stream:
        for text in stream.text_stream:
            # flush=True là BẮT BUỘC: end="" nghĩa là không có \n nào cho tới cuối câu trả lời,
            # nên thiếu flush thì Python giữ chữ trong buffer và xả ra một lúc ở cuối — mất
            # sạch lợi ích streaming, mà code vẫn chạy đúng và không báo gì. Bug im lặng.
            print(text, end="", flush=True)
            # "".join(parts) chứ không text += part: string immutable, mỗi += là copy lại
            # toàn bộ chuỗi -> O(n²).
            parts.append(text)
        final_message = stream.get_final_message()
    print()

    usage = {
        "input_tokens": final_message.usage.input_tokens,
        "output_tokens": final_message.usage.output_tokens,
    }
    return "".join(parts), usage


def estimate_cost(input_tokens: int, output_tokens: int) -> dict:
    """Token -> chi phí ước tính USD, tách riêng phần input và output.

    Trả: {"input_usd": .., "output_usd": .., "total_usd": ..}
    Ví dụ: estimate_cost(2000, 300) -> input 0.002 USD, output 0.0015 USD

    Tách input/output chứ không gộp một số, vì hai phần phản ứng với hai quyết định khác nhau:
        input  <- k (số chunk) và chunk_size  -> tăng k là tăng thẳng phần này
        output <- max_tokens và độ dài câu trả lời
    Gộp lại thì thấy "tốn 0.003 USD" nhưng không biết siết chỗ nào. Tách ra thì thấy ngay
    "61% tiền nằm ở context" -> biết phải chỉnh k, không phải chỉnh max_tokens.

    Đây là ƯỚC TÍNH, không phải hoá đơn: chưa tính prompt caching, batch discount, và giá có
    thể đã đổi.
    """
    # Chia 1_000_000 (giá niêm yết là per MILLION), không phải 1_000. Nhầm là lệch 1000 lần
    # mà con số vẫn in ra đẹp đẽ nên không ai nghi ngờ. Phép thử tự bắt: xem khối main.
    input_usd = input_tokens / 1_000_000 * PRICE_PER_MTOK_INPUT
    output_usd = output_tokens / 1_000_000 * PRICE_PER_MTOK_OUTPUT

    # round 6 chữ số: dưới mức đó là nhiễu, in ra chỉ rối mắt.
    return {
        "input_usd": round(input_usd, 6),
        "output_usd": round(output_usd, 6),
        "total_usd": round(input_usd + output_usd, 6)
    }


def log_query(record: dict, path: str = LOG_PATH) -> None:
    """Ghi 1 dòng JSON vào file log (định dạng JSONL: mỗi dòng là 1 JSON độc lập).

    Ví dụ 1 dòng: {"at": "2026-07-29T11:20:03Z", "question": "...", "k": 3,
                   "n_chunks": 2, "top_similarity": 0.62, "input_tokens": 1840, ...}

    JSONL chứ không phải 1 mảng JSON lớn: mảng thì mỗi lần ghi phải đọc cả file, parse, append,
    ghi đè lại — chậm dần và mất sạch dữ liệu nếu crash giữa lúc ghi đè. JSONL chỉ append thêm
    một dòng. Tuần 7 (golden dataset, LLM-as-judge) sẽ ăn thẳng file này làm đầu vào.

    ⚠️ KHÔNG ghi API key hay toàn văn chunk vào đây. Log metadata, không log nội dung.
    Nhớ giữ `logs/` trong .gitignore.
    """
    # exist_ok=True: thư mục chưa có thì tạo, có rồi thì lần chạy thứ hai không nổ.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # ** bung dict: khoá trùng thì cái SAU thắng -> chỗ gọi vẫn ghi đè "at" được.
    # timezone.utc chứ không datetime.now() trần: giờ máy không kèm múi giờ thì so log giữa
    # hai máy là sai giờ mà không biết.
    record = {"at": datetime.now(timezone.utc).isoformat(), **record}
    # mode "a" (append), không phải "w" — "w" xoá sạch log cũ mỗi lần chạy.
    # ensure_ascii=False + encoding="utf-8": thiếu thì log tiếng Việt thành ạ... không
    # đọc nổi bằng mắt, và Windows mặc định cp1252 sẽ nổ UnicodeEncodeError.
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_cost_line(usage: dict, cost: dict, elapsed_seconds: float, n_chunks: int) -> None:
    """In dòng tổng kết dưới câu trả lời.

    Ví dụ: ⏱ 3.2s · 2 chunk · 1840 in / 210 out token · ~0.003050 USD
    """
    print(
        f"\n⏱  {elapsed_seconds:.1f}s · {n_chunks} chunk · "
        f"{usage['input_tokens']} in / {usage['output_tokens']} out token · "
        f"~{cost['total_usd']:.6f} USD "
        f"(in {cost['input_usd']:.6f} + out {cost['output_usd']:.6f})"
    )


if __name__ == "__main__":
    # Client GIẢ: mô phỏng đúng hình dạng API thật, không tốn tiền, không cần mạng.
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
