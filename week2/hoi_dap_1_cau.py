"""Hỏi - đáp 1 câu với Anthropic Messages API (Week 2 - sổ tay P2).

Bài tập gộp 3 việc:
  - Task 0: khởi tạo Anthropic() + gọi client.messages.create
            (model, max_tokens BẮT BUỘC, system, messages).
  - Task 1: phân biệt 3 khái niệm hay nhầm — system / messages / content;
            lấy câu trả lời qua resp.content[0].text.
  - Task 2: hỏi - đáp đúng 1 câu, in kết quả ra terminal (chạy xanh = OK).

Cách chạy (từ thư mục gốc repo):
    .venv\\Scripts\\python.exe week2\\hoi_dap_1_cau.py     # Windows
    .venv/bin/python week2/hoi_dap_1_cau.py                # macOS/Linux
"""

import os
import sys

from dotenv import load_dotenv

# Terminal Windows mặc định là cp1252, không in được tiếng Việt -> ép UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Nạp .env -> biến môi trường. Key KHÔNG viết thẳng trong code (xem .gitignore).
load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")

from anthropic import Anthropic

# --- Task 0: khởi tạo client -------------------------------------------------
client = Anthropic(api_key=api_key)

MODEL = "claude-sonnet-5"

# system = "luật chơi" / tính cách của model. 1 chuỗi, đặt 1 lần, KHÔNG nằm trong hội thoại.
SYSTEM_PROMPT = "Bạn là trợ lý AI thân thiện, trả lời ngắn gọn và chính xác bằng tiếng Việt."

# messages = "ván cờ" đang diễn ra. Là 1 LIST các lượt, mỗi lượt có role + content.
# Hỏi - đáp 1 câu nên list chỉ có đúng 1 phần tử role="user".
question = "Giải thích ngắn gọn: token trong LLM là gì?"
messages = [{"role": "user", "content": question}]


def main() -> None:
    # --- Task 0: gọi API ----------------------------------------------------
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,          # BẮT BUỘC: trần độ dài câu trả lời (output), không có mặc định.
        system=SYSTEM_PROMPT,    # tách riêng khỏi messages.
        messages=messages,
    )

    # --- Task 1: đọc content ------------------------------------------------
    # response.content là 1 LIST các block (không phải string).
    # Với câu trả lời text thường, block đầu tiên là text -> .content[0].text.
    answer = response.content[0].text

    print(f"Hỏi: {question}")
    print(f"Đáp: {answer}")


if __name__ == "__main__":
    main()
