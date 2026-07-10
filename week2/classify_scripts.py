import os
import sys

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")

import anthropic

client = anthropic.Anthropic(timeout = 30, api_key=api_key)

MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = "Bạn là trợ lý AI thân thiện, bạn sẽ giúp tôi phân loại cảm xuc của văn bản thành 3 loại: tích cực, tiêu cực, trung lập. Trả lời ngắn gọn và chính xác bằng tiếng Việt trong một dòng."

def classify_text(text: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return response.content[0].text


def main() -> None:
    while True:
        user_input = input("Nhập văn bản để phân loại cảm xúc (hoặc 'exit' để thoát): ")

        if user_input.lower() in ("exit", "quit", "thoat"):
            print("Đang thoát chương trình...")
            break
        if not user_input:
            print("Vui lòng nhập văn bản.")
            continue

        try:
            classification = classify_text(user_input)
            print(f"Phân loại cảm xúc: {classification}")
        except anthropic.AuthenticationError:
            print("Đã xảy ra lỗi xác thực API.")
            break
        except anthropic.RateLimitError:
            print("Đã vượt quá giới hạn tốc độ API. Vui lòng thử lại sau.")
            continue
        except anthropic.APIStatusError as e:
            print(f"Đã xảy ra lỗi trạng thái API: {e}. Vui lòng thử lại sau.")
            continue
        except anthropic.APIConnectionError as e:
            print(f"Đã xảy ra lỗi kết nối API: {e}. Vui lòng thử lại sau.")
            continue


if __name__ == "__main__":
    main()