import os
import sys

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")

import anthropic

client = anthropic.Anthropic(timeout=30)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512

PROMPTS = {
    "1": ("Hỏi đáp",      "Bạn là trợ lý AI thân thiện, trả lời ngắn gọn, chính xác bằng tiếng Việt."),
    "2": ("Phân loại",    "Phân loại cảm xúc văn bản thành đúng 1 nhãn: TÍCH_CỰC / TIÊU_CỰC / TRUNG_LẬP. Chỉ trả về nhãn."),
    "3": ("Tóm tắt",      "Tóm tắt văn bản thành 2-3 gạch đầu dòng ngắn gọn bằng tiếng Việt."),
}

messages = []

def ask_claude(system_prompt: str, user_input: str, temperature: float = 0) -> str:
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            system=system_prompt,
            messages= messages + [{"role": "user", "content": user_input}],
        )
        return resp.content[0].text
    except anthropic.AuthenticationError:
        raise SystemExit("Lỗi xác thực — kiểm tra ANTHROPIC_API_KEY trong .env")
    except anthropic.RateLimitError:
        return "[Vượt giới hạn tốc độ — thử lại sau]"
    except anthropic.APITimeoutError:
        return "[Quá thời gian — thử lại]"
    except anthropic.APIConnectionError:
        return "[Lỗi kết nối mạng]"
    except anthropic.APIError as e:
        return f"[Lỗi API: {e}]"

def main() -> None:

    print("Chế độ: 1=Hỏi đáp  2=Phân loại cảm xúc  3=Tóm tắt  (exit để thoát)")
    mode = input("\nChọn chế độ [1/2/3]: ").strip()
    if mode.lower() in ("exit", "quit", "thoat"):
        print("Tạm biệt!")
        return
    if mode not in PROMPTS:
        print("Chế độ không hợp lệ."); return

    while True:
        label, system_prompt = PROMPTS[mode]
        text = input("Nhập văn bản: ").strip()
        if text.lower() in ("exit", "quit", "thoat"):
            print("Đang thoát chương trình...")
            break
        if not text:
            print("Vui lòng nhập văn bản."); continue
        
        messages.append({"role": "user", "content": text})
        
        answer = ask_claude(system_prompt, text)

        messages.append({"role": "assistant", "content": answer})

        print(f"{label}: {answer}")


if __name__ == "__main__":
    main()