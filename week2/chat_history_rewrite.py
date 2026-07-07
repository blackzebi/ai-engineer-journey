from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = "Bạn là trợ lý AI thân thiện, trả lời ngắn gọn và chính xác bằng tiếng Việt."
MAX_TOKENS = 512

client = anthropic.Anthropic(timeout=30)

def chat_history():
    messages = []

    while True:
        user_input = input("Bạn hỏi gì: ")

        if user_input.lower() in ("exit", "quit", "thoat"):
            print("Đang thoát chương trình...")
            break
        if not user_input:
            print("Vui lòng nhập câu hỏi.")
            continue

        messages.append({"role": "user", "content": user_input})
    
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
                temperature=0,
            )

        except anthropic.AuthenticationError:
            print("Đã xảy ra lỗi xác thực API.")
            break
        except anthropic.RateLimitError:
            print("Đã vượt quá giới hạn tốc độ API. Vui lòng thử lại sau.")
            messages.pop()  # Remove the last user message to avoid repeating it
            continue
        except anthropic.APIStatusError as e:
            print(f"Đã xảy ra lỗi trạng thái API: {e}. Vui lòng thử lại sau.")
            messages.pop()  # Remove the last user message to avoid repeating it
            continue

        messages.append({"role": "assistant", "content": response.content[0].text})
        print(f"Claude AI: {response.content[0].text}")


def main():
    chat_history()


if __name__ == "__main__":
    main()
