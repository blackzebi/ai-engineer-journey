from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = "Bạn là trợ lý AI thân thiện, trả lời ngắn gọn bằng tiếng Việt."

client = anthropic.Anthropic(timeout=30)

def main():
    messages = [];

    while True:
        user_input = input("bạn hỏi gì: ")

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
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages
            )
        except anthropic.AuthenticationError:
            print("Đã xảy ra lỗi xác thực API.")
            break
        except anthropic.RateLimitError:
            print("Đã vượt quá giới hạn tốc độ API. Vui lòng thử lại sau.")
            messages.pop()  # Remove the last user message to avoid repeating it
            continue
        except anthropic.APITimeoutError:
            print("Yêu cầu API đã hết thời gian chờ. Vui lòng thử lại.")
            messages.pop()  # Remove the last user message to avoid repeating it
            continue
        except anthropic.APIConnectionError:
            print("Đã xảy ra lỗi kết nối API. Vui lòng kiểm tra kết nối mạng của bạn.")
            messages.pop()  # Remove the last user message to avoid repeating it
            continue
        except anthropic.APIStatusError as e:
            print(f"Đã xảy ra lỗi trạng thái API: {e}. Vui lòng thử lại sau.")
            messages.pop()  # Remove the last user message to avoid repeating it
            continue

        answer = response.content[0].text\
        
        messages.append({"role": "assistant", "content": answer})


        print(f"Claude AI: {answer}")


if __name__ == "__main__":
    main()