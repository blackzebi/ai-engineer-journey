from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = "Bạn là trợ lý AI thân thiện, bạn sẻ giúp tôi tóm tắt các đoạn văn bản dài thành các đoạn văn bản ngắn gọn thành 3 gạch đầu dòng"
# SYSTEM_PROMPT2 = "Bạn là trợ lý AI hung dữ, cọc cằn và thô lỗ, bạn sẽ giúp tôi tóm tắt các đoạn văn bản thành các gạch đầu dòng."
MAX_TOKENS = 512

client = anthropic.Anthropic(timeout=30)

def summarize_text():
    messages = []

    while True:
        user_input = input("Nhập văn bản cần tóm tắt: ")

        if user_input.lower() in ("exit", "quit", "thoat"):
            print("Đang thoát chương trình...")
            break

        if not user_input:
            print("Vui lòng nhập văn bản.")
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
            continue
        except anthropic.RateLimitError:
            print("Đã vượt quá giới hạn tốc độ API. Vui lòng thử lại sau.")
            continue

        messages.append({"role": "assistant", "content": response.content[0].text})

        print(f"Tóm tắt: {response.content[0].text}")

def main():
    summarize_text()

if __name__ == "__main__":
    main()