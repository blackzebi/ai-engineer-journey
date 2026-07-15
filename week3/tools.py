import os
import sys
import datetime
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

secret_key = os.getenv("ANTHROPIC_API_KEY")
if secret_key is None:
    raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")

import anthropic

client = anthropic.Anthropic(timeout=30)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": (
        "A simple calculator that can perform basic arithmetic operations such as addition, subtraction, multiplication, and division."
        "Always using this tool when someone asks you to perform a calculation or some relevant number question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform."
            },
            "a": {
                "type": "number",
                "description": "The first operand."
            },
            "b": {
                "type": "number",
                "description": "The second operand."
            },
        },
        "required": ["operation", "a", "b"]
    },
}

TIME_TOOL = {
    "name": "get_current_time",
    "description": "Get the current time. Used when the user asks for the current time, date, or month.",
    "input_schema": {"type": "object", "properties": {}},
}

FILE_TOOL = {
    "name": "read_file",
    "description": "Read the contents of a text file in the current directory. Use this tool when the user asks about the file's content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "link file, example note.txt"}
        },
        "required": ["path"]
    },
}

def calculator(operation: str, a: float, b: float) -> str:
   if operation == "add": return str(a + b)
   if operation == "subtract": return str(a - b)
   if operation == "multiply": return str(a * b)
   if operation == "divide": return str(a / b)
   raise ValueError(f"Unknown operation: {operation}")

def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def execute_tool(name: str, tool_input: dict) -> str:
    if name == "read_file":
        return read_file(tool_input["path"])
    if name == "get_current_time":
        return get_current_time()
    if name == "calculator":
        return calculator(**tool_input)
    raise ValueError(f"No exist {name} tool!")

def run_agent_tool(messages: list) -> str:

    while True:
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                tools=[CALCULATOR_TOOL, TIME_TOOL, FILE_TOOL],
                messages=messages
            )
        except anthropic.AuthenticationError:
            raise SystemExit("Lỗi xác thực — kiểm tra ANTHROPIC_API_KEY trong .env")
        except anthropic.APITimeoutError:
            messages.pop()
            return "[Yêu cầu API đã hết thời gian chờ. Vui lòng thử lại.]"
        except anthropic.APIConnectionError:
            messages.pop()
            return "[Lỗi kết nối mạng]"
        except anthropic.APIError as e:
            messages.pop()
            return f"[Lỗi API: {e}]"


        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            return next(b.text for b in response.content if b.type == "text")
        
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                try: 
                    result = execute_tool(block.name, block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"lỗi khi chạy tool: {type(e).__name__}: {e}",
                        "is_error": True
                    })

        messages.append({"role": "user", "content": tool_results})


def main():
    messages = []

    while True:
        user_input = input("Nhập câu hỏi mà bạn muốn hỏi: ")

        if user_input.lower() in ("thoat", "exit", "quit"):
            print("Tạm biệt!")
            break
        if not user_input:
            print("Làm ơn nhập câu hỏi!")
            continue

        messages.append({"role": "user", "content": user_input})
        
        answer = run_agent_tool(messages)
        print(f"trả lời: {answer}")


if __name__ == "__main__":
    main()