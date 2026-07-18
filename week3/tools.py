import os
import sys
import json
import logging
import datetime
from dotenv import load_dotenv
from utils import parse_json_safely

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

secret_key = os.getenv("ANTHROPIC_API_KEY")
if secret_key is None:
    raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")

import anthropic

client = anthropic.Anthropic(timeout=30)

NOTES_FILE = "notes.json"

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

ADD_NOTES_TOOL = {
    "name": "add_notes",
    "description": "Add new notes to notes.json file to remember and handle the same case in the future",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "text to add notes"}
        },
        "required": ["text"]
    },
}

LIST_NOTES_TOOL = {
    "name": "list_notes",
    "description": "list to stored all notes have add before",
    "input_schema": {"type": "object", "properties": {}}
}

SEARCH_NOTES_TOOL = {
    "name": "search_notes",
    "description": "Will base on keywork provide to search in list notes to show if it exist or not",
    "input_schema": {
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "keyword to search into list notes"}
        },
        "required": ["keyword"],
    },
}

SYSTEM_PROMPT = """\
# Vai trò: trợ lý tính toán, tiết kiệm tool call.
# ĐƯỢC dùng tool: phép tính lớn/nhiều bước; đọc file khi user chỉ rõ path; hỏi giờ.
# KHÔNG dùng tool: phép tính đơn giản (2+2) tự trả lời; câu hỏi khái niệm.
# Cách trả lời: ngắn gọn tiếng Việt; nếu dùng tool nêu 1 câu lý do.
"""

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
    if name == "add_notes":
        return add_notes(tool_input["text"])
    if name == "list_notes":
        return list_notes()
    if name == "search_notes":
        return search_notes(tool_input["keyword"])
    raise ValueError(f"No exist {name} tool!")

def _load_notes() -> list:
    """Đọc notes.json -> list dict. File chưa có / hỏng -> trả list rỗng."""
    if not os.path.exists(NOTES_FILE):
        return []
    try:
        with open(NOTES_FILE, encoding="utf-8") as f:
            return parse_json_safely(f.read(), [])
    except OSError as e:
        logging.warning("Không đọc được %s: %s", NOTES_FILE, e)
        return []

def _save_notes(notes: list) -> None:
    """Ghi list note xuống notes.json (giữ tiếng Việt, format đẹp)."""
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

def add_notes(text: str) -> str:
    notes = _load_notes()
    new_id = max((n["id"] for n in notes), default=0) + 1
    notes.append({"id": new_id, "text": text, "created": get_current_time()})
    _save_notes(notes)
    return f"Đã thêm note #{new_id}: {text!r}"

def list_notes() -> str:
    notes = _load_notes()
    if not notes:
        return "Chưa có note nào."
    return "\n".join(f"#{n['id']} ({n['created']}): {n['text']}" for n in notes)

def search_notes(keyword: str) -> str:
    notes = _load_notes()
    kw = keyword.lower()
    found = [n for n in notes if kw in n["text"].lower()]
    if not found:
        return f"Không tìm thấy note nào chứa {keyword!r}."
    return "\n".join(f"#{n['id']}: {n['text']}" for n in found)


def demo_tool_choice(prompt: str):
    choices = {
        "auto": {"type": "auto"},
        "any": {"type": "any"},
        "tool (ep calculator)": {"type": "tool", "name": "calculator"},
    }
    print(f"\n=== So sanh tool_choice tren prompt: {prompt!r} ===")
    for label, choice in choices.items():
        messages = [{"role": "user", "content": prompt}]
        answer = run_agent_tool(messages, tool_choice=choice, system=None)
        print(f"\n--- tool_choice = {label} ---")
        print("tra loi:", answer)

def extract_info(text: str) -> str:
    prompt = (
        "Trích xuất tên, ngày, số tiền từ đoạn sau. "
        'Chỉ trả JSON dạng {"ten":..., "ngay":..., "so_tien":...}, '
        "thiếu field nào thì để null. Đoạn: " + text
    )

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = next(b.text for b in resp.content if b.type == "text")

    return parse_json_safely(raw)


def run_agent_tool(messages: list, max_turn: int = 10, tool_choice = None, system: str = SYSTEM_PROMPT) -> str:

    for i in range(max_turn):
        kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS,
                      tools=[CALCULATOR_TOOL, TIME_TOOL, FILE_TOOL,
                             ADD_NOTES_TOOL, LIST_NOTES_TOOL, SEARCH_NOTES_TOOL],
                      messages=messages)
        if system:
            kwargs["system"] = system
        if tool_choice and i == 0:
            kwargs["tool_choice"] = tool_choice
        try:
            response = client.messages.create(**kwargs)
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

    return "[Đã đạt số lượng goi tool!]"

def main():
    messages = []

    while True:
        user_input = input("Nhập câu hỏi mà bạn muốn hỏi: ")

        if user_input.lower() in ("thoat", "exit", "quit"):
            print("Tạm biệt!")
            break
        if user_input.lower() == "demo":
            demo_tool_choice("Tinh 1200 + 1500 + 1800 bang bao nhieu?")
            continue
        if user_input.lower() == "extract":
            text = "Tôi tên là Trí, hôm nay là ngày 18/07/2026 tôi đã kiếm được 20 triệu tiền vnd từ công việc hiện tại của tôi"
            result = extract_info(text)
            print("JSON trích xuất:", json.dumps(result, ensure_ascii=False, indent=2))
            continue
        if not user_input:
            print("Làm ơn nhập câu hỏi!")
            continue

        messages.append({"role": "user", "content": user_input})
        
        answer = run_agent_tool(messages)
        print(f"trả lời: {answer}")


if __name__ == "__main__":
    main()