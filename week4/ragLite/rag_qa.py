"""
rag_qa.py — Mini-project #5: RAG-lite hỏi-đáp notes end-to-end (T5 tuần 4, PROJECT 11:00–14:00)

    INPUT :  1 câu hỏi tiếng Việt về notes tuần 1–4
    OUTPUT:  câu trả lời của Claude + top-3 đoạn nguồn (file, điểm similarity)

Luồng đầy đủ của RAG (ghép mọi thứ đã học 4 tuần):

    câu hỏi ──embed──► vector ──pgvector ORDER BY <=>──► top-3 chunk
                                                              │
        câu trả lời ◄──Claude──◄ prompt (context + câu hỏi) ◄─┘

Tái dùng, KHÔNG viết lại:
    - ../pgvector/vector_ops.py     : get_conn, search_top_k         (block sáng T4)
    - ../ragPipeline/search_pg.py   : embed_query, get_model          (T4 chiều)
    - ../../week3/utils.py          : parse_json_safely               (tuần 3 — parse an toàn)
    - ./prompt_rag.py               : build_rag_prompt, format_citations, SYSTEM_PROMPT (sáng nay)

Chạy:
    python rag_qa.py "làm sao gọi API bất đồng bộ trong Python?"
    python rag_qa.py --eval          # chạy 5 câu test trong questions.json
"""

import json
import os
import sys

# --- Cho phép import chéo các module tuần trước ---------------------------------
HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "pgvector"))       # vector_ops
sys.path.append(os.path.join(HERE, "..", "ragPipeline"))    # search_pg
sys.path.append(os.path.join(HERE, "..", "..", "week3"))    # utils (parse an toàn)

from dotenv import load_dotenv                                  # noqa: E402

from prompt_rag import SYSTEM_PROMPT, build_rag_prompt, format_citations  # noqa: E402
from search_pg import embed_query                               # noqa: E402
from vector_ops import get_conn, search_top_k                   # noqa: E402
from utils import parse_json_safely                             # noqa: E402  (tuần 3)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"   # rẻ + đủ cho hỏi-đáp notes; đổi sang sonnet nếu cần chất
MAX_TOKENS = 512
TOP_K = 3                             # top-3 nguồn cho mỗi câu trả lời

# Chuỗi model PHẢI nói khi không có dữ kiện — trùng với câu trong SYSTEM_PROMPT.
# Dùng ở eval để tự động chấm 2 câu "không có trong tài liệu".
NO_ANSWER = "Tôi không tìm thấy thông tin này trong tài liệu."


def get_client():
    """Khởi tạo Anthropic client 1 lần, báo lỗi rõ ràng nếu thiếu API key."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("❌ ANTHROPIC_API_KEY chưa set. Copy .env.example -> .env và điền.")
    import anthropic
    return anthropic.Anthropic(timeout=30)


def retrieve(question: str, k: int = TOP_K):
    """Câu hỏi -> top-k chunk từ pgvector. Trả [(id, source, content, similarity), ...].

    Tái dùng nguyên si retrieval của T4 — đây chỉ là lớp mỏng gọi lại.

    Error handling: bọc get_conn() trong try/except để nếu container
    Postgres chưa chạy thì báo "DB chưa sẵn sàng, chạy docker compose up -d" thay vì stacktrace.
    """
    vec = embed_query(question)
    try:
        with get_conn() as conn:
            return search_top_k(conn, vec, k)
    except Exception as e:
        # KHÔNG dùng `except:` trần: nó nuốt cả lỗi thật (typo, KeyError...) rồi báo nhầm
        # là "DB chưa sẵn sàng" -> debug rất mệt. Luôn bắt Exception + in kèm lỗi gốc.
        raise SystemExit(
            f"❌ Không truy vấn được DB: {type(e).__name__}: {e}\n"
            "   Nếu là lỗi kết nối: chạy `docker compose up -d` trong ../pgvector"
        )


def generate(client, question: str, chunks) -> str:
    """Gọi Claude với context = các chunk, trả về text câu trả lời.

    Bẫy: đừng gộp SYSTEM_PROMPT vào messages — để ở tham số `system=` cho đúng chuẩn API
    và để model phân biệt "luật" với "dữ liệu".
    """
    if not chunks:
        return NO_ANSWER
    prompt = build_rag_prompt(question, chunks)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def answer(question: str, k: int = TOP_K) -> dict:
    """Hàm end-to-end 1 câu hỏi. Trả dict để vừa in đẹp vừa dùng lại trong eval.

    Trả:
        {"question": ..., "answer": ..., "sources": [(source, similarity), ...]}
    """
    client = get_client()
    chunks = retrieve(question, k=k)
    text = generate(client, question, chunks)
    sources = [(source, sim) for (_id, source, _content, sim) in chunks]
    return {"question": question, "answer": text, "sources": sources}


def print_answer(result: dict, chunks=None) -> None:
    """In câu trả lời + danh sách nguồn (trích nguồn để người đọc kiểm chứng)."""
    print(f"\n❓ {result['question']}")
    print(f"\n💬 {result['answer']}\n")
    if chunks:
        print(format_citations(chunks))
    else:
        print("Nguồn tham khảo:")
        for i, (source, sim) in enumerate(result["sources"], 1):
            print(f"  [{i}] {source}  (similarity {sim:.2f})")


def run_eval(path: str = "questions.json") -> None:
    """Chạy 5 câu test và tự chấm thô — chính là 'eval đầu tiên' của portfolio.

    questions.json: [{"q": "...", "expect": "answerable"|"no_answer", "note": "..."}]
    Tiêu chí chấm tự động (thô, đủ cho tuần này):
        - expect == "no_answer"  -> PASS nếu câu trả lời CÓ chứa NO_ANSWER
        - expect == "answerable" -> PASS nếu KHÔNG chứa NO_ANSWER và có ≥1 nguồn
      (chất lượng nội dung câu trả lời vẫn phải TỰ đọc mắt thường — eval xịn để tuần 7.)
    """
    raw = open(os.path.join(HERE, path), encoding="utf-8").read()
    cases = parse_json_safely(raw, default=[])
    if not cases:
        raise SystemExit(f"❌ Không đọc được {path} (rỗng hoặc sai JSON).")
    passed = 0
    rows = []
    for c in cases:
        res = answer(c["q"])
        has_no = NO_ANSWER in res["answer"]
        # bool(...) là BẮT BUỘC: chuỗi `A and B and res["sources"]` trả về chính cái LIST
        # sources (truthy) chứ không phải True -> `passed += ok` sẽ nổ TypeError (int + list).
        ok = bool(
            (c["expect"] == "no_answer" and has_no)
            or (c["expect"] == "answerable" and not has_no and res["sources"])
        )
        passed += ok
        print_answer(res)
        print(f"   → kỳ vọng={c['expect']} | {'✅ PASS' if ok else '❌ FAIL'}")
        rows.append((c["q"], c["expect"], "PASS" if ok else "FAIL"))

    print(f"\n===== KẾT QUẢ: {passed}/{len(cases)} PASS =====")

    print("\n| # | Câu hỏi | Kỳ vọng | Kết quả |\n|---|---|---|---|")
    for i, (q, exp, r) in enumerate(rows, 1):
        print(f"| {i} | {q} | {exp} | {r} |")


def main() -> None:
    if "--eval" in sys.argv:
        run_eval()
        return

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit('Dùng: python rag_qa.py "câu hỏi của bạn"   |   python rag_qa.py --eval')

    question = " ".join(args)
    client = get_client()
    chunks = retrieve(question)
    text = generate(client, question, chunks)
    result = {
        "question": question,
        "answer": text,
        "sources": [(source, sim) for (_id, source, _content, sim) in chunks],
    }
    print_answer(result, chunks)


if __name__ == "__main__":
    main()
