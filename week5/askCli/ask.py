"""
ask.py — CLI hỏi-đáp tài liệu. Mặt tiền của Dự án 1.

    INPUT :  python ask.py "câu hỏi" [--k 3] [--doc-type pdf]
    OUTPUT:  câu trả lời stream ra màn hình + nguồn [1][2][3] kèm điểm similarity
             + dòng tổng kết token/chi phí, và 1 dòng log JSONL

Tiêu chí: người khác clone repo về là chạy được, không phải hỏi thêm. Nên file này bị chấm ở
những chỗ không phải tính năng: lỗi có dễ hiểu không · thiếu config có nói rõ phải làm gì
không · key có lọt lên GitHub không.

Luồng, và chỗ mỗi module chịu trách nhiệm:

    argv ──parse_args──► question, k, doc_type
                              │
              errors.validate_question   ← chặn câu hỏi rỗng TRƯỚC khi tốn tiền
                              │
              retriever.retrieve ──► chunk đã lọc ngưỡng similarity
                              │              └─ rỗng ⇒ in "không tìm thấy", KHÔNG gọi LLM
                              │
              stream_answer.stream_and_collect ──► text + usage
                              │
              retriever.format_sources · estimate_cost · log_query

    Mọi lỗi trên đường đi ──AskError──► main() ──► format_user_error + exit code
                                          ▲
                                   CHỈ chỗ này được phép exit

Tái dùng: week4/ragLite/prompt_rag.py (qua stream_answer) · week4/pgvector/vector_ops.py
và week4/ragPipeline/search_pg.py (qua retriever). Không sửa một dòng nào của tuần 4.

Chạy:
    python ask.py "embedding là gì?"
    python ask.py "kết luận của báo cáo là gì?" --k 5 --doc-type pdf
    python ask.py --check          # chỉ kiểm tra cấu hình, không hỏi gì, không tốn tiền
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv

from errors import EXIT_CODES, AskError, ErrorKind, format_user_error, validate_question
from retriever import DEFAULT_TOP_K, MIN_SIMILARITY, format_sources, retrieve
from stream_answer import (
    estimate_cost,
    get_client,
    log_query,
    print_cost_line,
    stream_and_collect,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Biến môi trường bắt buộc + gợi ý sửa cho từng cái.
REQUIRED_ENV_VARS = {
    "ANTHROPIC_API_KEY": "lấy ở console.anthropic.com, dán vào .env",
    "DATABASE_URL": "xem .env.example — mặc định trỏ tới container ở week4/pgvector",
}

VALID_DOC_TYPES = ("pdf", "docx", "md", "txt")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """argv -> Namespace(question, k, doc_type, check).

    Ví dụ: ['RAG là gì?', '--k', '5'] -> Namespace(question='RAG là gì?', k=5, ...)

    Dùng argparse chứ không tự cắt sys.argv như tuần 4: bản tự làm
    `sys.argv[sys.argv.index("--doc-type") + 1]` nổ IndexError khi người dùng quên giá trị,
    và không có `-h` để người khác biết phải gõ gì. argparse cho miễn phí --help, báo lỗi sai
    kiểu, giá trị mặc định, và choices= chặn `--doc-type pdff` ngay tại cổng.

    Nhận argv làm THAM SỐ thay vì tự đọc sys.argv -> hàm này test được bằng list tự chế.

    Giới hạn đã biết: câu hỏi tiếng Việt có dấu cách, người dùng rất hay quên nháy kép
    (`python ask.py RAG là gì`) -> argparse coi 'là' là tham số thừa. Chọn nargs="?" + epilog
    nhắc dùng nháy kép thay vì nargs="*" rồi tự join, vì thông điệp lỗi vẫn rõ.
    """
    parser = argparse.ArgumentParser(
        prog="ask.py",
        description="Hỏi-đáp tài liệu đã ingest (RAG). Câu trả lời stream + trích nguồn.",
        epilog='Ví dụ: python ask.py "RAG là gì?" --k 5 --doc-type pdf',
    )

    # nargs="?" để `python ask.py --check` không bị đòi câu hỏi -> question có thể là None,
    # main() phải xử lý trước khi đưa xuống validate_question.
    parser.add_argument("question", nargs="?", default=None,
                        help="câu hỏi, nhớ đặt trong nháy kép")

    # type=int là bắt buộc: thiếu thì k là chuỗi "5", truyền xuống LIMIT %s psycopg vẫn chạy
    # (Postgres tự cast) nhưng so sánh số ở chỗ khác thì nổ ở tận đâu đó.
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K,
                        help=f"số chunk lấy về (mặc định {DEFAULT_TOP_K})")

    # dấu gạch trong --doc-type tự thành gạch dưới: args.doc_type
    parser.add_argument("--doc-type", choices=VALID_DOC_TYPES, default=None,
                        help="chỉ tìm trong 1 định dạng")

    parser.add_argument("--check", action="store_true",
                        help="chỉ kiểm tra cấu hình rồi thoát")

    return parser.parse_args(argv)


def check_env() -> list[str]:
    """Trả danh sách biến môi trường bắt buộc còn THIẾU. Rỗng nghĩa là đủ.

    Ví dụ: thiếu key -> ['ANTHROPIC_API_KEY']

    Kiểm tra TẤT CẢ rồi mới báo, thay vì gặp cái thiếu đầu tiên là ném luôn: báo từng cái một
    là bắt người ta sửa–chạy–lại 3 vòng mới biết hết mình thiếu gì.

    ⚠️ Chỉ trả về TÊN biến, tuyệt đối không in giá trị key ra màn hình để "debug cho tiện" —
    terminal đó có thể đang được quay màn hình.
    """
    missing = []
    for name in REQUIRED_ENV_VARS:
        # os.getenv trả chuỗi RỖNG khi .env viết `KEY=` (có dòng, không có giá trị).
        # `if not value` bắt được; `if name not in os.environ` thì KHÔNG — biến có tồn tại,
        # chỉ là rỗng. Đây là tình huống copy .env.example rồi quên điền, rất hay gặp.
        value = (os.getenv(name) or "").strip()
        if not value:
            missing.append(name)
        # Bắt trường hợp còn nguyên placeholder của .env.example.
        elif value.endswith("...") or value in ("sk-ant-...", "changeme"):
            missing.append(name)

    return missing


def run_once(question: str, k: int, doc_type: str | None) -> int:
    """Chạy trọn 1 câu hỏi. Trả exit code.

    Thứ tự các bước không ngẫu nhiên:
      1. validate  : chặn câu rác trước, vì embed + query + LLM đều tốn thời gian và tiền
      2. retrieve  : đã lọc ngưỡng bên trong; rỗng ⇒ dừng, KHÔNG gọi LLM (lớp chống bịa)
      3. stream    : chỉ tới đây mới tiêu tiền, và chỉ khi đã chắc có căn cứ
      4. nguồn     : in SAU câu trả lời — người đọc cần đối chiếu ngay khi vừa đọc xong
      5. log       : cuối cùng, và ghi cả trường hợp không tìm thấy (0 chunk cũng là dữ liệu
                     — tuần 7 cần đúng những câu này để biết retrieval hụt ở đâu)
    """
    question = validate_question(question)
    started_at = time.perf_counter()

    chunks = retrieve(question, k=k, doc_type=doc_type)
    top_similarity = chunks[0].similarity if chunks else 0.0

    if not chunks:
        print(f"\n🤷 Không tìm thấy đoạn nào đủ liên quan (ngưỡng {MIN_SIMILARITY}).")
        print("   → Tài liệu đã ingest chưa?  python ../ingestDocs/ingest_docs.py \"<folder>\"")
        if doc_type:
            print(f"   → Hoặc thử bỏ --doc-type {doc_type}, có thể câu trả lời nằm ở định dạng khác.")
        log_query({
            "question": question, "k": k, "doc_type": doc_type,
            "n_chunks": 0, "top_similarity": 0.0,
            "input_tokens": 0, "output_tokens": 0, "total_usd": 0.0,
            "elapsed_seconds": round(time.perf_counter() - started_at, 2),
            "outcome": "no_results",
        })
        return EXIT_CODES[ErrorKind.NO_RESULTS]

    client = get_client()
    print()
    text, usage = stream_and_collect(client, question, chunks)

    print("\n" + format_sources(chunks))
    cost = estimate_cost(usage["input_tokens"], usage["output_tokens"])
    elapsed = time.perf_counter() - started_at
    print_cost_line(usage, cost, elapsed, len(chunks))

    log_query({
        "question": question, "k": k, "doc_type": doc_type,
        "n_chunks": len(chunks), "top_similarity": round(top_similarity, 4),
        **usage, **cost,
        "elapsed_seconds": round(elapsed, 2),
        "answer_chars": len(text),
        "outcome": "answered",
    })
    return 0


def main() -> int:
    """Entry point. CHỖ DUY NHẤT được bắt lỗi và quyết định exit code.

    Mọi hàm bên dưới chỉ RAISE. Gom việc bắt lỗi về một chỗ nên: thông điệp lỗi nhất quán,
    và muốn đổi cách hiển thị (thêm màu, ghi ra file) thì sửa đúng một nơi.
    """
    args = parse_args(sys.argv[1:])

    missing = check_env()
    if missing:
        print("❌ Thiếu cấu hình:")
        for name in missing:
            print(f"   - {name}: {REQUIRED_ENV_VARS[name]}")
        return EXIT_CODES[ErrorKind.MISSING_CONFIG]

    if args.check:
        print("✅ Cấu hình đủ. Thử: python ask.py \"câu hỏi của bạn\"")
        return 0

    if args.question is None:
        print('Dùng: python ask.py "câu hỏi của bạn" [--k 3] [--doc-type pdf]')
        return EXIT_CODES[ErrorKind.EMPTY_QUESTION]

    try:
        return run_once(args.question, k=args.k, doc_type=args.doc_type)
    except AskError as error:
        print("\n" + format_user_error(error))
        return EXIT_CODES.get(error.kind, 1)
    except KeyboardInterrupt:
        print("\n⏹  Đã huỷ.")
        return 130                      # quy ước Unix: 128 + SIGINT(2)
    except Exception as error:          # noqa: BLE001 — lưới hứng cuối cùng, cố ý
        # Lỗi CHƯA phân loại thì in traceback THẬT. Nuốt nó đi là tự bịt mắt mình:
        # người dùng báo "app lỗi" mà không có manh mối nào để lần.
        import traceback

        print(f"\n❌ Lỗi ngoài dự kiến: {type(error).__name__}: {error}")
        traceback.print_exc()
        print("\n→ Đây là lỗi chưa được phân loại. Thêm nó vào errors.ErrorKind "
              "và classify_exception() để lần sau có thông điệp tử tế.")
        return EXIT_CODES[ErrorKind.UNKNOWN]


if __name__ == "__main__":
    sys.exit(main())

    # ✅ ĐẠT khi (6 phép thử):
    #   1. python ask.py -h                       -> bảng hướng dẫn có ví dụ
    #   2. python ask.py --check                  -> "Cấu hình đủ"
    #   3. python ask.py ""                       -> 1 dòng gọn, KHÔNG có chữ Traceback
    #   4. python ask.py "embedding là gì?"       -> chữ hiện DẦN + nguồn [1][2] + dòng chi phí
    #   5. docker compose stop, rồi chạy lại (4)  -> gợi ý "docker compose up -d", không traceback
    #   6. python ask.py "giá vàng hôm nay?"      -> "không tìm thấy", và log cho thấy 0 token
