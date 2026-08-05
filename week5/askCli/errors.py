"""
errors.py — Danh mục lỗi của app RAG + hành vi rõ ràng cho từng loại

Thứ giết trải nghiệm người dùng nhanh nhất không phải thiếu tính năng — là traceback 40 dòng
đập vào mặt họ khi họ chỉ quên bật Docker.

    tuần 4:  lỗi gì cũng -> raise SystemExit(chuỗi tự chế tại chỗ)
             (xem week4/ragLite/rag_qa.py::retrieve — SystemExit nằm SÂU trong hàm thư viện)
    tuần 5:  mọi lỗi -> AskError(kind, detail) --bay lên main()--> 1 khối in duy nhất
                                                    ▲
                                              chỉ main() mới được quyết định in gì và thoát

Vì sao KHÔNG raise SystemExit trong hàm sâu:
    Hàm nào gọi sys.exit là hàm đó tự quyết định số phận cả chương trình. Dùng lại retrieve()
    trong Streamlit / FastAPI / test thì nó giết luôn server, và không viết được test cho
    nhánh lỗi vì process chết trước khi kịp assert.
    Luật: hàm thư viện RAISE, chỉ entry point (main) mới CATCH và EXIT.

Danh mục lỗi — mỗi loại 1 hành vi, không loại nào rơi vào 'unknown':

    kind                | khi nào                        | hành vi
    --------------------|--------------------------------|---------------------------
    MISSING_CONFIG      | thiếu ANTHROPIC_API_KEY/DB_URL | in cách sửa, exit 3
    EMPTY_QUESTION      | người dùng gõ "" hoặc "   "    | in cách dùng, exit 2
    DB_UNAVAILABLE      | container Postgres chưa Up     | gợi ý docker compose up -d
    NO_RESULTS          | retrieve trả 0 chunk           | "chưa ingest tài liệu?" exit 0
    LOW_CONFIDENCE      | top-1 similarity < ngưỡng      | "không tìm thấy" — KHÔNG gọi LLM
    API_RATE_LIMIT      | 429 từ Anthropic               | retry backoff, hết thì báo
    API_FAILED          | 4xx/5xx khác, timeout          | in mã lỗi gốc, exit 5
    UNKNOWN             | còn lại                        | in traceback THẬT (đừng nuốt)

Self-test bằng exception giả — không cần DB, không cần API key, không cần model:
    python errors.py
"""

from __future__ import annotations

import sys
import time
from enum import Enum

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ErrorKind(str, Enum):
    """Danh mục lỗi. Kế thừa `str` để `kind.value` dùng thẳng làm khoá dict / ghi log JSON."""

    MISSING_CONFIG = "missing_config"
    EMPTY_QUESTION = "empty_question"
    DB_UNAVAILABLE = "db_unavailable"
    NO_RESULTS = "no_results"
    LOW_CONFIDENCE = "low_confidence"
    API_RATE_LIMIT = "api_rate_limit"
    API_FAILED = "api_failed"
    UNKNOWN = "unknown"


class AskError(Exception):
    """Lỗi đã được PHÂN LOẠI. Ném ở tầng sâu, bắt ở main().

    kind   : thuộc ErrorKind — quyết định thông điệp + exit code
    detail : chi tiết kỹ thuật (tên exception gốc, message gốc) để debug
    """

    def __init__(self, kind: ErrorKind, detail: str = "") -> None:
        super().__init__(f"{kind.value}: {detail}")
        self.kind = kind
        self.detail = detail


# Exit code khác nhau để chạy trong script/CI còn phân biệt được nguyên nhân.
# 0 = không phải lỗi hệ thống (app trả lời "không tìm thấy" vẫn là chạy đúng).
EXIT_CODES: dict[ErrorKind, int] = {
    ErrorKind.NO_RESULTS: 0,
    ErrorKind.LOW_CONFIDENCE: 0,
    ErrorKind.EMPTY_QUESTION: 2,
    ErrorKind.MISSING_CONFIG: 3,
    ErrorKind.DB_UNAVAILABLE: 4,
    ErrorKind.API_FAILED: 5,
    ErrorKind.API_RATE_LIMIT: 5,
    ErrorKind.UNKNOWN: 1,
}

MAX_QUESTION_CHARS = 500      # dài hơn mức này gần như chắc chắn là dán nhầm cả file vào
MIN_QUESTION_CHARS = 3        # "ok", "?" — không đủ để embed ra vector có nghĩa

# Lỗi đáng retry: là lỗi TẠM THỜI, thử lại có cơ may khác kết quả.
# DB_UNAVAILABLE không nằm đây: container chưa Up thì retry 5 lần vẫn chưa Up,
# chỉ tổ bắt người dùng chờ 15 giây rồi vẫn nhận đúng lỗi đó.
RETRYABLE_KINDS = frozenset({ErrorKind.API_RATE_LIMIT})


def classify_exception(exc: BaseException) -> ErrorKind:
    """Exception bất kỳ (psycopg / anthropic / httpx) -> ErrorKind.

    Ví dụ: psycopg.OperationalError("could not connect") -> ErrorKind.DB_UNAVAILABLE

    Nhận diện bằng TÊN CLASS chứ không `isinstance`, để module này không phải import
    anthropic/psycopg ở đầu file — self-test chạy tức thì, không kéo theo cả cụm dependency.
    Cái giá: thư viện đổi tên class ở phiên bản mới thì hỏng NGẦM. Vì vậy nhánh UNKNOWN
    bắt buộc phải in tên class thật ra, đó là cơ chế duy nhất để lỗi lộ ra thay vì trôi đi.

    Thứ tự kiểm tra từ CỤ THỂ tới TỔNG QUÁT: RateLimitError là con của APIStatusError,
    bắt cha trước thì 429 bị gán nhầm API_FAILED và mất luôn đường retry.

    Giới hạn đã biết: TimeoutError / httpx.ConnectError xuất hiện ở CẢ phía DB lẫn phía API,
    nhìn tên exception không phân biệt được. Nên chỗ gọi DB tự bọc và ném DB_UNAVAILABLE sẵn.
    Hàm này chỉ là lưới hứng cuối cùng.
    """
    # Trả sớm: AskError cũng là Exception, không chặn ở đây thì kind gốc bị ghi đè thành UNKNOWN.
    if isinstance(exc, AskError):
        return exc.kind

    name = type(exc).__name__     # không phải exc.__name__ — exception là INSTANCE
    message = str(exc).lower()

    if name in ("RateLimitError", "OverloadedError"):
        return ErrorKind.API_RATE_LIMIT
    if name in ("OperationalError", "InterfaceError"):
        return ErrorKind.DB_UNAVAILABLE
    if name.startswith(("API", "Anthropic")) or "anthropic" in message:
        return ErrorKind.API_FAILED

    # Lưới hứng cuối: 429 đôi khi chỉ nằm trong message, không có class riêng.
    if "429" in message or "rate limit" in message:
        return ErrorKind.API_RATE_LIMIT
    return ErrorKind.UNKNOWN


def format_user_error(error: AskError) -> str:
    """AskError -> khối text in ra màn hình: chuyện gì + phải làm gì.

    Ví dụ: AskError(DB_UNAVAILABLE, "connection refused") ->
        ❌ Không kết nối được cơ sở dữ liệu.
           → chạy: docker compose up -d (trong week4/pgvector), rồi thử lại.
           chi tiết: connection refused

    Một thông báo lỗi tốt có đúng 3 phần: CHUYỆN GÌ (người thường hiểu được) · LÀM GÌ TIẾP
    (câu lệnh copy-paste được) · CHI TIẾT KỸ THUẬT (cho mình debug). Thiếu phần 2 là biến
    người dùng thành thám tử.

    Dùng bảng dict thay vì chuỗi if/elif vì bảng đọc được như một bản đặc tả — nhìn phát
    biết còn thiếu loại lỗi nào chưa xử lý.
    """
    MESSAGES: dict[ErrorKind, tuple[str, str]] = {
        ErrorKind.DB_UNAVAILABLE: (
            "Không kết nối được cơ sở dữ liệu.",
            "chạy: docker compose up -d (trong week4/pgvector), rồi thử lại.",
        ),
        ErrorKind.MISSING_CONFIG: (
            "Thiếu cấu hình bắt buộc.",
            "Copy .env.example thành .env rồi điền ANTHROPIC_API_KEY."
        ),
        ErrorKind.EMPTY_QUESTION: (
            "Thiếu câu hỏi.",
            "Cần nhập câu hỏi mà bạn muốn hỏi.",
        ),
        ErrorKind.NO_RESULTS: (
            "Thiếu tài liệu.",
            "chạy: `python ingest_docs.py 'ingest-folder-path'` ",
        ),
        ErrorKind.LOW_CONFIDENCE: (
            "Không tìm thấy thông tin mà bạn đang cần.",
            "Bạn cần ingest thêm tài liệu chứa thông tin đó",
        ),
        ErrorKind.API_RATE_LIMIT:  (
            "Đạt giới hạn API limit.",
            "Bạn có thể tiến hành chạy lại."
        ),
        ErrorKind.API_FAILED: (
            "Lỗi API.",
            "Bạn cần kiểm tra lại lỗi chi tiết gây ra là gì",
        ),
        ErrorKind.UNKNOWN: (
            "Lỗi không xác định.",
            "Bạn cần kiểm tra lại lỗi chi tiết gây ra là gì"
        ),
     }
    # .get() chứ không [kind]: quên một ErrorKind thì [kind] nổ KeyError NGAY TRONG hàm
    # đang xử lý lỗi — lỗi khi đang báo lỗi, giấu mất nguyên nhân gốc.
    title, hint = MESSAGES.get(
        error.kind, ("Lỗi không xác định.", "Xem chi tiết bên dưới và mở issue.")
    )

    lines = [f"❌ {title}", f"   → {hint}"]
    if error.detail:
        lines.append(f"   chi tiết: {error.detail}")

    return "\n".join(lines)


def validate_question(raw: str) -> str:
    """Kiểm tra + làm sạch câu hỏi. Trả câu hỏi sạch, hoặc ném AskError(EMPTY_QUESTION).

    Ví dụ: '  Embedding là gì?  ' -> 'Embedding là gì?'   |   '   ' -> AskError

    Vì sao chặn ở ĐẦU VÀO chứ không để pipeline tự chết: câu hỏi rỗng vẫn embed được, vẫn
    có top-k, Claude vẫn trả lời. Không dòng lỗi nào. Người dùng nhận về câu trả lời bịa
    hoàn toàn và mình tốn tiền token cho nó — bug im lặng CÓ TÍNH PHÍ.

    Chặn trên cũng cần: dán nhầm cả file vào là một request vài chục nghìn token, lỗi đó
    chỉ hiện ra ở hoá đơn cuối tháng.

    Cố ý KHÔNG "sửa hộ" câu hỏi (thêm dấu ?, viết hoa). Sửa im lặng input của người dùng
    thì khi kết quả sai họ không hiểu vì sao. strip() là đủ.
    """
    # strip TRƯỚC rồi mới kiểm tra: `if not raw` không bắt được "   " (chuỗi space là truthy).
    question = (raw or "").strip()

    if len(question) < MIN_QUESTION_CHARS:
        raise AskError(ErrorKind.EMPTY_QUESTION,
                       f"câu hỏi chỉ có {len(question)} ký tự")

    if len(question) > MAX_QUESTION_CHARS:
        raise AskError(ErrorKind.EMPTY_QUESTION,
                       f"câu hỏi dài {len(question)} ký tự, tối đa {MAX_QUESTION_CHARS}")

    return question


def retry_with_backoff(call, attempts: int = 3, base_delay: float = 1.0):
    """Gọi `call()` tối đa `attempts` lần, giãn cách gấp đôi mỗi lần, chỉ retry lỗi tạm thời.

    Ví dụ: lần 1 lỗi 429 -> chờ 1s -> lần 2 lỗi 429 -> chờ 2s -> lần 3 OK -> trả kết quả.

    Chờ gấp đôi chứ không chờ đều: 429 nghĩa là phía server đang quá tải hoặc mình gọi quá
    nhanh; retry đều đặn là tiếp tục dội vào đúng lúc nó yếu nhất.
    (Production còn cộng jitter 0–0.3s để nhiều client không cùng retry một thời điểm.)

    `call` là hàm CHƯA gọi. Chỗ gọi truyền `lambda: client.messages.create(...)` — có lambda
    thì lời gọi mới bị hoãn để hàm này chủ động gọi lại được.
    """
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            last_error = exc
            # Không đáng retry thì ném NGAY. Retry mọi exception là sai: sai API key hay
            # typo tên biến thì thử 3 lần vẫn sai y hệt, chỉ tổ bắt người dùng chờ 7 giây.
            kind = classify_exception(exc)
            if kind not in RETRYABLE_KINDS:
                raise

            if attempt == attempts - 1:
                break
            delay = base_delay * (2 ** attempt)
            print(f"   ⏳ rate limit, chờ {delay:.0f}s rồi thử lại "
                  f"({attempt + 2}/{attempts})...")
            time.sleep(delay)
    # Hết lượt thì NÉM, không return/pass: trả None thì chỗ gọi nổ AttributeError ở một chỗ
    # cách nguyên nhân hàng chục dòng.
    raise AskError(ErrorKind.API_RATE_LIMIT,
                   f"thử {attempt} lần vẫn lỗi: {last_error}")

if __name__ == "__main__":
    # Exception giả: không cần cài anthropic/psycopg, không cần mạng.
    class RateLimitError(Exception):
        """Giả lập anthropic.RateLimitError."""

    class OperationalError(Exception):
        """Giả lập psycopg.OperationalError."""

    class APIStatusError(Exception):
        """Giả lập anthropic.APIStatusError."""

    print("===== 1. classify_exception =====")
    samples = [
        RateLimitError("429 too many requests"),
        OperationalError("could not connect to server"),
        APIStatusError("500 internal error"),
        ValueError("một lỗi lập trình bình thường"),
        AskError(ErrorKind.DB_UNAVAILABLE, "đã phân loại từ trước"),
    ]
    for exc in samples:
        print(f"  {type(exc).__name__:<20} -> {classify_exception(exc).value}")

    print("\n===== 2. format_user_error =====")
    for kind in ErrorKind:
        print(format_user_error(AskError(kind, "chi tiết kỹ thuật ở đây")))
        print(f"   (exit code {EXIT_CODES.get(kind, 1)})\n")

    print("===== 3. validate_question =====")
    for raw in ["  Embedding là gì?  ", "", "   ", "ok", "x" * 600]:
        try:
            print(f"  {raw[:20]!r:<25} -> {validate_question(raw)!r}")
        except AskError as e:
            print(f"  {raw[:20]!r:<25} -> ❌ {e.kind.value} ({e.detail})")

    print("\n===== 4. retry_with_backoff =====")
    call_count = 0

    def flaky_call() -> str:
        """Lỗi 429 hai lần đầu, lần thứ 3 thành công."""
        global call_count
        call_count += 1
        if call_count < 3:
            raise RateLimitError("429 slow down")
        return "OK"

    started_at = time.perf_counter()
    print("  kết quả:", retry_with_backoff(flaky_call, attempts=3, base_delay=0.2))
    print(f"  gọi {call_count} lần, mất {time.perf_counter() - started_at:.1f}s")

    try:
        retry_with_backoff(lambda: (_ for _ in ()).throw(ValueError("lỗi code")), attempts=3)
    except ValueError as e:
        print(f"  ValueError -> nổ ngay, KHÔNG retry: {e}")

    # ✅ ĐẠT khi:
    #   1. classify_exception: RateLimitError->api_rate_limit · OperationalError->db_unavailable
    #      · APIStatusError->api_failed · ValueError->unknown · AskError giữ nguyên db_unavailable
    #   2. In đủ 8 thông điệp, MỌI thông điệp đều có dòng "→" nói rõ phải làm gì
    #   3. validate_question: '' / '   ' / 'ok' đều ném EMPTY_QUESTION; chuỗi 600 ký tự cũng ném
    #   4. flaky_call gọi đúng 3 lần, tổng thời gian ~0.6s (0.2 + 0.4)
    #   5. ValueError nổ NGAY, không in dòng "chờ ... rồi thử lại" nào
