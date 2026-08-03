"""
errors.py — Danh mục lỗi của app RAG + hành vi rõ ràng cho từng loại

Đây là file NỀN của cả ngày hôm nay. Mục tiêu ngày là "người khác clone về chạy được",
mà thứ giết trải nghiệm đó nhanh nhất không phải thiếu tính năng — là **traceback 40 dòng
đập vào mặt người dùng khi họ chỉ quên bật Docker**.

    tuần 4:  lỗi gì cũng -> raise SystemExit(chuỗi tự chế tại chỗ)
             (xem week4/ragLite/rag_qa.py::retrieve — SystemExit nằm SÂU trong hàm thư viện)
    tuần 5:  mọi lỗi -> AskError(kind, detail) --bay lên main()--> 1 khối in duy nhất
                                                    ▲
                                              chỉ main() mới được quyết định in gì và thoát

Vì sao KHÔNG raise SystemExit trong hàm sâu (bug thiết kế của tuần 4):
    Hàm nào gọi sys.exit là hàm đó tự quyết định số phận cả chương trình. Sau này muốn
    dùng lại retrieve() trong Streamlit / FastAPI / test thì nó giết luôn server.
    Luật: hàm thư viện RAISE, chỉ entry point (main) mới CATCH và EXIT.

Danh mục lỗi (task 09:00–11:00 — mỗi loại 1 hành vi, không loại nào rơi vào 'unknown'):

    kind                | khi nào                        | hành vi mong muốn
    --------------------|--------------------------------|---------------------------
    MISSING_CONFIG      | thiếu ANTHROPIC_API_KEY/DB_URL | in cách sửa, exit 3
    EMPTY_QUESTION      | người dùng gõ "" hoặc "   "    | in cách dùng, exit 2
    DB_UNAVAILABLE      | container Postgres chưa Up     | gợi ý docker compose up -d
    NO_RESULTS          | retrieve trả 0 chunk           | "chưa ingest tài liệu?" exit 0
    LOW_CONFIDENCE      | top-1 similarity < ngưỡng      | "không tìm thấy" — KHÔNG gọi LLM
    API_RATE_LIMIT      | 429 từ Anthropic               | retry backoff, hết thì báo
    API_FAILED          | 4xx/5xx khác, timeout          | in mã lỗi gốc, exit 5
    UNKNOWN             | còn lại                        | in traceback THẬT (đừng nuốt)

📖 Mỗi hàm chia 2 khối: PHẦN 1 GIẢI THÍCH (đọc) · PHẦN 2 CODE CẦN VIẾT (gõ).

Self-test bằng exception giả — KHÔNG cần DB, KHÔNG cần API key, KHÔNG cần model:
    python errors.py
"""

from __future__ import annotations

import sys
import time
from enum import Enum

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Kiểu dữ liệu — VIẾT SẴN, chỉ cần đọc hiểu
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# TODO 1 — biến exception của thư viện thành ErrorKind của mình
# ---------------------------------------------------------------------------
def classify_exception(exc: BaseException) -> ErrorKind:
    """Exception bất kỳ (psycopg / anthropic / httpx) -> ErrorKind.

    Ví dụ: psycopg.OperationalError("could not connect") -> ErrorKind.DB_UNAVAILABLE
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Vì sao nhận diện bằng TÊN CLASS (chuỗi) chứ không `isinstance(exc, anthropic.RateLimitError)`?
    #   Dùng isinstance thì file này buộc phải `import anthropic, psycopg` ngay đầu module.
    #   Hậu quả: self-test cũng phải cài đủ 2 thư viện, khởi động chậm, và nếu ai đó chỉ
    #   muốn dùng errors.py trong một script nhỏ thì phải kéo theo cả cụm dependency.
    #   Đổi lại phải chấp nhận: so tên chuỗi thì đổi tên class ở phiên bản mới = hỏng ngầm.
    #   -> vì vậy PHẢI có nhánh UNKNOWN in ra tên class thật, để lỗi lộ ra chứ không trôi.
    #
    # ▸ Bẫy 1 — thứ tự kiểm tra. RateLimitError của anthropic là CON của APIStatusError.
    #   Nếu bắt "APIStatusError" trước thì 429 bị gán nhầm API_FAILED -> mất luôn retry.
    #   Luật chung: kiểm tra từ CỤ THỂ nhất tới TỔNG QUÁT nhất.
    #
    # ▸ Bẫy 2 — TimeoutError / httpx.ConnectError xuất hiện ở CẢ hai phía (DB và API).
    #   Chỉ nhìn tên exception thì không phân biệt được. Nên: chỗ gọi DB tự bọc và ném
    #   AskError(DB_UNAVAILABLE) sẵn, chỗ gọi API ném API_FAILED. Hàm này chỉ là lưới hứng
    #   cuối cùng — đừng cố làm nó thông minh hơn mức nó biết.
    #
    # ▸ Bẫy 3 — AskError cũng là Exception. Gọi classify_exception lên một AskError đã
    #   phân loại rồi sẽ ra UNKNOWN, ghi đè mất kind gốc. Phải trả về exc.kind ngay dòng đầu.
    #
    # ▸ Nhắc cú pháp: `type(exc).__name__` cho tên class ('RateLimitError'), không phải
    #   `exc.__name__` (exception là INSTANCE, không có thuộc tính đó -> AttributeError).

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — trả sớm nếu đã phân loại rồi (chặn Bẫy 3):
    #     if isinstance(exc, AskError):
    #         return exc.kind
    #
    # Bước 2 — lấy tên class và message thường hoá:
    #     name = type(exc).__name__
    #     message = str(exc).lower()
    #
    # Bước 3 — bảng ánh xạ, CỤ THỂ trước TỔNG QUÁT:
    #     if name in ("RateLimitError", "OverloadedError"):
    #         return ErrorKind.API_RATE_LIMIT
    #     if name in ("OperationalError", "InterfaceError"):
    #         return ErrorKind.DB_UNAVAILABLE
    #     if name.startswith(("API", "Anthropic")) or "anthropic" in message:
    #         return ErrorKind.API_FAILED
    #
    # Bước 4 — lưới hứng cuối: "429" đôi khi chỉ nằm trong message, không có class riêng
    #     if "429" in message or "rate limit" in message:
    #         return ErrorKind.API_RATE_LIMIT
    #     return ErrorKind.UNKNOWN
    #
    # ✅ Kiểm tra nhanh: chạy `python errors.py`, khối self-test phải in đúng kind cho cả
    #    5 exception giả, và AskError(DB_UNAVAILABLE) phải giữ nguyên kind (không thành unknown).
    if isinstance(exc, AskError):
        return exc.kind

    name = type(exc).__name__
    message = str(exc).lower()

    if name in ("RateLimitError", "OverloadedError"):
        return ErrorKind.API_RATE_LIMIT
    if name in ("OperationalError", "InterfaceError"):
        return ErrorKind.DB_UNAVAILABLE
    if name.startswith(("API", "Anthropic")) or "anthropic" in message:
        return ErrorKind.API_FAILED

    if "429" in message or "rate limit" in message:
        return ErrorKind.API_RATE_LIMIT
    return ErrorKind.UNKNOWN


# ---------------------------------------------------------------------------
# TODO 2 — mỗi loại lỗi một thông điệp NGƯỜI DÙNG SỬA ĐƯỢC
# ---------------------------------------------------------------------------
def format_user_error(error: AskError) -> str:
    """AskError -> khối text in ra màn hình: chuyện gì + phải làm gì.

    Ví dụ: AskError(DB_UNAVAILABLE, "connection refused") ->
        ❌ Không kết nối được cơ sở dữ liệu.
           → Chạy: docker compose up -d   (trong week4/pgvector)
           chi tiết: connection refused
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Một thông báo lỗi tốt có đúng 3 phần: CHUYỆN GÌ (người thường hiểu được) ·
    #   LÀM GÌ TIẾP (câu lệnh copy-paste được) · CHI TIẾT KỸ THUẬT (cho mình debug).
    #   Thiếu phần 2 là biến người dùng thành thám tử. "Error: connection refused" thiếu
    #   cả 1 lẫn 2.
    #
    # ▸ Vì sao là 1 bảng dict chứ không phải chuỗi if/elif?
    #   Bảng đọc được như một bản đặc tả — nhìn phát biết còn thiếu loại lỗi nào chưa xử lý.
    #   Chuỗi if/elif dài 40 dòng thì không ai soát nổi, và rất dễ quên 1 nhánh.
    #
    # ▸ Bẫy — quên một ErrorKind trong bảng thì `MESSAGES[error.kind]` nổ KeyError
    #   NGAY TRONG hàm xử lý lỗi. Lỗi khi đang báo lỗi là loại khó chịu nhất vì nó giấu mất
    #   nguyên nhân gốc. -> luôn dùng `.get(kind, fallback)`, đừng dùng `[kind]`.
    #
    # ▸ Nhắc cú pháp: `dict.get(key, default)` trả default thay vì nổ. Giá trị ở đây là
    #   tuple 2 phần tử nên unpack bằng `title, hint = MESSAGES.get(...)`.

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — bảng thông điệp. 2 dòng đầu làm mẫu, tự điền 6 dòng còn lại
    #          (đây chính là task "liệt kê lỗi PHẢI xử lý" của block sáng — điền đủ 8):
    #     MESSAGES: dict[ErrorKind, tuple[str, str]] = {
    #         ErrorKind.DB_UNAVAILABLE: (
    #             "Không kết nối được cơ sở dữ liệu.",
    #             "Chạy: docker compose up -d   (trong week4/pgvector), rồi thử lại.",
    #         ),
    #         ErrorKind.MISSING_CONFIG: (
    #             "Thiếu cấu hình bắt buộc.",
    #             "Copy .env.example thành .env rồi điền ANTHROPIC_API_KEY.",
    #         ),
    #         # ... EMPTY_QUESTION, NO_RESULTS, LOW_CONFIDENCE,
    #         #     API_RATE_LIMIT, API_FAILED, UNKNOWN
    #     }
    #
    # Bước 2 — tra bảng, có đường lui:
    #     title, hint = MESSAGES.get(
    #         error.kind, ("Lỗi không xác định.", "Xem chi tiết bên dưới và mở issue."))
    #
    # Bước 3 — ghép 3 phần, phần chi tiết chỉ in khi CÓ:
    #     lines = [f"❌ {title}", f"   → {hint}"]
    #     if error.detail:
    #         lines.append(f"   chi tiết: {error.detail}")
    #     return "\n".join(lines)
    #
    # ✅ Kiểm tra nhanh: đọc to từng thông điệp và tự hỏi "người chưa từng đọc code này có
    #    biết phải gõ gì tiếp không?". Nếu không -> phần hint còn kém.
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
    title, hint = MESSAGES.get(
        error.kind, ("Lỗi không xác định.", "Xem chi tiết bên dưới và mở issue.")    
    )

    lines = [f"❌ {title}", f"   → {hint}"]
    if error.detail:
        lines.append(f"   chi tiết: {error.detail}")
        
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TODO 3 — chặn câu hỏi rác NGAY ĐẦU VÀO
# ---------------------------------------------------------------------------
def validate_question(raw: str) -> str:
    """Kiểm tra + làm sạch câu hỏi. Trả câu hỏi sạch, hoặc ném AskError(EMPTY_QUESTION).

    Ví dụ: '  Embedding là gì?  ' -> 'Embedding là gì?'   |   '   ' -> AskError
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Vì sao chặn ở ĐẦU VÀO chứ không để pipeline tự chết?
    #   Câu hỏi rỗng vẫn embed được — model trả về một vector hợp lệ. pgvector vẫn trả top-k.
    #   Claude vẫn trả lời. Không có dòng lỗi nào. Người dùng nhận về một câu trả lời bịa
    #   hoàn toàn và mình tốn tiền token cho nó. Đây là **bug im lặng có tính phí**.
    #
    # ▸ Bẫy 1 — `if not raw` KHÔNG bắt được "   " (chuỗi space là truthy). Phải strip TRƯỚC
    #   rồi mới kiểm tra. Với argv thì " ".join(args) rất hay tạo ra chuỗi toàn space.
    #
    # ▸ Bẫy 2 — chặn trên quá dài cũng cần thiết: dán nhầm cả file vào -> 1 request vài chục
    #   nghìn token. Không có chặn trên thì lỗi này chỉ hiện ra ở hoá đơn cuối tháng.
    #
    # ▸ Bẫy 3 — đừng "sửa hộ" câu hỏi (tự thêm dấu ?, tự viết hoa). Sửa im lặng input của
    #   người dùng là thói quen xấu: khi kết quả sai họ không hiểu vì sao. strip() là đủ.
    #
    # ▸ Nhắc cú pháp: raise AskError(...) — không return None rồi để chỗ gọi tự đoán.
    #   Hàm trả về str thì nó phải LUÔN trả str hoặc ném; đừng trả `str | None`.

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — làm sạch trước, kiểm tra sau:
    #     question = (raw or "").strip()
    #
    # Bước 2 — rỗng / quá ngắn:
    #     if len(question) < MIN_QUESTION_CHARS:
    #         raise AskError(ErrorKind.EMPTY_QUESTION,
    #                        f"câu hỏi chỉ có {len(question)} ký tự")
    #
    # Bước 3 — quá dài:
    #     if len(question) > MAX_QUESTION_CHARS:
    #         raise AskError(ErrorKind.EMPTY_QUESTION,
    #                        f"câu hỏi dài {len(question)} ký tự, tối đa {MAX_QUESTION_CHARS}")
    #
    # Bước 4 — trả về:
    #     return question
    #
    # ✅ Kiểm tra nhanh: `python ask.py ""` và `python ask.py "   "` phải ra CÙNG một thông
    #    điệp gọn gàng, không có chữ 'Traceback' nào trên màn hình.
    question = (raw or "").strip()

    if len(question) < MIN_QUESTION_CHARS:
        raise AskError(ErrorKind.EMPTY_QUESTION,
                       f"câu hỏi chỉ có {len(question)} ký tự")

    if len(question) > MAX_QUESTION_CHARS:
        raise AskError(ErrorKind.EMPTY_QUESTION,
                       f"câu hỏi dài {len(question)} ký tự, tối đa {MAX_QUESTION_CHARS}")

    return question


# ---------------------------------------------------------------------------
# TODO 4 — rate limit: chờ rồi thử lại, nhưng chỉ với lỗi đáng chờ
# ---------------------------------------------------------------------------
def retry_with_backoff(call, attempts: int = 3, base_delay: float = 1.0):
    """Gọi `call()` tối đa `attempts` lần, giãn cách gấp đôi mỗi lần, chỉ retry lỗi tạm thời.

    Ví dụ: lần 1 lỗi 429 -> chờ 1s -> lần 2 lỗi 429 -> chờ 2s -> lần 3 OK -> trả kết quả.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 1 — GIẢI THÍCH  (đọc, không gõ)
    # ═══════════════════════════════════════════════════════════════════════
    # ▸ Vì sao chờ GẤP ĐÔI (1s, 2s, 4s) chứ không chờ đều 1s?
    #   429 nghĩa là phía server đang quá tải hoặc mình đang gọi quá nhanh. Retry đều đặn
    #   là tiếp tục dội vào đúng lúc nó yếu nhất. Giãn cách tăng dần cho hệ thống thời gian
    #   hồi phục — đây là hành vi "công dân tốt" mà mọi SDK production đều làm.
    #
    # ▸ Bẫy CHẾT NGƯỜI — retry MỌI exception. Câu hỏi rỗng, sai API key, typo tên biến:
    #   thử lại 3 lần vẫn sai y hệt, chỉ tổ làm người dùng chờ 7 giây rồi nhận cùng lỗi.
    #   Retry chỉ đúng với lỗi TẠM THỜI (xem RETRYABLE_KINDS) — đó là lý do phải
    #   classify_exception TRƯỚC khi quyết định retry.
    #
    # ▸ Bẫy 2 — hết lượt mà `return` hoặc `pass` thì hàm trả None, chỗ gọi nhận None và nổ
    #   ở một chỗ hoàn toàn khác (AttributeError: 'NoneType'...), xa hàng chục dòng so với
    #   nguyên nhân thật. Hết lượt thì phải NÉM lại lỗi cuối cùng.
    #
    # ▸ Nhắc cú pháp: `call` ở đây là một HÀM chưa gọi (callable). Chỗ gọi truyền vào
    #   `lambda: client.messages.create(...)` — có `lambda:` thì lời gọi mới bị hoãn lại
    #   để retry_with_backoff chủ động gọi. Quên `lambda:` là đã gọi mất rồi, retry vô nghĩa.
    #
    # ▸ Thực tế production còn thêm "jitter" (cộng ngẫu nhiên 0–0.3s) để 100 client cùng
    #   lỗi không cùng retry đúng một thời điểm. Hôm nay chỉ 1 client, bỏ qua được — nhưng
    #   biết để nói khi phỏng vấn.

    # ═══════════════════════════════════════════════════════════════════════
    # PHẦN 2 — CODE CẦN VIẾT
    # ═══════════════════════════════════════════════════════════════════════
    # Bước 1 — vòng lặp có đánh số lần thử (bắt đầu từ 0 để tính 2**attempt cho tiện):
    #     last_error: BaseException | None = None
    #     for attempt in range(attempts):
    #         try:
    #             return call()
    #         except Exception as exc:
    #             last_error = exc
    #
    # Bước 2 — (vẫn trong except) không đáng retry thì ném NGAY, đừng chờ:
    #             kind = classify_exception(exc)
    #             if kind not in RETRYABLE_KINDS:
    #                 raise
    #
    # Bước 3 — (vẫn trong except) lần cuối rồi thì cũng ném luôn, đừng chờ vô ích:
    #             if attempt == attempts - 1:
    #                 break
    #             delay = base_delay * (2 ** attempt)      # 1s, 2s, 4s...
    #             print(f"   ⏳ rate limit, chờ {delay:.0f}s rồi thử lại "
    #                   f"({attempt + 2}/{attempts})...")
    #             time.sleep(delay)
    #
    # Bước 4 — ra khỏi vòng lặp = đã hết lượt, biến thành AskError có phân loại:
    #     raise AskError(ErrorKind.API_RATE_LIMIT,
    #                    f"thử {attempts} lần vẫn lỗi: {last_error}")
    #
    # ✅ Kiểm tra nhanh: self-test bên dưới có 1 hàm giả lỗi 429 hai lần rồi thành công.
    #    Phải thấy đúng 2 dòng "chờ ... rồi thử lại" và kết quả cuối là "OK".
    #    Đổi sang hàm ném ValueError -> phải nổ NGAY, không chờ giây nào.
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            last_error = exc
            kind = classify_exception(exc)
            if kind not in RETRYABLE_KINDS:
                raise

            if attempt == attempts - 1:
                break
            delay = base_delay * (2 ** attempt)
            print(f"   ⏳ rate limit, chờ {delay:.0f}s rồi thử lại "
                  f"({attempt + 2}/{attempts})...")
            time.sleep(delay)
    raise AskError(ErrorKind.API_RATE_LIMIT,
                   f"thử {attempt} lần vẫn lỗi: {last_error}")

if __name__ == "__main__":
    # ── Exception giả: KHÔNG cần cài anthropic/psycopg, KHÔNG cần mạng ──────────────
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
