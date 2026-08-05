"""
grader.py — Chấm tự động 1 câu trả lời theo 4 luật, viết TRƯỚC khi nhìn kết quả

    INPUT :  EvalCase (kỳ vọng) + AnswerRecord (thực tế app trả về)
    OUTPUT:  CaseVerdict — PASS/FAIL + LÝ DO của từng luật con

Nâng cấp so với cách chấm của tuần 4:

    tuần 4 (week4/ragLite/rag_qa.py::run_eval):
        ok = (expect == "no_answer" and NO_ANSWER in answer) \
             or (expect == "answerable" and NO_ANSWER not in answer and sources)
        └─ 1 luật duy nhất. Model trả lời SAI HOÀN TOÀN nhưng không từ chối và có nguồn
           -> vẫn PASS. Tức là tuần 4 chỉ đo được "có từ chối đúng lúc không",
           KHÔNG đo được "trả lời có đúng không".

    hôm nay: 4 luật độc lập, mỗi luật trả (passed, detail)
        1. refusal   — có từ chối hay không, đúng như kỳ vọng chưa
        2. keywords  — câu trả lời có chứa các thuật ngữ bắt buộc (chỉ ca answerable)
        3. citation  — có [n] trong bài, và mọi [n] đều trỏ tới nguồn CÓ THẬT
        4. source    — nguồn số [1] có đúng file mong đợi (khi expect_source được khai)
                                   │
        CaseVerdict.passed = AND của các luật ÁP DỤNG cho ca đó

Giới hạn đã biết — phải ghi vào README, đừng giấu:
    Chấm bằng từ khoá là PROXY, không phải hiểu nghĩa. Nó bắt được "model không nhắc tới
    khái niệm cần nhắc", nhưng KHÔNG bắt được "model nhắc đúng từ mà lập luận sai".
    Vì vậy 10/10 PASS ở đây KHÔNG có nghĩa là hệ thống đúng — vẫn phải tự đọc bằng mắt.
    Bản chấm bằng LLM-as-judge là việc của tuần 7.


Self-test bằng câu trả lời GIẢ — không cần DB, không cần API key, không cần model:
    python grader.py
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "askCli"))

from eval_cases import EvalCase  # noqa: E402

# NO_ANSWER phải IMPORT, tuyệt đối không gõ lại chuỗi đó ở đây.
# Chuỗi này hiện đang tồn tại ở 4 chỗ trong repo (xem mục "PHẢI sửa" trong guide.md).
# Gõ lại là tạo bản sao thứ 5: sửa prompt một chữ -> grader vẫn so với chuỗi cũ ->
# MỌI ca no_answer FAIL cùng lúc và mình sẽ đi debug nhầm phía model.
from stream_answer import NO_ANSWER  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Regex bắt marker trích dẫn dạng [1], [2], [12]. Không dùng \[\d\] (1 chữ số) —
# k có thể > 9 khi thử --k 10, khi đó [10] bị bắt hụt thành [1] + '0'.
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass
class AnswerRecord:
    """Thứ mà app THỰC SỰ trả về cho 1 câu hỏi. Viết sẵn — chỉ đọc hiểu.

    Cố ý KHÔNG dùng thẳng RetrievedChunk của askCli/retriever.py: grader chỉ cần 3 thứ
    (text, danh sách nguồn, số chunk). Phụ thuộc vào dataclass 8 trường kia thì grader
    không self-test được mà không kéo theo cả tầng DB.

    answer   : toàn văn câu trả lời của model
    sources  : [(source_path, similarity), ...] theo đúng thứ tự [1], [2], [3]
    n_chunks : số chunk đưa vào prompt (0 nghĩa là đã chặn ở ngưỡng, KHÔNG gọi LLM)
    error    : chuỗi lỗi nếu ca này nổ giữa chừng; None nếu chạy trót lọt
    """

    answer: str
    sources: list[tuple[str, float]] = field(default_factory=list)
    n_chunks: int = 0
    error: str | None = None


@dataclass
class CheckResult:
    """Kết quả 1 luật con. `detail` luôn phải nói ĐƯỢC/HỎNG Ở ĐÂU, không chỉ True/False."""

    name: str
    passed: bool
    detail: str


@dataclass
class CaseVerdict:
    """Kết quả chấm trọn 1 ca."""

    case_id: str
    passed: bool
    checks: list[CheckResult]

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def reason(self) -> str:
        """1 dòng gọn để in vào bảng markdown."""
        failed = self.failed_checks()
        return "—" if not failed else "; ".join(f"{c.name}: {c.detail}" for c in failed)


def normalize_for_match(text: str) -> str:
    """Chuẩn hoá text trước khi so khớp: thường hoá + gom khoảng trắng. Viết sẵn.

    ⚠️ CỐ Ý KHÔNG bỏ dấu tiếng Việt. 'mã hoá' và 'ma hoa' là hai thứ khác nhau; bỏ dấu để
    "so cho dễ" sẽ khiến từ khoá khớp bừa và bộ eval báo PASS oan.
    """
    return " ".join(text.lower().split())


def check_refusal(case: EvalCase, record: AnswerRecord) -> CheckResult:
    """Model có từ chối đúng lúc không? Luật áp dụng cho CẢ hai loại ca.

    Ví dụ: case.expect='no_answer', answer chứa NO_ANSWER -> passed=True
           case.expect='answerable', answer chứa NO_ANSWER -> passed=False (từ chối oan)
    """

    answer_norm = normalize_for_match(record.answer)
    refusal_norm = normalize_for_match(NO_ANSWER).rstrip(".")
    refused = refusal_norm in answer_norm

    passed = (refused == case.is_trap)

    if passed:
        detail = "từ chối đúng lúc" if refused else "có trả lời, đúng kỳ vọng"
    elif case.is_trap:
        detail = "Bịa - đáng lẽ phải từ chối"
    else:
        detail = "Từ chối oan - tài liệu có câu trả lời"

    return CheckResult("refusal", passed, detail)


def check_keywords(case: EvalCase, record: AnswerRecord) -> CheckResult:
    """Câu trả lời có chứa ĐỦ các từ khoá bắt buộc không? Chỉ áp dụng cho ca answerable.

    Ví dụ: expect_keywords=['chunk', 'overlap'], answer nhắc cả hai -> passed=True
           answer chỉ nhắc 'chunk' -> passed=False, detail nói thiếu 'overlap'
    """

    if case.is_trap or not case.expect_keywords:
        return CheckResult("keywords", True, "không áp dụng")

    answer_norm = normalize_for_match(record.answer)

    missing = [kw for kw in case.expect_keywords
               if normalize_for_match(kw) not in answer_norm]

    passed = not missing
    detail = ("đủ " + str(len(case.expect_keywords)) + " từ khoá") if passed \
             else "thiếu: " + ", ".join(missing)

    return CheckResult("keywords", passed, detail)


def check_citations(case: EvalCase, record: AnswerRecord) -> CheckResult:
    """Câu trả lời có trích dẫn [n], và mọi [n] có trỏ tới nguồn CÓ THẬT không?

    Ví dụ: answer 'RAG gồm 2 pha [1][2].' với 3 nguồn -> passed=True
           answer '... [4].' với 3 nguồn -> passed=False (trích dẫn ma)
    """

    if case.is_trap:
        return CheckResult("citation", True, "không áp dụng")
    if not record.sources:
        return CheckResult("citation", False, "0 nguồn — retrieval không trả về gì")
    cited = {int(n) for n in CITATION_PATTERN.findall(record.answer)}
    if not cited:
        return CheckResult("citation", False, "không có marker [n] nào trong câu trả lời")

    n_sources = len(record.sources)
    invalid = sorted(n for n in cited if n < 1 or n > n_sources)
    passed = not invalid
    detail = (f"trích {sorted(cited)} / {n_sources} nguồn" if passed
              else f"trích dẫn ma {invalid} nhưng chỉ có {n_sources} nguồn")

    return CheckResult("citation", passed, detail)


def check_source(case: EvalCase, record: AnswerRecord) -> CheckResult:
    """Nguồn trả về có đúng file mong đợi không? Chỉ chạy khi case.expect_source được khai.

    Ví dụ: expect_source='bao-cao-q3', sources[0]='docs/bao-cao-q3.pdf' -> passed=True
    """

    if case.is_trap or not case.expect_source:
        return CheckResult("source", True, "Không áp dụng")

    wanted = normalize_for_match(case.expect_source)
    matched_rank = None
    for rank, (source_path, _similarity) in enumerate(record.sources, 1):
        if wanted in normalize_for_match(source_path):
            matched_rank = rank
            break

    passed = matched_rank is not None
    detail = (f"khớp ở nguồn [{matched_rank}]" if passed
              else f"không thấy '{case.expect_source}' trong "
                   + ", ".join(s for s, _ in record.sources))

    return CheckResult("source", passed, detail)


def grade_case(case: EvalCase, record: AnswerRecord) -> CaseVerdict:
    """Chạy đủ 4 luật cho 1 ca, gộp thành 1 verdict.

    Ví dụ: grade_case(case_A1, record) -> CaseVerdict('A1', True, [4 CheckResult])
    """

    if record.error:
        return CaseVerdict(case.id, False,
                          [CheckResult("run", False, f"lỗi khi chạy: {record.error}")])

    checks = [
        check_refusal(case, record),
        check_keywords(case, record),
        check_citations(case, record),
        check_source(case, record)
    ]

    return CaseVerdict(case.id, all(c.passed for c in checks), checks)


if __name__ == "__main__":
    # Câu trả lời GIẢ mô phỏng đúng các kiểu hỏng hay gặp — không cần API, chạy tức thì.
    case_answerable = EvalCase(
        "A1", "Vì sao phải chunk tài liệu trước khi embed?", "answerable", "easy",
        expect_keywords=["chunk", "overlap"], expect_source="rag.md",
    )
    case_trap = EvalCase("N1", "Giá cổ phiếu Apple hôm nay?", "no_answer", "easy")

    scenarios = [
        ("ca tốt — đủ mọi thứ", case_answerable, AnswerRecord(
            answer="Chia nhỏ thành chunk 800 ký tự với overlap 100 [1][2].",
            sources=[("notes/rag.md", 0.62), ("docs/bao-cao.pdf", 0.41)], n_chunks=2)),
        ("thiếu 1 từ khoá", case_answerable, AnswerRecord(
            answer="Tài liệu dài nên phải cắt thành từng chunk nhỏ [1].",
            sources=[("notes/rag.md", 0.62)], n_chunks=1)),
        ("trích dẫn ma [4]", case_answerable, AnswerRecord(
            answer="Chunk 800 ký tự, overlap 100 [4].",
            sources=[("notes/rag.md", 0.62), ("docs/x.pdf", 0.40)], n_chunks=2)),
        ("từ chối OAN", case_answerable, AnswerRecord(
            answer=NO_ANSWER, sources=[("notes/rag.md", 0.62)], n_chunks=1)),
        ("retrieval sai file", case_answerable, AnswerRecord(
            answer="Chunk 800 ký tự, overlap 100 [1].",
            sources=[("docs/khac.pdf", 0.55)], n_chunks=1)),
        ("bẫy — từ chối đúng (thiếu dấu chấm)", case_trap, AnswerRecord(
            answer="Tôi không tìm thấy thông tin này trong tài liệu", sources=[], n_chunks=0)),
        ("bẫy — BỊA", case_trap, AnswerRecord(
            answer="Giá cổ phiếu Apple hôm nay là 214 USD [1].",
            sources=[("docs/x.pdf", 0.36)], n_chunks=1)),
        ("nổ giữa chừng", case_answerable, AnswerRecord(
            answer="", error="RateLimitError: 429")),
    ]

    print("===== Chấm 8 kịch bản giả =====\n")
    for label, case, record in scenarios:
        verdict = grade_case(case, record)
        mark = "✅ PASS" if verdict.passed else "❌ FAIL"
        print(f"{mark}  {label}")
        for check in verdict.checks:
            flag = "·" if check.passed else "✗"
            print(f"        {flag} {check.name:<9} {check.detail}")
        print()

    # ✅ ĐẠT khi:
    #   1. Đúng 2 kịch bản PASS: "ca tốt" và "bẫy — từ chối đúng (thiếu dấu chấm)"
    #   2. "thiếu 1 từ khoá" -> chỉ luật keywords FAIL, detail ghi "thiếu: overlap"
    #   3. "trích dẫn ma [4]" -> chỉ luật citation FAIL, detail có chữ "ma"
    #   4. "từ chối OAN" -> luật refusal FAIL với chữ "TỪ CHỐI OAN"; "bẫy — BỊA" -> chữ "BỊA"
    #   5. "retrieval sai file" -> keywords PASS nhưng source FAIL (đúng tinh thần: hai tầng
    #      khác nhau, model nói đúng nhờ kiến thức sẵn có mà retrieval lấy nhầm file)
    #   6. "nổ giữa chừng" -> đúng 1 check tên 'run', KHÔNG có 4 check kia
