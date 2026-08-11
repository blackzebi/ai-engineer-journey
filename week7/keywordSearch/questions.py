"""
questions.py — Bộ 8 câu hỏi đo của TUẦN 6: nạp + kiểm tra trước khi dùng

    INPUT :  questions.json
    OUTPUT:  list[RetrievalCase] đã xác nhận hợp lệ, hoặc nổ NGAY với lý do rõ ràng

Vì sao KHÔNG dùng lại week5/evalRag/eval_cases.py mà viết bộ mới — ĐƠN VỊ ĐO KHÁC NHAU:

    tuần 5 (EvalCase)          đo CÂU TRẢ LỜI của LLM
        expect_keywords, citation, refusal  ->  cần gọi API, tốn tiền, có ngẫu nhiên
    tuần 7 (RetrievalCase)     đo KẾT QUẢ TRUY XUẤT
        expect_source + thứ hạng            ->  không gọi LLM, chạy 100 lần ra 100 kết quả
                                                y hệt nhau
                ▲
                └── đó là lý do bộ này chạy được mỗi ngày trong tuần mà không tốn đồng nào,
                    và là lý do nó phải TÁCH khỏi eval_cases.json chứ không nhét thêm cột.

⚠️ Còn một lý do rất cụ thể nữa: `EvalCase(**raw)` ở tuần 5 sẽ ném TypeError khi JSON có
   khoá lạ (cố ý — xem docstring load_cases). Bộ hôm nay có thêm `query_kind` và
   `expect_winner`, nên nạp bằng loader tuần 5 là hỏng ngay. Đừng "sửa" bằng cách nới lỏng
   loader tuần 5 — cái chặt chẽ đó đang bảo vệ bộ eval khỏi lỗi gõ nhầm tên trường.

⚠️ Và một cái bẫy đã nằm sẵn trong repo: `validate_cases()` của tuần 5 kiểm tỉ lệ bằng hai
   HẰNG CỨNG REQUIRED_ANSWERABLE = 7 / REQUIRED_NO_ANSWER = 3. Bộ hôm nay là 6/2, nên nếu
   lỡ chạy qua validator tuần 5 sẽ nhận 2 lỗi tỉ lệ HOÀN TOÀN SAI. Bài học thiết kế: tham
   số hoá cái gì có thể đổi giữa các lần dùng — xem REQUIRED_* bên dưới nhận qua tham số.


Self-test bằng bộ giả — không cần file, không cần DB:
    python questions.py
Kiểm tra file thật:
    python questions.py questions.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS_PATH = os.path.join(HERE, "questions.json")

REQUIRED_ANSWERABLE = 6
REQUIRED_NO_ANSWER = 2

VALID_EXPECT = ("answerable", "no_answer")
VALID_QUERY_KIND = ("keyword", "semantic", "mixed")
VALID_WINNER = ("keyword", "vector", "either", "none")

# Tối thiểu mỗi loại query_kind phải có bao nhiêu ca answerable. Có 2 câu keyword và 2 câu
# semantic thì bảng so sánh cuối ngày mới nói được điều gì; 1 câu mỗi loại thì thắng-thua
# chỉ là may rủi của đúng một câu hỏi.
MIN_PER_KIND = {"keyword": 2, "semantic": 2, "mixed": 1}


@dataclass
class RetrievalCase:
    """1 câu hỏi trong bộ đo truy xuất.

    id            : mã ngắn ('K1' keyword, 'S1' semantic, 'M1' mixed, 'N1' no-answer)
    question      : câu hỏi tiếng Việt, y như người dùng sẽ gõ
    expect        : 'answerable' | 'no_answer'
    query_kind    : 'keyword' | 'semantic' | 'mixed' — loại truy vấn, quyết định kỳ vọng
    expect_source : một phần tên file chứa câu trả lời. None với ca no_answer
    expect_winner : nhánh mình DỰ ĐOÁN sẽ thắng, ghi TRƯỚC khi chạy
                    -> đây là cách tự chấm hiểu biết của mình, không phải chấm hệ thống.
                       Dự đoán sai mà giải thích được VÌ SAO sai là ngày học tốt nhất.
    note          : vì sao chọn câu này
    """

    id: str
    question: str
    expect: str
    query_kind: str = "mixed"
    expect_source: str | None = None
    expect_winner: str = "either"
    note: str = ""

    @property
    def is_trap(self) -> bool:
        """Ca bẫy = ca mà kết quả đúng là KHÔNG tìm thấy gì đáng tin."""
        return self.expect == "no_answer"


def load_questions(path: str = DEFAULT_QUESTIONS_PATH) -> tuple[list[RetrievalCase], str]:
    """Đọc questions.json -> (list[RetrievalCase], corpus_note).

    `RetrievalCase(**item)` bung dict thành tham số theo TÊN — khoá lạ trong JSON là
    TypeError ngay tại đây. Cố ý giữ nguyên hành vi chặt chẽ của tuần 5.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    cases = [RetrievalCase(**item) for item in raw["cases"]]
    return cases, raw.get("corpus_note", "")


def validate_questions(
    cases: list[RetrievalCase],
    required_answerable: int = REQUIRED_ANSWERABLE,
    required_no_answer: int = REQUIRED_NO_ANSWER,
) -> list[str]:
    """Trả DANH SÁCH lỗi của bộ câu hỏi. List rỗng = hợp lệ.

    Ví dụ: [RetrievalCase('K1', '', 'answerable')] -> ["K1: question rỗng", ...]
    """

    errors: list[str] = []
    for case_id, count in Counter(case.id for case in cases).items():
        if count > 1:
            errors.append(f"id '{case_id}' xuất hiện {count} lần")

    for case in cases:
        if not case.question.strip():
            errors.append(f"{case.id}: question rỗng - chưa điền câu hỏi thật")
        if case.expect not in VALID_EXPECT:
            errors.append(f"{case.id}: expect='{case.expect}' không hợp lệ")
        if case.query_kind not in VALID_QUERY_KIND:
            errors.append(f"{case.id}: query_kind='{case.query_kind}' không hợp lệ")
        if case.expect_winner not in VALID_WINNER:
            errors.append(f"{case.id}: expect_winner='{case.expect_winner}' không hợp lệ")
        if case.expect == "answerable" and not case.expect_source:
            errors.append(f"{case.id}: answerable nhưng thiếu expect_source — "
                          f"không có đáp án thì không chấm được nhánh nào đúng")
        if case.is_trap and case.expect_source:
            errors.append(f"{case.id}: no_answer thì không được có expect_source")
        if case.query_kind == "keyword" and case.expect_winner == "vector":
            errors.append(f"{case.id}: query_kind=keyword mà dự đoán vector thắng — "
                            f"kiểm tra lại, có phải copy nhầm không?")

    n_answerable = sum(1 for c in cases if c.expect == "answerable")
    n_trap = sum(1 for c in cases if c.is_trap)
    if n_answerable != required_answerable:
        errors.append(f"cần {required_answerable} ca answerable, đang có {n_answerable}")
    if n_trap != required_no_answer:
        errors.append(f"cần {required_no_answer} ca no_answer, đang có {n_trap}")

    kind_counter = Counter(c.query_kind for c in cases if c.expect == "answerable")
    for kind, minimum in MIN_PER_KIND.items():
        if kind_counter[kind] < minimum:
            errors.append(f"cần ít nhất {minimum} ca answerable loại '{kind}', "
                          f"đang có {kind_counter[kind]}")
            
    return errors


def summarize_questions(cases: list[RetrievalCase]) -> str:
    """Mô tả bộ câu hỏi bằng 1 khối text — dán thẳng vào note và README tuần 7.

    Mong muốn:
        Bộ đo truy xuất: 8 câu (6 answerable / 2 no_answer)
          theo loại truy vấn: keyword 2 · semantic 2 · mixed 2
          dự đoán trước khi chạy: keyword thắng 2 · vector thắng 2 · either 2
    """

    expect_counter = Counter(c.expect for c in cases)
    kind_counter = Counter(c.query_kind for c in cases if c.expect == "answerable")
    winner_counter = Counter(c.expect_winner for c in cases if c.expect == "answerable")

    lines = [
        f"Bộ đo truy xuất: {len(cases)} câu "
        f"({expect_counter['answerable']} answerable / {expect_counter['no_answer']} no_answer)",
        "  theo loại truy vấn: "
        + " · ".join(f"{kind} {kind_counter[kind]}" for kind in VALID_QUERY_KIND),
        "  dự đoán trước khi chạy: "
        + " · ".join(f"{w} {winner_counter[w]}" for w in VALID_WINNER if winner_counter[w]),
    ]

    return "\n".join(lines)

if __name__ == "__main__":
    # ---- Chế độ 1: kiểm tra FILE THẬT (python questions.py questions.json) ----------
    if len(sys.argv) > 1:
        cases, corpus_note = load_questions(sys.argv[1])
        errors = validate_questions(cases)
        print(f"📂 {sys.argv[1]}  ·  {len(cases)} câu")
        print(f"📚 corpus: {corpus_note or '(chưa điền corpus_note)'}\n")
        if errors:
            print(f"❌ {len(errors)} lỗi — SỬA HẾT rồi mới chạy compare_branches.py:")
            for e in errors:
                print(f"   - {e}")
            raise SystemExit(1)
        print("✅ Bộ câu hỏi hợp lệ.\n")
        print(summarize_questions(cases))
        raise SystemExit(0)

    # ---- Chế độ 2: self-test bằng 2 bộ giả — không đọc file, không cần gì cả --------
    print("===== 1. Bộ HỎNG: phải bắt được đủ mọi lỗi =====")
    broken = [
        RetrievalCase("K1", "", "answerable", "keyword", "a.pdf", "keyword"),   # question rỗng
        RetrievalCase("K1", "Trùng id?", "answerable", "keyword", "a.pdf", "keyword"),  # id trùng
        RetrievalCase("K2", "Mã E402?", "answerable", "keyword", None, "keyword"),  # thiếu source
        RetrievalCase("N1", "Câu bẫy?", "no_answer", "semantic", "b.pdf", "none"),  # trap có source
        RetrievalCase("S1", "Paraphrase?", "answerable", "ngữ nghĩa", "c.pdf", "vector"),  # kind lạ
        RetrievalCase("K3", "Mã E403?", "answerable", "keyword", "a.pdf", "vector"),  # mâu thuẫn
    ]
    for err in validate_questions(broken):
        print(f"   - {err}")

    print("\n===== 2. Bộ TỐT: phải ra 0 lỗi =====")
    good = [
        RetrievalCase("K1", "Mã lỗi E402 nghĩa là gì?", "answerable", "keyword", "a.pdf", "keyword"),
        RetrievalCase("K2", "Ai ký quyết định 15/QĐ?", "answerable", "keyword", "b.pdf", "keyword"),
        RetrievalCase("S1", "Làm sao cắt nhỏ tài liệu cho hợp lý?", "answerable", "semantic", "c.md", "vector"),
        RetrievalCase("S2", "Khi nào nên tách dịch vụ ra riêng?", "answerable", "semantic", "c.md", "vector"),
        RetrievalCase("M1", "HNSW xử lý dữ liệu mới thế nào?", "answerable", "mixed", "d.md", "either"),
        RetrievalCase("M2", "Quy trình duyệt ở bước cuối là gì?", "answerable", "mixed", "b.pdf", "either"),
        RetrievalCase("N1", "Mã lỗe E999 nghĩa là gì?", "no_answer", "keyword", None, "none"),
        RetrievalCase("N2", "Chính sách nghỉ phép ra sao?", "no_answer", "semantic", None, "none"),
    ]
    errors = validate_questions(good)
    print(f"   {len(errors)} lỗi")

    print("\n===== 3. summarize_questions =====")
    print(summarize_questions(good))

    # ✅ ĐẠT khi:
    #   1. Bộ HỎNG in ra ĐỦ 6 lỗi riêng của từng ca (rỗng · trùng id · thiếu expect_source ·
    #      trap có source · query_kind lạ · keyword-mà-đoán-vector)
    #   2. Bộ HỎNG in THÊM lỗi tỉ lệ (nó có 5 answerable / 1 trap, không phải 6/2) và lỗi
    #      thiếu ca 'mixed' -> tổng phải > 6 dòng
    #   3. Bộ TỐT in đúng "0 lỗi"
    #   4. summarize in "8 câu (6 answerable / 2 no_answer)" và "keyword 2 · semantic 2 ·
    #      mixed 2" — cộng lại đúng 6
