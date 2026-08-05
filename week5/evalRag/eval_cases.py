"""
eval_cases.py — Bộ eval 10 câu: định nghĩa, nạp, và KIỂM TRA bộ dữ liệu trước khi chạy

    INPUT :  eval_cases.json
    OUTPUT:  list[EvalCase] đã được xác nhận là hợp lệ, hoặc nổ NGAY với lý do rõ ràng

Vì sao tách file dữ liệu ra khỏi file chạy:

    tuần 4:  week4/ragLite/questions.json  — 5 câu, 3 trường (q / expect / note)
                                             chấm bằng đúng 1 luật: "có chứa NO_ANSWER không"
    tuần 5:  eval_cases.json              — 10 câu, có difficulty + expect_keywords
                                             + expect_source  ──► grader.py chấm 4 luật
                       ▲                                              ▲
                       └── thêm 3 trường này là toàn bộ                └── xem grader.py
                           lý do bộ eval hôm nay mạnh hơn

⚠️ Đổi tên trường: tuần 4 dùng `q`, hôm nay dùng `question`. Cố ý KHÔNG giữ tên cũ vì `q`
   đứng cạnh `question` trong cùng repo là câu đố. Hệ quả: KHÔNG nạp thẳng questions.json
   của tuần 4 bằng loader này được — validate_cases sẽ bắt được và báo thiếu trường.

Nguyên tắc của ngày hôm nay (chép từ cột Ghi chú trong xlsx):
    "Viết tiêu chí chấm TRƯỚC khi chạy để tránh chấm theo cảm tính sau khi đã thấy kết quả."
    Đó chính là lý do expect_keywords nằm trong FILE DỮ LIỆU, không nằm trong đầu mình.
    Nhìn kết quả rồi mới nghĩ ra tiêu chí thì bộ eval nào cũng PASS 10/10.


Self-test bằng dữ liệu giả — không cần DB, không cần API, không cần model:
    python eval_cases.py
Kiểm tra file thật:
    python eval_cases.py eval_cases.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CASES_PATH = os.path.join(HERE, "eval_cases.json")

# Số câu bắt buộc theo plan: 7 câu CÓ trong tài liệu + 3 câu KHÔNG có.
# Đặt thành hằng để validate_cases chặn được lúc mình viết thiếu — 10 câu mà 9 câu
# answerable thì tỉ lệ PASS đẹp một cách vô nghĩa.
REQUIRED_ANSWERABLE = 7
REQUIRED_NO_ANSWER = 3

VALID_EXPECT = ("answerable", "no_answer")
VALID_DIFFICULTY = ("easy", "medium", "hard")


@dataclass
class EvalCase:
    """1 ca kiểm thử. Viết sẵn — chỉ cần đọc hiểu ý nghĩa từng trường.

    id              : mã ngắn ('A1', 'N2') để chỉ đích danh khi bàn về 1 ca cụ thể
    question        : câu hỏi tiếng Việt, y như người dùng sẽ gõ
    expect          : 'answerable' (tài liệu CÓ câu trả lời) | 'no_answer' (KHÔNG có)
    difficulty      : 'easy' | 'medium' | 'hard'
                      hard = phải GHÉP thông tin từ ≥2 chỗ mới trả lời được, không chép 1 đoạn
    expect_keywords : các từ/cụm BẮT BUỘC xuất hiện trong câu trả lời đúng
                      -> đây là "đáp án" ở dạng máy chấm được. Với no_answer thì để rỗng.
    expect_source   : tên file (hoặc một phần tên) mà nguồn đúng phải trỏ tới. None = không xét
    note            : vì sao chọn câu này — để 3 tháng sau đọc lại còn hiểu ý đồ
    """

    id: str
    question: str
    expect: str
    difficulty: str = "medium"
    expect_keywords: list[str] = field(default_factory=list)
    expect_source: str | None = None
    note: str = ""

    @property
    def is_trap(self) -> bool:
        """Ca bẫy = ca mà câu trả lời ĐÚNG là lời từ chối."""
        return self.expect == "no_answer"


def load_cases(path: str = DEFAULT_CASES_PATH) -> tuple[list[EvalCase], str]:
    """Đọc eval_cases.json -> (list[EvalCase], corpus_note).

    Ví dụ: load_cases() -> ([EvalCase(id='A1', ...), ...], 'Bộ 8 PDF về ...')

    `EvalCase(**raw)` bung dict thành tham số theo TÊN. Thừa một khoá lạ trong JSON là
    TypeError ngay tại đây — cố ý, vì gõ nhầm `expected_keywords` (thừa chữ 'ed') mà
    im lặng bỏ qua thì ca đó vĩnh viễn không được chấm keyword và mình không biết.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    cases = [EvalCase(**item) for item in raw["cases"]]
    return cases, raw.get("corpus_note", "")


def validate_cases(cases: list[EvalCase]) -> list[str]:
    """Trả DANH SÁCH lỗi của bộ eval. List rỗng = bộ dữ liệu hợp lệ.

    Ví dụ: [EvalCase(id='A1', question='', ...)] -> ["A1: question rỗng"]
    """

    from collections import Counter
    errors: list[str] = []
    id_counter = Counter(case.id for case in cases)
    for case_id, count in id_counter.items():
        if count > 1:
            errors.append(f"id '{case_id}' xuất hiện {count} lần")

    for case in cases:
        if not case.question.strip():
            errors.append(f"{case.id}: question rỗng — chưa điền câu hỏi thật")
        if case.expect not in VALID_EXPECT:
            errors.append(f"{case.id}: expect='{case.expect}' không hợp lệ")
        if case.difficulty not in VALID_DIFFICULTY:
            errors.append(f"{case.id}: difficulty='{case.difficulty}' không hợp lệ")
        if case.expect == "answerable" and not case.expect_keywords:
            errors.append(f"{case.id}: answerable nhưng expect_keywords rỗng")
        if case.is_trap and case.expect_keywords:
            errors.append(f"{case.id}: no_answer thì không được có expect_keywords")

    n_answerable = sum(1 for c in cases if c.expect == "answerable")
    n_trap = sum(1 for c in cases if c.is_trap)

    if n_answerable != REQUIRED_ANSWERABLE:
        errors.append(f"cần {REQUIRED_ANSWERABLE} ca answerable, đang có {n_answerable}")

    if n_trap != REQUIRED_NO_ANSWER:
        errors.append(f"cần {REQUIRED_NO_ANSWER} ca no_answer, đang có {n_trap}")

    return errors


def summarize_dataset(cases: list[EvalCase]) -> str:
    """Mô tả bộ eval bằng 1 khối text — để dán thẳng vào README portfolio.

    Mong muốn:
        Bộ eval: 10 ca (7 answerable / 3 no_answer)
          theo độ khó: easy 3 · medium 3 · hard 1
          ca hard: A6, ...
    """

    from collections import Counter

    expect_counter = Counter(case.expect for case in cases)
    difficulty_counter = Counter(c.difficulty for c in cases if c.expect == "answerable")

    lines = [
        f"Bộ eval: {len(cases)} ca "
        f"({expect_counter['answerable']} answerable / {expect_counter['no_answer']} no_answer)",
        "  theo độ khó (chỉ ca answerable): "
        + " · ".join(f"{d} {difficulty_counter[d]}" for d in VALID_DIFFICULTY),
    ]

    hard_ids = [c.id for c in cases if c.difficulty == "hard"]
    if hard_ids:
        lines.append("  ca hard (phải ghép ≥2 nguồn): " + ", ".join(hard_ids))

    return "\n".join(lines)


if __name__ == "__main__":
    # ---- Chế độ 1: kiểm tra FILE THẬT (python eval_cases.py eval_cases.json) --------
    if len(sys.argv) > 1:
        cases, corpus_note = load_cases(sys.argv[1])
        errors = validate_cases(cases)
        print(f"📂 {sys.argv[1]}  ·  {len(cases)} ca")
        print(f"📚 corpus: {corpus_note or '(chưa điền corpus_note)'}\n")
        if errors:
            print(f"❌ {len(errors)} lỗi — SỬA HẾT rồi mới chạy run_eval.py:")
            for e in errors:
                print(f"   - {e}")
            raise SystemExit(1)
        print("✅ Bộ eval hợp lệ.\n")
        print(summarize_dataset(cases))
        raise SystemExit(0)

    # ---- Chế độ 2: self-test bằng 2 bộ giả — không đọc file, không cần gì cả --------
    print("===== 1. Bộ HỎNG: phải bắt được đủ mọi lỗi =====")
    broken = [
        EvalCase("A1", "", "answerable", "easy", ["x"]),          # question rỗng
        EvalCase("A1", "Câu hỏi trùng id?", "answerable", "easy", ["x"]),  # id trùng
        EvalCase("A2", "Chunking là gì?", "answerable", "easy", []),      # thiếu keywords
        EvalCase("N1", "Giá vàng hôm nay?", "no_answer", "easy", ["vàng"]),  # trap có keywords
        EvalCase("A3", "Câu này sao?", "answerable", "cực khó", ["y"]),   # difficulty lạ
    ]
    for err in validate_cases(broken):
        print(f"   - {err}")

    print("\n===== 2. Bộ TỐT: phải ra 0 lỗi =====")
    good = (
        [EvalCase(f"A{i}", f"Câu hỏi số {i}?", "answerable",
                  "hard" if i == 7 else ("easy" if i <= 3 else "medium"),
                  [f"tu_khoa_{i}"])
         for i in range(1, 8)]
        + [EvalCase(f"N{i}", f"Câu bẫy số {i}?", "no_answer") for i in range(1, 4)]
    )
    errors = validate_cases(good)
    print(f"   {len(errors)} lỗi")

    print("\n===== 3. summarize_dataset =====")
    print(summarize_dataset(good))

    # ✅ ĐẠT khi:
    #   1. Bộ HỎNG in ra ĐỦ 5 lỗi (rỗng · trùng id · thiếu keywords · trap có keywords ·
    #      difficulty lạ) — thiếu lỗi nào thì đúng điều kiện đó chưa được kiểm
    #   2. Bộ HỎNG KHÔNG được báo lỗi tỉ lệ 7/3 nhầm chỗ — nó có 4 answerable/1 trap nên
    #      PHẢI có thêm 2 lỗi tỉ lệ nữa, tổng 7 dòng
    #   3. Bộ TỐT in đúng "0 lỗi"
    #   4. summarize_dataset in "10 ca (7 answerable / 3 no_answer)" và
    #      "easy 3 · medium 3 · hard 1" — cộng lại đúng 7
