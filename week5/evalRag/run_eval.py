"""
run_eval.py — Chạy 10 ca eval end-to-end qua ĐÚNG pipeline thật, xuất bảng markdown

    INPUT :  eval_cases.json + DB đã ingest bộ tài liệu thật + ANTHROPIC_API_KEY
    OUTPUT:  eval_results.md (bảng dán thẳng vào README) + eval_results.json (cho tuần 7)
             + tổng chi phí thật của cả lượt chạy

Luồng, và chỗ tái dùng lại tuần 5 — KHÔNG viết lại retrieval hay generation:

    eval_cases.json ──load_cases──► [EvalCase] ──validate_cases──► dừng nếu có lỗi
                                          │
                                     mỗi ca:
                                          │
              askCli/retriever.retrieve ──► [RetrievedChunk]   (đã lọc ngưỡng 0.35)
                                          │        └─ rỗng ⇒ KHÔNG gọi LLM, answer = NO_ANSWER
                                          │
              askCli/stream_answer.stream_and_collect ──► (text, usage)
                                          │
                                    AnswerRecord ──grader.grade_case──► CaseVerdict
                                          │
                   render_markdown_table ─┴─ print_summary ── log_query(tag='eval')

⚠️ Đây là ĐO PIPELINE THẬT, không phải đo một bản mô phỏng. Nếu ở đây mà tự viết lại
   retrieval "cho gọn" thì con số đo được là của một hệ thống KHÁC hệ thống đem đi khoe.
   Đó là lỗi đã mắc ở tuần 4 (compare_with_python so 2 kho khác nhau) — đừng lặp lại.

Chi phí: 10 ca × ~1 lần gọi Claude Haiku ≈ vài cent. Nhưng chạy 10 lần trong lúc debug
thì tiền và thời gian đều đáng kể -> luôn dùng `--dry` trước, và `--only A1` khi chỉ sửa 1 ca.


Chạy:
    python run_eval.py --dry              # câu trả lời GIẢ, không cần DB/API — kiểm bảng + chấm
    python run_eval.py                    # chạy thật cả 10 ca
    python run_eval.py --only A7          # chạy đúng 1 ca (khi đang debug ca khó)
    python run_eval.py --k 5              # thử ảnh hưởng của k lên chất lượng
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import uuid
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "askCli"))

from dotenv import load_dotenv  # noqa: E402

from eval_cases import DEFAULT_CASES_PATH, EvalCase, load_cases, summarize_dataset, validate_cases  # noqa: E402
from grader import AnswerRecord, CaseVerdict, grade_case  # noqa: E402
from retriever import DEFAULT_TOP_K, retrieve  # noqa: E402
from stream_answer import (  # noqa: E402
    NO_ANSWER,
    estimate_cost,
    get_client,
    log_query,
    stream_and_collect,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

RESULTS_MD = os.path.join(HERE, "eval_results.md")
RESULTS_JSON = os.path.join(HERE, "eval_results.json")


def run_one_case(client, case: EvalCase, k: int) -> tuple[AnswerRecord, dict]:
    """Chạy 1 ca qua pipeline thật. Trả (AnswerRecord, metrics).

    metrics = {"input_tokens", "output_tokens", "total_usd", "elapsed_seconds", "top_similarity"}
    Ví dụ: run_one_case(client, case_A1, k=3) -> (AnswerRecord(answer='...', ...), {...})
    """

    started_at = time.perf_counter()
    empty_metrics = {"input_tokens": 0, "output_tokens": 0, "total_usd": 0.0,
                     "top_similarity": 0.0, "elapsed_seconds": 0.0}

    try:
        chunks = retrieve(case.question, k=k)
        top_similarity = chunks[0].similarity if chunks else 0.0

        if not chunks:
            metrics = {**empty_metrics, "top_similarity": 0.0,
                       "elapsed_seconds": round(time.perf_counter() - started_at, 2)}
            return AnswerRecord(answer = NO_ANSWER, sources=[], n_chunks=0), metrics

        with contextlib.redirect_stdout(io.StringIO()):
            text, usage = stream_and_collect(client, case.question, chunks)

        sources = [(c.source, c.similarity) for c in chunks]

        cost = estimate_cost(usage["input_tokens"], usage["output_tokens"])
        metrics = {**usage, "total_usd": cost["total_usd"],
                   "top_similarity": round(top_similarity, 4),
                   "elapsed_seconds": round(time.perf_counter() - started_at, 2)}
        return AnswerRecord(text, sources, len(chunks)), metrics

    except Exception as exc:
        return (AnswerRecord(answer="", error=f"{type(exc).__name__}: {exc}"),
                {**empty_metrics,
                 "elapsed_seconds": round(time.perf_counter() - started_at, 2)})


def run_all(cases: list[EvalCase], k: int, run_id: str) -> list[dict]:
    """Chạy toàn bộ ca, chấm luôn. Trả list dict — mỗi dict là 1 dòng kết quả đầy đủ.

    1 phần tử: {"case": EvalCase, "record": AnswerRecord, "verdict": CaseVerdict,
                "metrics": {...}}
    """

    client = get_client()
    results = []

    for i, case in enumerate(cases, 1):
        record, metrics = run_one_case(client, case, k=k)
        verdict = grade_case(case, record)
        results.append({"case": case, "record": record,
                        "verdict": verdict, "metrics": metrics})

        mark = "✅" if verdict.passed else "❌"
        print(f"[{i:2}/{len(cases)}] {mark} {case.id:<3} "
              f"{case.question[:45]:<45} {metrics['elapsed_seconds']:>5.1f}s "
              f"{metrics['total_usd']:.5f} USD")

        if not verdict.passed:
            print(f"          └─ {verdict.reason()}")

        log_query({
            "eval_run_id": run_id, "case_id": case.id,
            "question": case.question, "k": k,
            "n_chunks": record.n_chunks, **metrics,
            "passed": verdict.passed, "outcome": "eval",
        })

    return results


def render_markdown_table(results: list[dict]) -> str:
    """Bảng markdown 6 cột, dán thẳng vào README portfolio.

    Mong muốn:
        | # | Câu hỏi | Kỳ vọng | Thực tế | Kết quả | Ghi chú |
        |---|---------|---------|---------|---------|---------|
        | A1 | Vì sao phải chunk...? | answerable | trả lời + 2 nguồn | ✅ PASS | — |
        | N1 | Chi phí vận hành...? | no_answer | BỊA ra con số | ❌ FAIL | refusal: BỊA... |
    """

    def cell(text: str, limit: int = 70) -> str:
        one_line = " ".join(str(text).split()) 
        one_line = one_line.replace("|", "\\|")
        return one_line[:limit] + ("…" if len(one_line) > limit else "")

    lines = ["| # | Câu hỏi | Kỳ vọng | Thực tế | Kết quả | Ghi chú |",
             "|---|---------|---------|---------|---------|---------|"]

    for item in results:
        case, record, verdict = item["case"], item["record"], item["verdict"]
        if record.error:
            actual = "lỗi khi chạy"
        elif record.n_chunks == 0:
            actual = "từ chối (0 chunk qua ngưỡng)"
        else:
            actual = f"trả lời + {len(record.sources)} nguồn"

        status = "✅ PASS" if verdict.passed else "❌ FAIL"
        lines.append(
            f"| {case.id} | {cell(case.question)} | {case.expect} | "
            f"{actual} | {status} | {cell(verdict.reason())} |"
        )
    return "\n".join(lines)


def print_summary(results: list[dict], k: int, run_id: str) -> str:
    """Khối tổng kết: tỉ lệ PASS, tách theo loại ca, chi phí, độ trễ. Trả text để ghi file.

    Mong muốn:
        ===== KẾT QUẢ: 9/10 PASS =====
          answerable : 6/7      no_answer : 3/3
          chi phí lượt chạy : 0.02841 USD   (trung bình 0.00284 / câu)
          độ trễ trung vị   : 3.4s
          ca FAIL: A7 (keywords: thiếu ...)
    """

    def ratio(expect: str) -> str:
        group = [r for r in results if r["case"].expect == expect]
        if not group:
            return "—"
        n_pass = sum(1 for r in group if r["verdict"].passed)
        return f"{n_pass}/{len(group)}"

    total_usd = sum(r["metrics"]["total_usd"] for r in results)
    latencies = [r["metrics"]["elapsed_seconds"] for r in results]
    median_latency = statistics.median(latencies) if latencies else 0.0
    n_pass = sum(1 for r in results if r["verdict"].passed)

    lines = [
        f"===== KẾT QUẢ: {n_pass}/{len(results)} PASS  (k={k}, run={run_id}) =====",
        f"  answerable : {ratio('answerable')}      no_answer : {ratio('no_answer')}",
        f"  chi phí lượt chạy : {total_usd:.5f} USD "
        f"(trung bình {total_usd / max(len(results), 1):.5f} / câu)",
        f"  độ trễ trung vị   : {median_latency:.1f}s",
    ]

    failed = [r for r in results if not r["verdict"].passed]
    if failed:
        lines.append("  ca FAIL:")
        for r in failed:
            lines.append(f"    - {r['case'].id}: {r['verdict'].reason()}")

    text = "\n".join(lines)
    print("\n" + text)
    return text


def save_results(results: list[dict], summary_text: str, table_text: str, run_id: str, k: int) -> None:
    """Ghi eval_results.md (cho README) + eval_results.json (cho tuần 7)."""
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(f"# Kết quả eval — run `{run_id}` · k={k}\n\n")
        f.write("```\n" + summary_text + "\n```\n\n")
        f.write(table_text + "\n")

    payload = [
        {
            "case_id": r["case"].id,
            "question": r["case"].question,
            "expect": r["case"].expect,
            "difficulty": r["case"].difficulty,
            "answer": r["record"].answer,
            "sources": r["record"].sources,
            "error": r["record"].error,
            "passed": r["verdict"].passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                       for c in r["verdict"].checks],
            "metrics": r["metrics"],
        }
        for r in results
    ]
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "k": k, "results": payload},
                  f, ensure_ascii=False, indent=2)
    print(f"\n💾 {RESULTS_MD}\n💾 {RESULTS_JSON}")


def make_fake_results(cases: list[EvalCase]) -> list[dict]:
    """Chế độ --dry: câu trả lời giả để kiểm grader + bảng + tổng kết, không tốn tiền.

    Ca answerable: ghép đủ từ khoá + [1] -> phải PASS.
    Ca no_answer : trả NO_ANSWER -> phải PASS.
    Ca 'A7'      : cố tình thiếu từ khoá -> phải FAIL, để nhìn thấy nhánh FAIL trong bảng.
    """
    results = []
    for case in cases:
        if case.is_trap:
            record = AnswerRecord(answer=NO_ANSWER, sources=[], n_chunks=0)
        elif case.id == "A7":
            record = AnswerRecord(answer="Một câu trả lời chung chung [1].",
                                  sources=[("docs/khong-lien-quan.pdf", 0.40)], n_chunks=1)
        else:
            keywords = " ".join(case.expect_keywords)
            source = case.expect_source or "docs/fake.pdf"
            record = AnswerRecord(answer=f"Câu trả lời giả có {keywords} [1].",
                                  sources=[(source, 0.61)], n_chunks=1)
        verdict = grade_case(case, record)
        metrics = {"input_tokens": 0, "output_tokens": 0, "total_usd": 0.0,
                   "top_similarity": 0.0, "elapsed_seconds": 0.0}
        mark = "✅" if verdict.passed else "❌"
        print(f"[dry] {mark} {case.id:<3} {case.question[:50]}")
        results.append({"case": case, "record": record, "verdict": verdict, "metrics": metrics})
    return results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Chạy bộ eval 10 câu qua pipeline RAG thật.",
        epilog="Luôn chạy --dry trước khi chạy thật.",
    )
    parser.add_argument("--dry", action="store_true",
                        help="dùng câu trả lời giả, không cần DB/API")
    parser.add_argument("--only", default=None, metavar="CASE_ID",
                        help="chỉ chạy 1 ca, ví dụ --only A7")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K,
                        help=f"số chunk mỗi câu (mặc định {DEFAULT_TOP_K})")
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH, help="đường dẫn eval_cases.json")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    cases, corpus_note = load_cases(args.cases)

    # Validate TRƯỚC khi tốn một đồng nào — đây là lý do validate_cases tồn tại.
    errors = validate_cases(cases)
    if errors:
        print(f"❌ Bộ eval có {len(errors)} lỗi, sửa xong mới chạy được:")
        for e in errors:
            print(f"   - {e}")
        return 2

    if args.only:
        cases = [c for c in cases if c.id == args.only]
        if not cases:
            print(f"❌ Không có ca nào id={args.only}")
            return 2

    run_id = uuid.uuid4().hex[:8]
    print(summarize_dataset(cases))
    print(f"📚 corpus: {corpus_note}")
    print(f"▶️  run={run_id} · k={args.k} · {len(cases)} ca"
          + ("  (--dry: KHÔNG gọi API)" if args.dry else "") + "\n")

    results = make_fake_results(cases) if args.dry else run_all(cases, k=args.k, run_id=run_id)

    table_text = render_markdown_table(results)
    summary_text = print_summary(results, k=args.k, run_id=run_id)
    print("\n" + table_text)
    save_results(results, summary_text, table_text, run_id, args.k)

    n_pass = sum(1 for r in results if r["verdict"].passed)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

    # ✅ ĐẠT khi (5 phép thử — làm đủ cả 5):
    #   1. python run_eval.py --dry        -> 10 dòng [dry], A7 ❌, 9/10 PASS, KHÔNG gọi API
    #   2. Mở eval_results.md              -> bảng 6 cột thẳng hàng, mọi dòng cùng số dấu |
    #   3. python run_eval.py              -> chạy thật, tổng chi phí > 0 và < 0.1 USD
    #   4. docker compose stop rồi chạy (3)-> mọi ca ERROR, có bảng tổng kết, KHÔNG traceback
    #   5. grep run_id logs/queries.jsonl  -> đúng 10 dòng, ca N3 có input_tokens = 0
