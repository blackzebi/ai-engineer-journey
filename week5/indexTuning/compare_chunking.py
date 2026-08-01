"""
compare_chunking.py — ĐO 2 cấu hình chunking, không đoán (T3 tuần 5)

    INPUT :  1 folder tài liệu + 2 cấu hình (size=500/overlap=80) vs (size=1000/overlap=150)
    OUTPUT:  bảng so số chunk + top-3 của 5 câu hỏi mẫu, cạnh nhau

Chạy trong RAM, KHÔNG ghi vào DB:

    folder --load_any--> LoadedDoc[Block]
                              |
              +---------------+---------------+
              |                               |
      cfg A: 500/80                    cfg B: 1000/150
              |                               |
      [chunk...] --embed--> [vector]  [chunk...] --embed--> [vector]
              |                               |
        5 câu hỏi --embed--> cosine --> top-3 mỗi bên
              |                               |
              +------> bảng so sánh <---------+

Vì sao không ghi vào bảng chunks: so 2 cấu hình cần 2 kho song song. Qua DB thì phải 2 bảng
/ 2 lần ingest + xoá, mỗi vòng thử mất 10 phút. Trong RAM thì 1 lần chạy có cả 2. Đây là
mầm của eval golden dataset ở tuần 7.

Dùng lại `blocks_to_chunks` của ingest_docs.py thay vì tự cắt, vì hàm đó đã chứa 2 quyết
định của T2: chunk TỪNG BLOCK (không tạo "chunk lai" nửa trang 5 nửa trang 6) và bỏ chunk
< MIN_CHUNK_CHARS. Tự cắt lại = đo một pipeline KHÁC pipeline thật.

Chạy:
    python compare_chunking.py --dry                      # self-check, không cần model/DB
    python compare_chunking.py "D:/Study/tai-lieu-test"
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "ingestDocs"))
sys.path.append(os.path.join(HERE, "..", "..", "week4", "ragPipeline"))

from compare_topk import cosine_similarity  # noqa: E402
from ingest_docs import blocks_to_chunks, collect_files  # noqa: E402
from loaders import load_any  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_N = 3
DOCS_DIR = os.getenv("DOCS_DIR", r"D:\Study\tai-lieu-test")

# Dùng lại 5 câu hỏi của tuần 4 (3 answerable + 2 bẫy) để 2 tuần so được với nhau.
QUESTIONS_FILE = os.path.join(HERE, "..", "..", "week4", "ragLite", "questions.json")

_model = None


@dataclass
class ChunkConfig:
    """1 cấu hình chunking, có tên để in bảng cho gọn."""
    name: str
    chunk_size: int
    overlap: int

    @property
    def overlap_ratio(self) -> float:
        """Tỷ lệ chồng lấn — con số đáng so hơn overlap tuyệt đối.

        500/80 = 16%, 1000/150 = 15% -> gần bằng nhau, nên phép so này cô lập được ảnh hưởng
        của chunk_size. Nếu 2 tỷ lệ lệch nhiều (vd 16% vs 8%) thì đang đổi 2 biến một lúc,
        kết quả không quy được về nguyên nhân nào.
        """
        return self.overlap / self.chunk_size


CONFIG_A = ChunkConfig("A: 500/80", 500, 80)
CONFIG_B = ChunkConfig("B: 1000/150", 1000, 150)


def get_model():
    """Lazy load — để `--dry` chạy được mà không nạp 470MB."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_questions() -> list[str]:
    """Đọc 5 câu hỏi từ week4/ragLite/questions.json. Không có file thì dùng bản dự phòng."""
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, encoding="utf-8") as f:
            return [item["q"] for item in json.load(f)]
    return [
        "asyncio trong Python dùng để làm gì?",
        "Embedding là gì và dùng để làm gì trong semantic search?",
        "Vì sao phải chunk tài liệu trước khi embed?",
        "HNSW khác IVFFlat ở điểm nào?",
        "Cách nấu phở bò truyền thống gồm những bước nào?",
    ]


def load_documents(folder: str) -> list:
    """folder -> list[LoadedDoc]. Dùng lại nguyên si collect_files + load_any của T2 tuần 5."""
    documents = []
    for path in collect_files(folder):
        doc = load_any(path, folder)
        if doc is not None and doc.blocks:
            documents.append(doc)
    return documents


def chunk_with_config(documents: list, config: ChunkConfig) -> list[tuple[str, str]]:
    """Chunk mọi tài liệu theo `config` -> [(source, content), ...].

    Số chunk KHÔNG tỷ lệ nghịch tuyến tính với chunk_size — "size gấp đôi thì chunk còn một
    nửa" là sai, vì:
      · bước nhảy là size - overlap (420 vs 850), không phải size
      · block ngắn hơn chunk_size thì thành 1 chunk BẤT KỂ size 500 hay 1000, nên tài liệu
        nhiều heading/đoạn ngắn gần như không đổi số chunk
    -> tỷ lệ thật luôn nhỏ hơn 2. Đây chính là thứ phải đo.

    Ghi chú chưa xử lý: MIN_CHUNK_CHARS = 50 là ngưỡng TUYỆT ĐỐI, nên với size=1000 nó lọc
    ra ít chunk rác hơn hẳn — 2 cấu hình đang chịu 2 mức lọc khác nhau về mặt tương đối.
    """
    chunks = []
    for doc in documents:
        pieces = blocks_to_chunks(doc, chunk_size=config.chunk_size, overlap=config.overlap)
        for content, _page, _heading, _index in pieces:
            chunks.append((doc.source, content))
    return chunks


def chunk_stats(chunks: list[tuple[str, str]]) -> dict:
    """Thống kê để so 2 cấu hình bằng số:
    n_chunks, n_sources, avg_chars, min_chars, max_chars, n_tiny, total_chars.

    Cần cả min/max chứ không chỉ avg: 2 cấu hình có thể cùng avg 400 ký tự nhưng một bên là
    "toàn chunk 400", bên kia "một nửa 50, một nửa 750" — bên thứ hai retrieval tệ hơn nhiều.

    total_chars để kiểm tra tính nhất quán: 2 cấu hình phải xấp xỉ nhau. Lệch > 30% nghĩa là
    một cấu hình đang MẤT nội dung — pipeline chạy xong, không lỗi, thiếu 1/3 tài liệu.

    n_tiny (chunk < 100 ký tự) = thước đo "rác": cấu hình nào n_tiny cao thì phần chunk tăng
    thêm phần lớn là nhiễu, không phải thông tin.
    """
    # Chặn rỗng trước mọi phép tính: min()/max() trên list rỗng nổ ValueError, và chia cho 0.
    if not chunks:
        return {"n_chunks": 0, "n_sources": 0, "avg_chars": 0.0,
                "min_chars": 0, "max_chars": 0, "n_tiny": 0, "total_chars": 0}

    lengths = [len(content) for _source, content in chunks]

    return {
        "n_chunks": len(chunks),
        "n_sources": len({source for source, _content in chunks}),
        "avg_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "n_tiny": sum(1 for length in lengths if length < 100),   # đếm không tạo list trung gian
        "total_chars": sum(lengths),
    }


def rank_questions(
    chunks: list[tuple[str, str]],
    questions: list[str],
    top_n: int = TOP_N,
) -> dict[str, list[tuple[str, str, float]]]:
    """Với mỗi câu hỏi, trả top-N chunk gần nghĩa nhất: {question: [(source, content, score)]}.

    Embed 1 LẦN cho cả kho rồi mới vào vòng lặp câu hỏi: gọi encode() lồng trong 2 vòng lặp
    là embed lại kho 5 lần, chậm gấp 5 mà kết quả y hệt. Embed là phần đắt nhất, luôn kéo
    ra ngoài vòng lặp.

    ⚠️ Điểm similarity của 2 cấu hình KHÔNG so trực tiếp được: chunk dài hơn thường có
       similarity thấp hơn với câu hỏi ngắn (vector bị pha loãng bởi nội dung ngoài lề).
       Cấu hình B điểm thấp hơn KHÔNG có nghĩa B tệ hơn. Thứ đáng đọc là: đoạn top-1 có
       TRẢ LỜI được câu hỏi không. Chấm bằng mắt — tuần 7 mới có eval chuẩn thay mắt.
    """
    if not chunks:
        return {question: [] for question in questions}

    model = get_model()
    chunk_vectors = model.encode([content for _source, content in chunks],
                                 batch_size=64, show_progress_bar=True).tolist()
    question_vectors = model.encode(questions).tolist()

    results = {}
    for question, question_vector in zip(questions, question_vectors):
        scored = [
            (source, content, cosine_similarity(question_vector, chunk_vector))
            # encode() trả đúng thứ tự input, đó là lý do zip an toàn ở đây
            for (source, content), chunk_vector in zip(chunks, chunk_vectors)
        ]
        scored.sort(key=lambda item: item[2], reverse=True)   # GIẢM dần: đây là similarity
        results[question] = scored[:top_n]
    return results


def print_comparison(
    config_a: ChunkConfig,
    stats_a: dict,
    ranking_a: dict,
    config_b: ChunkConfig,
    stats_b: dict,
    ranking_b: dict,
) -> None:
    """In 2 bảng: số liệu chunking cạnh nhau, rồi top-3 mỗi câu hỏi của 2 bên.

    Phải in CẠNH NHAU: mắt không so được 2 khối cách nhau 40 dòng. So sánh chỉ hữu ích khi
    2 giá trị nằm cùng dòng — đó là lý do file này tồn tại thay vì chạy pipeline 2 lần.
    """
    print(f"\n{'chỉ số':<14}{config_a.name:>16}{config_b.name:>16}")
    print("-" * 46)
    for key in ("n_chunks", "n_sources", "avg_chars", "min_chars",
                "max_chars", "n_tiny", "total_chars"):
        print(f"{key:<14}{stats_a[key]:>16.1f}{stats_b[key]:>16.1f}")

    if stats_a["total_chars"] and stats_b["total_chars"]:
        ratio = stats_b["total_chars"] / stats_a["total_chars"]
        if not 0.7 <= ratio <= 1.3:
            print(f"⚠️  total_chars lệch {ratio:.2f}x — 1 cấu hình đang MẤT nội dung.")

    for question in ranking_a:
        print(f"\n❓ {question}")
        for label, ranking in ((config_a.name, ranking_a), (config_b.name, ranking_b)):
            print(f"  [{label}]")
            for rank, (source, content, score) in enumerate(ranking[question], 1):
                snippet = " ".join(content.split())[:90]   # bóp \n từ PDF, kẻo vỡ bảng
                print(f"    {rank}. [{score:.4f}] {source} · {snippet}...")


def self_check() -> None:
    """Assert cho chunk_stats bằng dữ liệu giả — không cần model, không cần DB."""
    assert chunk_stats([])["n_chunks"] == 0
    fake_chunks = [("a.md", "x" * 100), ("a.md", "y" * 200), ("b.md", "z" * 300)]
    stats = chunk_stats(fake_chunks)
    assert stats["n_chunks"] == 3 and stats["n_sources"] == 2
    assert stats["avg_chars"] == 200.0 and stats["min_chars"] == 100 and stats["max_chars"] == 300
    assert stats["n_tiny"] == 0 and stats["total_chars"] == 600
    assert abs(CONFIG_A.overlap_ratio - 0.16) < 1e-9
    print("✅ self_check: chunk_stats pass (không cần model/DB)")


def main() -> None:
    if "--dry" in sys.argv:
        self_check()
        return

    positional_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    folder = positional_args[0] if positional_args else DOCS_DIR
    if not os.path.isdir(folder):
        raise SystemExit(f"❌ Không thấy folder: {folder}")

    documents = load_documents(folder)
    if not documents:
        raise SystemExit(f"❌ Không load được tài liệu nào trong {folder}")
    print(f"📂 {len(documents)} tài liệu từ {folder}")

    questions = load_questions()
    results = {}
    for config in (CONFIG_A, CONFIG_B):
        chunks = chunk_with_config(documents, config)
        print(f"  {config.name}: {len(chunks)} chunk (overlap {config.overlap_ratio:.0%})")
        results[config.name] = (chunk_stats(chunks), rank_questions(chunks, questions))

    stats_a, ranking_a = results[CONFIG_A.name]
    stats_b, ranking_b = results[CONFIG_B.name]
    print_comparison(CONFIG_A, stats_a, ranking_a, CONFIG_B, stats_b, ranking_b)

    print("\n📋 Viết vào note — 3 câu, có số:")
    print("   1. Cấu hình nào nhiều chunk hơn, tỷ lệ bao nhiêu, có đúng 2:1 như trực giác không?")
    print("   2. Với 5 câu hỏi mẫu, top-1 của bên nào TRẢ LỜI được câu hỏi tốt hơn?")
    print("   3. Nếu phải chọn 1 cấu hình cho Dự án 1, chọn cái nào và VÌ SAO?")

    # Kỳ vọng khi chạy đúng:
    #   - `--dry` in "self_check ... pass"
    #   - n_chunks của A > n_chunks của B, nhưng tỷ lệ NHỎ HƠN 2
    #     (bằng nhau -> blocks_to_chunks chưa nhận tham số, đang so 800/100 với chính nó)
    #   - KHÔNG có cảnh báo "total_chars lệch"
    #   - top-1 của câu "Cách nấu phở bò..." phải có điểm thấp hơn hẳn các câu còn lại


if __name__ == "__main__":
    main()
