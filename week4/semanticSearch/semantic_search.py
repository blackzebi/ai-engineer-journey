"""
semantic_search.py — Semantic Search mini (local, miễn phí)

Embed các đoạn text trong corpus.txt -> lưu vector ra vectors.json -> search(query)
bằng cosine similarity -> top 3 đoạn gần nghĩa nhất. Chạy local với sentence-transformers,
không cần API key, không cần vector DB (list Python + cosine là đủ ở quy mô nhỏ).

Chạy:
    python semantic_search.py build                      # embed kho -> vectors.json
    python semantic_search.py search "câu hỏi của bạn"   # query top 3
"""

import os
import sys
import json
import math

from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# all-MiniLM-L6-v2: nhẹ/nhanh nhưng thiên tiếng Anh.
# paraphrase-multilingual-MiniLM-L12-v2: đa ngôn ngữ, tốt cho tiếng Việt. Cả hai đều 384 chiều.
# Đổi model thì phải chạy lại `build` (vector của model khác nhau không so được).
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTORS_FILE = os.path.join(os.path.dirname(__file__), "vectors.json")
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "corpus.txt")

_model = SentenceTransformer(MODEL_NAME)


def load_corpus() -> list[str]:
    """Đọc corpus.txt -> list các đoạn (bỏ dòng trống và dòng comment '#')."""
    if not os.path.exists(CORPUS_FILE):
        raise SystemExit(f"Chưa có {CORPUS_FILE}. Thêm ~20 đoạn (mỗi dòng 1 đoạn) rồi chạy lại.")
    with open(CORPUS_FILE, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed 1 list text (batch) -> list vector. Dùng cùng model cho cả kho lẫn câu hỏi."""
    vectors = _model.encode(texts)
    return vectors.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(a, b) = dot(a, b) / (norm(a) * norm(b)), giá trị trong [-1, 1]. Guard chia 0."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(y ** 2 for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def save_vectors(store: list[dict]) -> None:
    """Lưu list {text, vector} xuống vectors.json (giữ tiếng Việt đọc được)."""
    with open(VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def load_vectors() -> list[dict]:
    """Đọc lại kho đã embed. Chưa build -> báo lỗi thân thiện."""
    if not os.path.exists(VECTORS_FILE):
        raise SystemExit(f"Chưa có {VECTORS_FILE}. Chạy `python semantic_search.py build` trước.")
    with open(VECTORS_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_index() -> None:
    """Embed toàn bộ corpus rồi lưu ra vectors.json (cache để khỏi embed lại mỗi lần)."""
    chunks = load_corpus()
    vectors = embed_texts(chunks)
    store = [{"text": t, "vector": v} for t, v in zip(chunks, vectors)]
    save_vectors(store)
    print(f"✅ Đã embed {len(chunks)} đoạn, mỗi vector {len(vectors[0])} chiều -> {VECTORS_FILE}")


def search(query: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Embed câu hỏi -> cosine với từng đoạn -> sort giảm dần -> top_k tuple (text, score)."""
    q_vec = embed_texts([query])[0]
    store = load_vectors()
    scored = [(item["text"], cosine_similarity(q_vec, item["vector"])) for item in store]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def print_results(query: str, results: list[tuple[str, float]]) -> None:
    print(f"\nQuery: {query!r}")
    print("-" * 70)
    for rank, (text, score) in enumerate(results, 1):
        print(f"{rank}. [{score:6.4f}] {text}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    if cmd == "build":
        build_index()
    elif cmd == "search":
        if len(args) < 2:
            print('Cú pháp: python semantic_search.py search "câu hỏi"')
            return
        query = " ".join(args[1:])
        print_results(query, search(query))
    else:
        print(f"Lệnh không hiểu: {cmd!r}. Dùng 'build' hoặc 'search'.")


if __name__ == "__main__":
    main()
