# Latency 4 tầng: vector -> +keyword -> +fusion -> +rerank

**Điều kiện đo**: 8 câu · repeat=5 · warmup=1 · pool=20 · top_k=3 · rrf_k=60 · cross-encoder/ms-marco-MiniLM-L-6-v2 · CPU

**TODO điền tay**: số dòng bảng chunks = ? · máy = ?

Corpus: Kho D:\Study\tai-lieu-test — 6 file tài liệu ôn phỏng vấn lập trình (2 PDF, 2 DOCX, 1 TXT, 1 MD): '100+ React Interview Questions' (PDF, 66 trang, tiếng Anh), 'Top 50 Full Stack Developer Interview Qu...

| tầng | trung vị (ms) | p95 (ms) | tầng này thêm (ms) |
|---|---:|---:|---:|
| vector | 73.0 | 79.0 | — |
| + keyword | 79.9 | 91.1 | +6.9 |
| + fusion (RRF) | 80.7 | 108.5 | +0.8 |
| + rerank (cross-encoder) | 1599.3 | 1813.0 | +1518.7 |

```
=== ĐỌC BẢNG LATENCY ===
  Rerank 20 ứng viên thêm 1519 ms (trung vị), đưa tổng từ 81 ms lên 1599 ms.
  p95 tổng: 1813 ms.
  ⚠️ Tầng đầu 73 ms — đang đo cả thời gian LOAD
     MODEL. Tăng warmup, hoặc gọi preload_models() trước khi đo.

  📋 Chép vào note KÈM: số dòng bảng chunks · pool · top_k · CPU hay GPU
     · đã warm-up · repeat=? — thiếu một mục là số này hết so được.
```
