# So 3 chế độ retrieval — bộ câu hỏi Kho D:\Study\tai-lieu-test — 6 file tài liệu ôn phỏng vấn lập trình (2 PDF, 2 DOCX, 1 TXT, 1 MD): '100+ React Interview Questions' (PDF, 66 trang, tiếng Anh), 'Top 50 Full Stack Developer Interview Questions' (PDF, tiếng Anh), 'Backend-NodeJS-Interview-Knowledge' (DOCX, tiếng Việt), 'Frontend Interview Prep - JS, React, AntD, HTML-CSS, RN' (DOCX, tiếng Việt), 'nodejs_interview_questions.txt' (TXT ngắn, tiếng Anh), 'weekly-tracker.md' (MD, nhật ký lộ trình của Tri — CHỦ ĐỀ LẠC so với 5 file kia, nên nó hay lọt top khi câu hỏi toàn hư từ; đó là tín hiệu tốt để phát hiện truy vấn không có từ khoá phân biệt). Chủ đề chung: phỏng vấn frontend/backend/fullstack. Mọi expect_source dưới đây đã kiểm bằng grep: token quyết định chỉ xuất hiện trong ĐÚNG MỘT file. TODO: điền số dòng chunks thật từ bảng 'Phân bố theo định dạng' của week5/ingestDocs/ingest_docs.py.

k=10 · candidate_pool=20 · RRF k=60

| id | loại | câu hỏi | ① vector-only rank | ② hybrid (RRF) rank | ③ hybrid + rerank rank | ① vector-only ms | ② hybrid (RRF) ms | ③ hybrid + rerank ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| K1 | keyword | Chuỗi kết nối MongoDB ở cổng 27017 viết như thế nà | 1 | 6 | — | 145 | 126 | 2167 |
| K2 | keyword | Hermes engine trong React Native là gì? | 3 | 1 | 1 | 87 | 98 | 1892 |
| S1 | semantic | Làm sao để không gọi API liên tục mỗi khi người dù | 2 | 8 | 10 | 100 | 55 | 2308 |
| S2 | semantic | Vì sao một trang hiển thị danh sách lại tạo ra hàn | 4 | 7 | 6 | 98 | 96 | 2177 |
| M1 | mixed | Circuit breaker ở trạng thái HALF-OPEN thì xử lý r | 1 | 1 | 1 | 103 | 90 | 1714 |
| M2 | mixed | Prop action của thẻ form trong React 19 hoạt động  | 1 | 1 | 4 | 114 | 99 | 2075 |
| N1 | no_answer | Hash mật khẩu bằng Argon2id thì nên đặt memory cos | — | — | — | 103 | 101 | 1925 |
| N2 | no_answer | Ứng dụng viết bằng Flutter thì tối ưu danh sách dà | — | — | — | 117 | 116 | 2987 |

| chế độ | MRR (6 câu) | hit@3 | p50 latency | ghi chú |
|---|---:|---:|---:|---|
| ① vector-only | 0.68 | 0.83 | 103 ms |  |
| ② hybrid (RRF) | 0.57 | 0.50 | 98 ms | -4 ms so với ① |
| ③ hybrid + rerank | 0.42 | 0.33 | 2121 ms | +2018 ms so với ① |

```
🏆 MRR cao nhất: ① vector-only (0.68, hit@3 0.83)
🥇 Chế độ MỐC thắng — không tầng nào thêm vào có ích trên bộ này.
📉 ② hybrid (RRF) tệ hơn ① ở: K1 (1 -> 6), S1 (2 -> 8), S2 (4 -> 7)
📉 ③ hybrid + rerank tệ hơn ① ở: K1 (1 -> —), S1 (2 -> 10), S2 (4 -> 6), M2 (1 -> 4)
```
