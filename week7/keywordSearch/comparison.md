# So sánh vector-only vs keyword-only (top-10)

Corpus: Kho D:\Study\tai-lieu-test — 6 file tài liệu ôn phỏng vấn lập trình (2 PDF, 2 DOCX, 1 TXT, 1 MD): '100+ React Interview Questions' (PDF, 66 trang, tiếng Anh), 'Top 50 Full Stack Developer Interview Questions' (PDF, tiếng Anh), 'Backend-NodeJS-Interview-Knowledge' (DOCX, tiếng Việt), 'Frontend Interview Prep - JS, React, AntD, HTML-CSS, RN' (DOCX, tiếng Việt), 'nodejs_interview_questions.txt' (TXT ngắn, tiếng Anh), 'weekly-tracker.md' (MD, nhật ký lộ trình của Tri — CHỦ ĐỀ LẠC so với 5 file kia, nên nó hay lọt top khi câu hỏi toàn hư từ; đó là tín hiệu tốt để phát hiện truy vấn không có từ khoá phân biệt). Chủ đề chung: phỏng vấn frontend/backend/fullstack. Mọi expect_source dưới đây đã kiểm bằng grep: token quyết định chỉ xuất hiện trong ĐÚNG MỘT file. TODO: điền số dòng chunks thật từ bảng 'Phân bố theo định dạng' của week5/ingestDocs/ingest_docs.py.

| id | loại | câu hỏi | vector | keyword | thắng | đoán đúng? |
|---|---|---|---:|---:|---|---|
| K1 | keyword | Chuỗi kết nối MongoDB ở cổng 27017 viết như thế nà | 1 | — | **vector** | ❌ |
| K2 | keyword | Hermes engine trong React Native là gì? | 3 | 1 | **keyword** | ✅ |
| S1 | semantic | Làm sao để không gọi API liên tục mỗi khi người dù | 2 | — | **vector** | ✅ |
| S2 | semantic | Vì sao một trang hiển thị danh sách lại tạo ra hàn | 4 | 9 | **vector** | ✅ |
| M1 | mixed | Circuit breaker ở trạng thái HALF-OPEN thì xử lý r | 1 | 1 | **tie** | ❌ |
| M2 | mixed | Prop action của thẻ form trong React 19 hoạt động  | 1 | 3 | **vector** | ❌ |
| N1 | keyword | Hash mật khẩu bằng Argon2id thì nên đặt memory cos | — | — | **both_miss** | ✅ |
| N2 | semantic | Ứng dụng viết bằng Flutter thì tối ưu danh sách dà | — | — | **both_miss** | ✅ |
