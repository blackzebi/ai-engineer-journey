# Rerank làm gì với đoạn đúng (hybrid -> +cross-encoder)

**Điều kiện đo**: candidate_pool=20 · top_k=3 · rrf_k=60 · model=cross-encoder/ms-marco-MiniLM-L-6-v2

| id | loại | câu hỏi | hybrid | +rerank (top-3) | đổi |
|---|---|---|---:|---:|---|
| K1 | keyword | Chuỗi kết nối MongoDB ở cổng 27017 viết như thế nà | 6 | 16 | **demoted** |
| K2 | keyword | Hermes engine trong React Native là gì? | 1 | 1 | unchanged |
| S1 | semantic | Làm sao để không gọi API liên tục mỗi khi người dù | 8 | 10 | **demoted** |
| S2 | semantic | Vì sao một trang hiển thị danh sách lại tạo ra hàn | 7 | 6 | unchanged |
| M1 | mixed | Circuit breaker ở trạng thái HALF-OPEN thì xử lý r | 1 | 1 | unchanged |
| M2 | mixed | Prop action của thẻ form trong React 19 hoạt động  | 1 | 4 | **dropped** |
| N1 | keyword | Hash mật khẩu bằng Argon2id thì nên đặt memory cos | — | — | not_in_pool |
| N2 | semantic | Ứng dụng viết bằng Flutter thì tối ưu danh sách dà | — | — | not_in_pool |

```
=== RERANK LÀM GÌ VỚI ĐOẠN ĐÚNG ===
  promoted    : 0
  unchanged   : 3
  demoted     : 2
  dropped     : 1
  not_in_pool : 2

  ✅ Có 3 ca tệ đi -> phép đo
     đáng tin. Viết vào note MỘT CÂU vì sao từng ca tụt.
  🚩 1 ca rerank ĐÁ đoạn đúng khỏi top-k.
     Xem chunk nào đã thay chỗ nó: tiếng Anh hay tiếng Việt?
     Đây là chỗ kiểm giả thuyết 'ms-marco yếu với tiếng Việt'.
  ℹ️ 2 ca đoạn đúng KHÔNG nằm trong pool 20.
     Đây là giới hạn của TẦNG 1, rerank vô can. Muốn cứu thì tăng
     pool hoặc sửa tsv (ca K1), không phải đổi model rerank.
```
