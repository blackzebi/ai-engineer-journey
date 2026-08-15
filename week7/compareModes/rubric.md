# Tiêu chí chấm — điền TRƯỚC khi chạy compare_modes.py

> ⚠️ Điền file này lúc 09:00–10:00, **trước khi nhìn bất kỳ kết quả nào**.
> Viết tiêu chí sau khi đã thấy số là confirmation bias có tổ chức: mình sẽ vô thức chọn
> ngưỡng nào làm cho chế độ mình thích thắng. Điền xong thì commit luôn file này — dấu thời
> gian của git là bằng chứng mình chốt trước.

Ngày chốt: 13/08/2026 · Bộ câu hỏi: `week7/keywordSearch/questions.json` (8 câu)

---

## 1. Thế nào là "đoạn đúng"?

`rank_of_source` khớp **một phần tên file**, không phân biệt hoa thường.
Trả lời trước khi chạy:

- [X] Một chunk từ đúng file nhưng **sai mục** (ví dụ đúng file Frontend Interview Prep
      nhưng là đoạn nói về AntD chứ không phải Hermes) — có tính là đúng không?
      → Quyết định: trong trường hợp này vẫn là sai.
      → Hệ quả nếu tính là đúng: MRR bị thổi lên vì file lớn dễ trúng. Ghi hệ quả này vào README.

- [X] Với 2 câu no_answer (N1 Argon2id, N2 Flutter): `expect_source = null` nên hạng luôn
      là `—`. Đây là **thiết kế**, không phải lỗi. Chúng được đo bằng `distinct_sources`.
      → Xác nhận đã hiểu: X

---

## 2. Ba câu hỏi phải trả lời bằng SỐ, chốt ngưỡng trước

| # | Câu hỏi | Đo bằng | Ngưỡng tự chốt (điền TRƯỚC) |
|---|---|---|---|
| 1 | Top-1 có dùng được không? | `hit@1` (rank == 1) | "chấp nhận được" khi ≥ 0.5 |
| 2 | Đoạn đúng nằm ở hạng mấy? | `MRR` trên 6 câu answerable | chênh dưới 3 coi là nhiễu |
| 3 | Người dùng có đọc được câu trả lời không? | `hit@3` | mục tiêu ≥ 0.6 |
| 4 | Đắt thêm bao nhiêu? | `p50 latency` (n=8) | chấp nhận thêm tối đa 6 ms |

> Gợi ý neo, không phải đáp án: với 6 câu answerable, MRR chênh dưới **0.05** thì không
> phân biệt được (cùng tinh thần `CLEAR_WIN_MARGIN=2` của T2). Latency: người dùng chịu
> được ~1s cho một lần hỏi, nhưng retrieval chỉ là một phần — phần sinh câu trả lời của
> tuần 5 còn tốn thêm vài giây nữa.

---

## 3. Điều kiện để nói "chế độ X đáng dùng cho Dự án 1"

Chốt **luật quyết định** trước, rồi mới chạy. Điền vào chỗ trống:

> Chọn chế độ đơn giản nhất mà `hit@3` ≥ 0.6 **và** `p50` ≤ 6 ms.
> Chỉ đổi sang chế độ phức tạp hơn khi nó cải thiện `MRR` ít nhất 3 .

Vì sao viết thành luật: sau khi nhìn bảng, mình sẽ luôn tìm được lý do để chọn cái mới nhất
và "xịn" nhất. Luật viết trước là thứ duy nhất cãi lại được cái đó.

---

## 4. Ca nào PHẢI có mặt trong README (kể cả khi xấu)

- [X] Ít nhất **một câu mà chế độ phức tạp hơn lại tệ hơn** — nếu không có, xem cảnh báo
      trong `print_verdict` (rất có thể chế độ ③ đang lặng lẽ trả về y hệt ②).
- [ ] Kết quả của **N1 và N2** — kể cả khi tín hiệu `distinct_sources` không phân biệt được
      gì. "Đo rồi, chưa kết luận được với n=2" là một kết quả hợp lệ.
- [X] Giới hạn đã biết: cross-encoder `ms-marco-MiniLM` huấn luyện trên **tiếng Anh**, mà
      4/8 câu hỏi và 2/6 tài liệu là tiếng Việt. Full-text dùng config `simple` (không stem
      tiếng Việt).

---

## 5. Sau khi chạy — điền lại đúng một lần, không sửa ngược lên trên

| Chế độ | MRR | hit@3 | p50 ms | Đạt luật ở mục 3? |
|---|---:|---:|---:|---|
| ① vector-only | 0.68 | 0.83 | 104 ms |  |
| ② hybrid (RRF) | 0.57 | 0.50 | 91 ms | -13 ms so với ① |
| ③ hybrid + rerank | 0.42 | 0.33 | 1782 ms | +1678 ms so với ① |

**Chế độ chọn cho Dự án 1:** ..........
**Vì sao (1 câu, phải dẫn số ở bảng trên):** ..........
**Chỗ mình đã đoán sai trước khi chạy:** đã đoán sai về hit@3 >= 0.6 và p50 < 6ms
