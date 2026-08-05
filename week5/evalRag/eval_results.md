# Kết quả eval — run `3e9a01e4` · k=3

```
===== KẾT QUẢ: 8/10 PASS  (k=3, run=3e9a01e4) =====
  answerable : 5/7      no_answer : 3/3
  chi phí lượt chạy : 0.01731 USD (trung bình 0.00173 / câu)
  độ trễ trung vị   : 3.2s
  ca FAIL:
    - A1: refusal: Từ chối oan - tài liệu có câu trả lời; keywords: thiếu: SIGTERM, server.close
    - A3: refusal: Từ chối oan - tài liệu có câu trả lời; keywords: thiếu: useHistory
```

| # | Câu hỏi | Kỳ vọng | Thực tế | Kết quả | Ghi chú |
|---|---------|---------|---------|---------|---------|
| A1 | Graceful shutdown trong Node gồm những bước nào? | answerable | trả lời + 3 nguồn | ❌ FAIL | refusal: Từ chối oan - tài liệu có câu trả lời; keywords: thiếu: SIGTE… |
| A2 | Vì sao không nên dùng index của mảng làm rowKey trong bảng Ant Design? | answerable | trả lời + 3 nguồn | ✅ PASS | — |
| A3 | Trong React Router, điều hướng bằng code ở v6 dùng cái gì, và ở v5 thì… | answerable | trả lời + 3 nguồn | ❌ FAIL | refusal: Từ chối oan - tài liệu có câu trả lời; keywords: thiếu: useHi… |
| A4 | Làm sao để tránh hàng loạt request cùng đổ dồn vào database ngay khi m… | answerable | trả lời + 3 nguồn | ✅ PASS | — |
| A5 | PropTypes có còn được dùng để kiểm tra kiểu của prop nữa không? Nếu kh… | answerable | trả lời + 3 nguồn | ✅ PASS | — |
| A6 | Ở môi trường dev với React 18 StrictMode, useEffect có dependency arra… | answerable | trả lời + 3 nguồn | ✅ PASS | — |
| A7 | Event loop của Node đi qua những phase nào trong một vòng lặp, và Prom… | answerable | trả lời + 3 nguồn | ✅ PASS | — |
| N1 | Tài liệu khuyến nghị dùng phiên bản Node.js nào cho production năm 202… | no_answer | trả lời + 3 nguồn | ✅ PASS | — |
| N2 | Giá cổ phiếu Apple hôm nay là bao nhiêu? | no_answer | từ chối (0 chunk qua ngưỡng) | ✅ PASS | — |
| N3 | Cách nấu phở bò truyền thống gồm những bước nào? | no_answer | từ chối (0 chunk qua ngưỡng) | ✅ PASS | — |
