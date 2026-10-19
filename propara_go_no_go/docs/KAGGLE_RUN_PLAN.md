# KAGGLE_RUN_PLAN: số lượt chạy tối thiểu (cập nhật 19/10/2026)

**Nguyên tắc:** việc gì chạy được bằng CPU thì trợ lý tự chạy trên máy (dữ liệu ProPara chỉ ~1 MB). Kaggle chỉ dùng cho GPU. Mỗi notebook GPU **tự chứa**: tự tải dữ liệu, tự nhúng bộ chấm và mã lõi, tự chạy kiểm thử, có smoke test tự động. **Không** cần Add Input output của notebook trước.

| Lượt | Notebook | Ai chạy | Phần cứng | Trạng thái | Vì sao tách riêng |
|---|---|---|---|---|---|
| — | 00 data audit | trợ lý (máy, CPU) | CPU | ✅ PASS cục bộ. Chạy trên Kaggle là **tuỳ chọn** | — |
| — | 01 bộ chấm + heuristic | trợ lý (máy, CPU) | CPU | ✅ PASS cục bộ = **Cửa nghiệm thu 1 đạt** | — |
| **K1** | `03_qwen7b_dev.ipynb` | **bạn** | GPU T4 x2 | ⏳ sẵn sàng | cần GPU; kết quả dev quyết định khoá prompt |
| **K2** | `07_pp4_train_dev.ipynb` (M0, M0c, M1, M2r, M3, M3-no-E, M3-no-pair; CV 5-fold) | **bạn** | GPU | ⬜ trợ lý viết sau K1 | cần GPU; cần biết lỗi của Qwen trên dev để thiết kế phân tích |
| K2b | `06_pp3_wikihow` (trích cặp + độ phủ: CPU; huấn luyện E(X,Y): GPU nhỏ) | trợ lý / bạn | CPU → GPU | ⬜ | phụ thuộc nguồn dữ liệu WikiHow (cần xác minh) |
| **K3** | `04_test_locked.ipynb` (Qwen + M3 trên test **một lần**; biến thể diễn đạt lại/đổi tên) | **bạn** | GPU | ⬜ | **bắt buộc** sau khi khoá cấu hình trên dev (chống rò rỉ test) |
| — | 09–10 phân tích, thống kê, báo cáo Go/No-Go | trợ lý (máy, CPU) | CPU | ⬜ | — |

## Vì sao không gộp tất cả thành một lượt
1. Test chỉ được mở **sau khi** khoá cấu hình trên dev. Nếu gộp thì điểm test mất giá trị làm bằng chứng (protocol §9).
2. Phiên GPU của Kaggle có giới hạn thời lượng và quota theo tuần. Gộp lượt dài thì lỗi giữa chừng sẽ mất cả lượt.
3. PP4/PP3 phải viết dựa trên dữ liệu thật về lỗi của Qwen và độ phủ WikiHow, nếu không sẽ lãng phí GPU.

## Ước tính K1
Chưa đo. Độ dài prompt đo bằng tokenizer thật của Qwen2.5-7B-Instruct@a09a354:
- Q0 ~480 token, few-shot ~1.700, structured ~500, oracle ~500 × 290 lời gọi.
- Tổng thời gian dự kiến: tải mô hình 5–10 phút + nạp 2–5 phút + sinh khoảng 20–40 phút trên 2×T4 fp16.

Thời gian thật được ghi trong `summary_03.json`.
