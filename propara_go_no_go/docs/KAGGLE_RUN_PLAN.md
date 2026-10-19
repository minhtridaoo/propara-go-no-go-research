# KAGGLE_RUN_PLAN: số lượt chạy tối thiểu (cập nhật 19/10/2026 ~23h30)

> PP3 trước đây dự kiến là notebook riêng (`06_pp3_wikihow`), nay đã **gộp vào K2** (notebook 05), nên không còn lượt K2b.

**Nguyên tắc:** việc gì chạy được bằng CPU thì trợ lý tự chạy trên máy (dữ liệu ProPara chỉ ~1 MB). Kaggle chỉ dùng cho GPU. Mỗi notebook GPU **tự chứa**: tự tải dữ liệu, tự nhúng bộ chấm và mã lõi, tự chạy kiểm thử, có smoke test tự động. **Không** cần Add Input output của notebook trước.

| Lượt | Notebook | Ai chạy | Phần cứng | Trạng thái | Vì sao tách riêng |
|---|---|---|---|---|---|
| — | 00 data audit | trợ lý (máy, CPU) | CPU | ✅ PASS cục bộ. Chạy trên Kaggle là **tuỳ chọn** | — |
| — | 01 bộ chấm + heuristic | trợ lý (máy, CPU) | CPU | ✅ PASS cục bộ = **Cửa nghiệm thu 1 đạt** | — |
| **K1** | `03_qwen7b_dev.ipynb` | **bạn** | GPU T4 x2 | ⏳ sẵn sàng | cần GPU; kết quả dev quyết định khoá prompt |
| **K2** | `05_pp3_pp4_dev_cv.ipynb`, bản `pp34_v2` (gộp PP3 WikiHow + PP4 có đầu vị trí + PP2 theo τ1/τ2; 6 hệ thống × 3 seed; CV 5 phần; đổi tên NONCE; checkpoint) | **bạn** | GPU | ⏳ đã viết + kiểm thử cục bộ | cần GPU; chạy độc lập với K1 |
| **K3** | `04_test_locked.ipynb` (Qwen + M3 trên test **một lần**; biến thể diễn đạt lại/đổi tên) | **bạn** | GPU | ⬜ | **bắt buộc** sau khi khoá cấu hình trên dev (chống rò rỉ test) |
| — | 09–10 phân tích, thống kê, báo cáo Go/No-Go | trợ lý (máy, CPU) | CPU | ⬜ | — |

## Vì sao không gộp tất cả thành một lượt
1. Test chỉ được mở **sau khi** khoá cấu hình trên dev. Nếu gộp thì điểm test mất giá trị làm bằng chứng (protocol §9).
2. Phiên GPU của Kaggle có giới hạn thời lượng và quota theo tuần. Gộp lượt dài thì lỗi giữa chừng sẽ mất cả lượt.
3. Test chỉ được chạy sau khi đã xem kết quả dev và khoá cấu hình.

## Ước tính K1
Chưa đo. Độ dài prompt đo bằng tokenizer thật của Qwen2.5-7B-Instruct@a09a354:
- Q0 ~480 token, few-shot ~1.700, structured ~500, oracle ~500 × 290 lời gọi.
- Tổng thời gian dự kiến: tải mô hình 5–10 phút + nạp 2–5 phút + sinh khoảng 20–40 phút trên 2×T4 fp16.

Thời gian thật được ghi trong `summary_03.json`.
