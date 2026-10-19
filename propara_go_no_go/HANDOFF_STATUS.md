# HANDOFF_STATUS: đang ở đâu, còn gì, ai làm, mất bao lâu

> **Cập nhật lần cuối:** 19/10/2026, trợ lý.
> File này dùng để **theo dõi nhanh**. Chi tiết từng thí nghiệm nằm ở `EXPERIMENT_LOG.md`.
> Khi mở cuộc trò chuyện mới: gửi **file này + EXPERIMENT_LOG.md**.
> Mỗi khi xong một việc, sửa ô *Trạng thái* (⬜ chưa · 🟡 đang · ✅ xong · ❌ lỗi) và ghi một dòng vào *Work log* ở cuối.

---

## 1. TỐI NAY (deadline): báo cáo tiến độ (làm vào buổi tối, theo yêu cầu của bạn)

**Sản phẩm nộp tối nay = Báo cáo Go/No-Go sơ bộ** (`reports/go_no_go_preliminary.md`). Bản nháp **đã viết sẵn** với số liệu thật từ 00/01. Chỉ còn trống phần Qwen.

| # | Việc | Ai | Thời gian **bạn bỏ ra** | Thời gian chờ | Trạng thái |
|---|---|---|---|---|---|
| T1 | Chạy `03_qwen7b_dev.ipynb` trên Kaggle (hướng dẫn: `README.md` mục K1) | **Bạn** | 10–15 phút (import, chỉnh settings, bấm chạy) | 30–60 phút máy chạy (ước tính, chưa đo); trong lúc đó làm việc khác | ⬜ |
| T2 | Gửi trợ lý khối `=== GỬI LẠI ===` + `summary_03.json` | **Bạn** | 2 phút | — | ⬜ |
| T3 | Phân tích kết quả Qwen, điền vào báo cáo sơ bộ, chốt kết luận tạm | Trợ lý | 0 | khoảng 20–40 phút | ⬜ |
| T4 | Đọc lại báo cáo, sửa câu chữ, nộp | **Bạn** | 20–30 phút | — | ⬜ |

**Tổng thời gian bạn thực sự phải làm tối nay: khoảng 35–50 phút. Tổng thời gian chờ đợi: khoảng 1,5–2,5 giờ.**

**Phương án dự phòng:** nếu K1 lỗi hoặc không kịp (ví dụ hết quota GPU, tài khoản chưa xác minh số điện thoại), vẫn nộp được báo cáo sơ bộ ở trạng thái hiện tại:
- dữ liệu và bộ chấm đã kiểm chứng;
- 2 baseline gốc chấm cùng bộ chấm;
- bằng chứng "phương pháp cũ bắt được 0 cặp ngầm";
- kết luận **INCONCLUSIVE**, kèm kế hoạch các bước tiếp.

Đây vẫn là câu trả lời trung thực cho yêu cầu của giảng viên.

---

## 2. PIPELINE TOÀN BỘ: đang ở đâu (cập nhật chiều 19/10)

```
[✅] D0 Phản biện báo cáo + khoá đề cương
[✅] D1 Dữ liệu (NB00): PASS
[✅] D2 Bộ chấm + heuristic (NB01): PASS (Cửa nghiệm thu 1)
[✅] D3 Baseline: heuristic B1a/b/c ✅ · heuristic đầu-cuối B1-E2E ✅ · ProStruct/ProGlobal ✅ · **CGLI ✅** (chấm file dự đoán công bố) · MeeT/XPAD: chưa có mã công khai
[🟡] D4 Qwen dev: NB03 = K1 ← BẠN CHẠY NGAY
[🟡] D5 PP3 + PP4 + ablation + CV: NB05 = K2 ← mã xong, đang kiểm thử cục bộ; bạn chạy SAU K1 (hoặc song song nếu tài khoản cho 2 phiên GPU)
[🟡] D6 Test khoá + đổi tên: NB04 = K3 ← mã xong; chạy SAU khi trợ lý xác nhận khoá prompt từ K1
[⬜] D7 Chấm test (đã lưu kín), thống kê, biểu đồ, so bài báo: trợ lý làm trên CPU khi có output K1+K2+K3
[⬜] D8 Báo cáo Go/No-Go cuối: trợ lý viết, bạn duyệt
```

## 3. VIỆC CỦA BẠN TRÊN KAGGLE (cả pipeline chỉ có 3 lượt)

| Lượt | Notebook | Khi nào | Settings | Công sức bạn | Máy chạy (ước tính, chưa đo) | Gửi lại |
|---|---|---|---|---|---|---|
| **K1** | `03_qwen7b_dev.ipynb` | **ngay bây giờ** | GPU T4 x2, Internet On | 10 phút | 30–60 phút | khối GỬI LẠI + `summary_03.json` (+ zip nếu được) |
| **K2** | `05_pp3_pp4_dev_cv.ipynb` | sau K1 (hoặc song song) | GPU (T4/P100), Internet On | 10 phút | 2–3 giờ | khối GỬI LẠI + `pp34_v2.zip` |
| **K3** | `04_qwen7b_test_locked.ipynb` | khi trợ lý báo "đã khoá" sau K1 | GPU T4 x2, Internet On | 10 phút | 40–80 phút | khối GỬI LẠI + `qwen7b_test_v1.zip` |

Cách chạy cả 3 lượt giống nhau: **File → Import Notebook → chỉnh Settings → Save Version → Save & Run All (Commit)**. Xong thì vào tab **Output** của phiên bản để tải zip.

**Tổng công sức của bạn cho cả pipeline: khoảng 30–40 phút thao tác + 1–2 lần gửi file.** Thời gian máy chạy khoảng 4–5 giờ, chạy nền, bạn không cần ngồi chờ.

Sau K3, trợ lý tự làm phần còn lại (D7–D8) trên CPU, khoảng 1–2 giờ.

## 4. File quan trọng

| File | Dùng để |
|---|---|
| `PIPELINE_TONG_QUAN.md` | sơ đồ pipeline tổng quan + công dụng từng file code |
| `HANDOFF_STATUS.md` (file này) | xem nhanh tiến độ + việc của bạn |
| `EXPERIMENT_LOG.md` | nhật ký chi tiết từng thí nghiệm (giả thuyết, thiết lập, kết quả, quyết định) |
| `reports/go_no_go_preliminary.md` | **báo cáo nộp tối nay** |
| `README.md` | hướng dẫn bấm từng bước trên Kaggle |
| `docs/research_audit.md` | các phát hiện V1–V12 khi phản biện báo cáo |
| `docs/experiment_protocol.md` | thước đo, quy tắc quyết định đã khoá |
| `docs/KAGGLE_RUN_PLAN.md` | các lượt Kaggle K1–K3 |

---

## 5. Work log (thêm dòng mới ở cuối)

| Thời điểm | Ai | Việc | Kết quả |
|---|---|---|---|
| 19/10 ~02:30 | Trợ lý | Đọc 4 tài liệu, tải ProPara + evaluator chính thức, viết research_audit + protocol | ✅ 14/14 nhóm số liệu của báo cáo tái hiện được; phát hiện Q3 của báo cáo ≠ Q3 chính thức |
| 19/10 ~02:45 | Trợ lý | NB00 kiểm tra dữ liệu (chạy cục bộ) | ✅ PASS |
| 19/10 ~12:45 | Trợ lý | NB01 bộ chấm + heuristic + ProStruct/ProGlobal (chạy cục bộ) | ✅ PASS; 2 phương pháp gốc bắt 0 cặp ngầm |
| 19/10 ~12:50 | Trợ lý | NB03 Qwen dev: viết + kiểm thử luồng bằng mô hình tí hon | ✅ sẵn sàng cho Kaggle |
| 19/10 ~13:30 | Trợ lý | Viết HANDOFF_STATUS + bản nháp báo cáo sơ bộ | ✅ |
| 19/10 ~22:55 | Trợ lý | Tự đối chiếu với báo cáo PDF (DOI_CHIEU_BAO_CAO.md); sửa 2 thiếu sót mức Cao: đầu dự đoán vị trí + PP2 theo τ1/τ2 | ✅ K2 bản mới đã kiểm thử; **dùng bản mới khi chạy K2** |
| 19/10 ~22:25 | Trợ lý | Thêm đối chứng CGLI (B2) + heuristic đầu-cuối (B1-E2E) vào NB01, chạy lại PASS; bảng đối chiếu phương pháp trong PIPELINE_TONG_QUAN | ✅ CGLI PAIR_F1 0,545; đích ngầm chỉ 0,091 |
| 19/10 tối | Trợ lý | Thêm checkpoint theo epoch + dừng sớm + khôi phục sang phiên mới cho K2; khôi phục đầu ra thô cho K1/K3 | ✅ đã kiểm thử (ngắt rồi tiếp tục cho kết quả trùng khớp) |
| 19/10 ~14:00 | Trợ lý | PP4: CRF chính xác (kiểm thử khớp liệt kê vét cạn), RoBERTa+BiLSTM, 6 biến thể; PP3: trích mẫu WikiHow + E(X,Y); biến thể đổi tên NONCE; NB05 (K2), NB04 (K3) | 🟡 đang kiểm thử cục bộ |
| 19/10 ~23:10 | Trợ lý | Tự rà soát “có bịa không”: sửa bảng đếm sai trong DOI_CHIEU, E0 chỉ khớp khi gộp, số bài WikiHow chưa kiểm chứng (bài gốc ghi >230k), năm bài KOALA, mốc giờ work log, ghi rõ máy đo tốc độ | ✅ |
| 19/10 ~23:20 | Trợ lý | Đổi tên output K2 thành `pp34_v2` (bản có đầu vị trí + PP2 τ + checkpoint) để không lẫn với bản cũ `pp34_v1` | ✅ |
| | Bạn | Chạy K1 trên Kaggle | ⬜ |
