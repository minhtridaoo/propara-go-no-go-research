# propara_go_no_go: kiểm chứng Go/No-Go đề tài “Ghép cặp chuyển đổi thực thể”

Điều phối theo `../MASTER_PROMPT_PROPARA_KAGGLE_GO_NO_GO.md`. Tiến độ ghi trong `EXPERIMENT_LOG.md`. Khi mở cuộc trò chuyện mới, **gửi file log này** để tiếp tục. Kế hoạch các lượt Kaggle nằm trong `docs/KAGGLE_RUN_PLAN.md`.

```
propara_go_no_go/
├── PIPELINE_TONG_QUAN.md             ← ĐỌC ĐẦU TIÊN: sơ đồ pipeline + mỗi file code làm gì
├── docs/DOI_CHIEU_BAO_CAO.md         ← đối chiếu từng mục báo cáo PDF: đúng / khác / thiếu
├── HANDOFF_STATUS.md                 ← đang ở đâu, việc của bạn, mất bao lâu, work log
├── EXPERIMENT_LOG.md                 ← nhật ký chi tiết từng thí nghiệm
├── reports/go_no_go_preliminary.md   ← báo cáo sơ bộ (nộp tối nay; chờ điền số Qwen)
├── docs/research_audit.md            ← phản biện báo cáo, số liệu tái hiện, vấn đề V1–V8
├── docs/experiment_protocol.md       ← giả thuyết, thước đo, thống kê, quy tắc quyết định (v1.1, đã khoá)
├── docs/KAGGLE_RUN_PLAN.md           ← các lượt Kaggle K1–K3 và lý do không gộp
├── notebooks/
│   ├── 00_propara_data_audit.ipynb   ← CPU · trợ lý đã chạy cục bộ: PASS (chạy Kaggle: tuỳ chọn)
│   ├── 01_metrics_heuristics.ipynb   ← CPU · trợ lý đã chạy cục bộ: PASS = Cửa nghiệm thu 1
│   ├── 03_qwen7b_dev.ipynb           ← GPU · **LƯỢT K1** (chạy ngay)
│   ├── 05_pp3_pp4_dev_cv.ipynb       ← GPU · **LƯỢT K2** (PP3+PP4+ablation+CV)
│   └── 04_qwen7b_test_locked.ipynb   ← GPU · **LƯỢT K3** (chỉ chạy khi trợ lý báo đã khoá)
├── kaggle_upload/propara-raw-v1.zip  ← dữ liệu gốc dự phòng khi không bật được Internet
├── src/                              ← mã lõi + kiểm thử (đã nhúng sẵn vào notebook)
├── vendor/aristo_propara_evaluator.zip ← evaluator chính thức (nguyên văn, Apache-2.0)
├── tools/                            ← sinh lại notebook từ src/
└── results/                          ← kết quả chạy CỤC BỘ (00, 01) để đối chiếu
```

---

## LƯỢT K1: chạy `03_qwen7b_dev.ipynb` trên Kaggle (GPU, một lần Run All)

1. **Import:** kaggle.com → **Create → New Notebook** → **File → Import Notebook** → chọn `notebooks/03_qwen7b_dev.ipynb`.
2. **Settings** (khung phải → *Session options*):
   - **Accelerator: GPU T4 x2.** Nếu chỉ có 1 GPU (T4/P100) vẫn chạy được: notebook tự chuyển sang 4-bit và cần Internet để cài `bitsandbytes`.
   - **Internet: On**. Nếu không thấy công tắc này, thường là do tài khoản chưa xác minh số điện thoại (theo hiểu biết của trợ lý về Kaggle, chưa kiểm tra lại trong phiên này).
   - *(Tuỳ chọn, giúp nạp nhanh hơn)* **Add Input → Models** → tìm `Qwen2.5` → framework **Transformers** → biến thể **7b-instruct**. Notebook tự dò. Nếu không có thì notebook tự tải từ Hugging Face (commit ghim `a09a354`), **không** ghi trọng số vào Output.
3. **Chạy nền ngay để khỏi phải giữ trình duyệt:** **Save Version → Save & Run All (Commit) → Save**. Hoặc **Run All** để xem trực tiếp rồi Save Version sau.
   - Notebook tự chạy *smoke test* trên 3 đoạn **train**. Nếu đạt thì **tự chạy tiếp toàn bộ dev** (43 đoạn × 4 chế độ). Nếu không đạt thì **tự dừng** để không tốn GPU.
   - Thời gian **ước tính** (chưa đo trên Kaggle): khoảng 30–60 phút.
4. **Xem kết quả:** mở phiên bản đã chạy → tab **Output** → `qwen7b_dev_v1/` và `qwen7b_dev_v1.zip`. Kiểm tra trạng thái phiên bản là *Successful*.
5. **Gửi lại cho trợ lý:**
   - toàn bộ khối `=== GỬI LẠI === … === HẾT ===` ở ô cuối;
   - file `qwen7b_dev_v1/summary_03.json`;
   - *(nếu được)* file `qwen7b_dev_v1.zip`, để trợ lý phân tích lỗi từng đoạn.

Nếu bị lỗi: gửi **traceback** của ô lỗi kèm số thứ tự ô. Không cần tự sửa.

### Lỗi thường gặp (K1)
| Triệu chứng | Cách xử lý |
|---|---|
| `Không thấy GPU` | Settings → Accelerator → GPU T4 x2 |
| Tải mô hình thất bại / không có Internet | Bật Internet, hoặc Add Input mô hình Qwen2.5-7B-Instruct (Transformers) từ Kaggle Models |
| `[OOM] giảm lô còn …` | Bình thường: notebook tự giảm kích thước lô và chạy tiếp |
| `SMOKE KHÔNG ĐẠT` | Không chạy lại. Gửi output (có đầu ra thô mẫu) cho trợ lý |
| Phiên bị ngắt giữa chừng | Run All lại **trong cùng phiên**: các đoạn đã sinh được giữ trong `raw_outputs/` và bỏ qua. Sang phiên mới: Add Input → Notebooks → output phiên bản trước của notebook này → Run All |
| Hết quota GPU tuần | Đợi quota làm mới (xem Settings tài khoản → *GPU quota*) |

---

---

## LƯỢT K2: `05_pp3_pp4_dev_cv.ipynb` (mô hình nhóm; GPU; ước tính 3–5 giờ, chưa đo)
1. **File → Import Notebook** → chọn `notebooks/05_pp3_pp4_dev_cv.ipynb`.
2. **Settings:** Accelerator **GPU** (T4 x2 hoặc P100; notebook dùng 1 GPU) · **Internet On**.
3. **Save Version → Save & Run All (Commit)**, rồi để chạy nền. Notebook tự tải `roberta-base` và WikiHow (~620 MB) vào bộ nhớ tạm.
4. Khi xong: tab **Output** → tải `pp34_v2.zip`. Gửi trợ lý khối `=== GỬI LẠI ===` + file zip này.
- Có thể chạy **song song** với K1 nếu tài khoản cho phép 2 phiên GPU cùng lúc; nếu không thì chạy sau K1.
- **Checkpoint:**
  - Sau **mỗi epoch**, notebook lưu đủ trạng thái huấn luyện vào `state/inprogress_*.pt`.
  - Bị ngắt giữa chừng thì **Run All lại trong cùng phiên**: notebook tiếp tục đúng epoch đó (đã kiểm chứng, kết quả trùng khớp với chạy liền mạch).
  - Checkpoint **tốt nhất** (theo dev) của cả 6 hệ thống, seed đầu, nằm ở `checkpoints/*_best.pt`.
- **Sang phiên mới** (ví dụ phiên cũ hết giờ):
  - Mở lại notebook → **Add Input → Notebooks → Your Work** → chọn **output phiên bản trước** của chính notebook này → Run All.
  - Notebook tự khôi phục và **bỏ qua** các biến thể/fold đã xong.
- `MAX_HOURS = 9`: notebook không bắt đầu lượt huấn luyện mới sau 9 giờ, để kết thúc êm và kịp lưu Output trước giới hạn phiên của Kaggle.

## LƯỢT K3: `04_qwen7b_test_locked.ipynb` (Qwen trên test + đổi tên; GPU; ước tính 40–80 phút)
**Chỉ chạy khi trợ lý báo "đã khoá cấu hình" sau khi xem kết quả K1.** Nếu K1 cho thấy phải sửa prompt, trợ lý sẽ gửi bản notebook mới.
Cách chạy giống K1 (GPU T4 x2, Internet On, Save & Run All (Commit)). Gửi lại: khối `=== GỬI LẠI ===` + file `qwen7b_test_v1.zip`.

---

## (Tuỳ chọn) Chạy 00 / 01 trên Kaggle
Không bắt buộc, vì trợ lý đã chạy và kiểm chứng cục bộ (kết quả trong `results/`). Nếu muốn có bằng chứng chạy trên Kaggle:
- Import notebook → **Accelerator: None** → Internet On → Run All.
- Ô cuối phải in `GATE_STATUS=PASS`.
- Không có Internet: tạo Dataset riêng tư từ `kaggle_upload/propara-raw-v1.zip` rồi Add Input.

## Sửa mã (người phát triển)
```bash
cd src && python -m pytest -q propara_core_tests.py propara_metrics_tests.py propara_llm_tests.py   # 46 kiểm thử
python -c "import propara_pp4_tests as t; print(t.run_all())"   # 6 kiểm thử CRF + xác suất tồn tại (cần torch)
python tools/build_nb00.py . ; python tools/build_nb01.py . ; python tools/build_nb03.py . ; python tools/build_nb04.py . ; python tools/build_nb05.py .
python tools/final_analysis.py --data <raw ProPara> --k1 <qwen7b_dev_v1> --k2 <pp34_v2> --k3 <qwen7b_test_v1> --out results/final
```
Chạy thử ngoài Kaggle: đặt `PROPARA_INPUT_ROOT`, `PROPARA_WORK_ROOT`. Riêng 03 có thêm các biến chỉ dùng khi kiểm thử cục bộ: `PROPARA_EXPECT_7B=0`, `PROPARA_LIMIT_PARAS`, `PROPARA_RUN_FULL=force`.
