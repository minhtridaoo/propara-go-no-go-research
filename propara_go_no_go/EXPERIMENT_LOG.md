# EXPERIMENT_LOG — Go/No-Go ProPara

> Gửi file này khi bắt đầu cuộc trò chuyện mới để tiếp tục từ tiến độ đã ghi. Kế hoạch lượt Kaggle: `docs/KAGGLE_RUN_PLAN.md`.

## Tiến độ tổng quát (cập nhật 19/10/2026)

| Giai đoạn | Notebook | Trạng thái | Cổng |
|---|---|---|---|
| D0 Đọc + phản biện | `docs/research_audit.md`, `docs/experiment_protocol.md` (v1.1) | ✅ | G0: protocol đã khoá; ngưỡng 5 điểm **cần GV xác nhận** |
| D1 Dữ liệu | `00_propara_data_audit.ipynb` | ✅ PASS cục bộ (Kaggle: tuỳ chọn) | G1-data ✅ |
| D2 Bộ chấm + heuristic | `01_metrics_heuristics.ipynb` | ✅ PASS cục bộ, 0 FAIL / 0 WARN | **Cửa nghiệm thu 1 ✅** |
| D3 Baseline bài báo | ProStruct/ProGlobal (dự đoán công bố) đã chấm trong 01; CGLI/MeeT ⬜ | 🟡 | |
| D4 Qwen dev | `03_qwen7b_dev.ipynb` (K1) | 🟡 Luồng mã kiểm thử cục bộ bằng mô hình tí hon: OK. **Chờ bạn chạy trên Kaggle** | G2, G3 |
| D5 PP2/PP3/PP4 | `05_pp3_pp4_dev_cv.ipynb` (K2) | 🟡 mã xong + kiểm thử cục bộ OK. **Chờ bạn chạy Kaggle** | G4 |
| D4/D5 test khoá | `04_qwen7b_test_locked.ipynb` (K3) + `tools/final_analysis.py` | 🟡 mã xong + kiểm thử cục bộ OK; chạy sau khi khoá từ K1 | G5 |
| D6–D8 | 09–10 | ⬜ | G6–G7 |

**Quyết định hiện tại: INCONCLUSIVE** (chưa có điểm Qwen, chưa có mô hình nhóm).

---

## EXP-000: Kiểm tra nghiên cứu (D0)
- **Kết quả:** 14/14 nhóm số liệu dữ liệu tái hiện được. Phát hiện V1–V8 (xem `research_audit.md`):
  - **V1:** Q3 chính thức ≠ công thức (27) của báo cáo (481 conversion vàng thay vì 533; Jaccard mềm; macro theo đoạn).
  - **V2:** với vàng D×C, cặp là hàm của nhãn C/D. Oracle “cho trạng thái vàng” vô nghĩa nên thay bằng `ORACLE_PREV_STATE`.
  - **V3:** test chỉ có 17 cặp nguồn ngầm và 10/12 điểm quyết định nên kết luận về ngầm dựa trên CV 5-fold theo chủ đề.
- **Quyết định:** `PAIR_F1` là thước đo chính, `Q3_OFFICIAL` là phụ (mã gốc); khoá protocol.

## EXP-001: Kiểm tra dữ liệu (D1) · `00_propara_data_audit.ipynb` · core_v1
- **Trạng thái:** ✅ cục bộ (CPU) cả 3 chế độ: Input / tự tải / SMOKE. `GATE_STATUS=PASS`, FAIL=0, WARN=2 (S04 thực thể trùng tên, D01 Q3≠533; đều đã biết).
- **Kết quả:** 488/3302/1915; Bảng 1 9/9 + 9/9; 533=378/58/97; dp 57/76; lift 9/9; PP2 11144/533; answers.tsv khớp; CV5 theo chủ đề 95–99 đoạn/fold.

## EXP-002: Bộ chấm + heuristic (D2) · `01_metrics_heuristics.ipynb` · metrics_v1, heur_v1
- **Giả thuyết:** bộ chấm đúng (tái hiện điểm chính thức) và heuristic tái hiện Bảng 3 báo cáo.
- **Thiết lập:** CPU cục bộ; evaluator gốc `aristo-leaderboard@60d5be3` được nhúng nguyên văn; 16+17 kiểm thử tự viết + 17 kiểm thử gốc; bootstrap 10.000 lần theo đoạn, seed 2026.
- **Kết quả (Cửa nghiệm thu 1 ✅, 0 FAIL / 0 WARN):**
  - Hồi quy evaluator: ProStruct test **0,545**, ProGlobal test **0,519**, testfiles-3 0,686, file sai bị từ chối (6/6).
  - Vàng: PAIR_F1 = 1 (cả 3 tập); Q3_OFFICIAL = 1 (dev/test), 0,998 (train, do thực thể trùng tên).
  - Bảng 3 tái hiện (TB 20 seed): ngẫu nhiên 50,1/40,1 (BC 51/40); ngẫu nhiên-không-nhắc 59,6/41,4 (BC 59/42); nhắc-gần-nhất **75,1/32,3** (BC 74/33).
  - Hệ thống tham chiếu trên **test** (PAIR_F1 [CI95] · @any_implicit · Q3conv):
    - ProStruct 2018: 0,252 [0,147–0,370] · **0,000** · 0,326
    - ProGlobal 2018: 0,127 [0,056–0,203] · **0,000** · 0,391
    - ALL_NONE: 0,000 · 0,000 · **0,312** (mức sàn do trường hợp biên macro của Q3 chính thức)
    - Chẩn đoán (dùng vàng) ORACLE_MENTIONED_ONLY: 0,802 · 0,000. Thêm heuristic nhắc-gần-nhất cho ô ngầm: 0,815 · 0,385.
  - ProStruct − ProGlobal: ΔPAIR_F1 = +0,125, CI95 [+0,023; +0,240].
- **Nhận xét:**
  1. Hai hệ thống 2018 **không bắt được cặp ngầm nào** trên test. Đây là bằng chứng ban đầu ủng hộ khoảng trống đề tài nhắm tới (mới là mốc cũ, còn phải xem CGLI/Qwen).
  2. Q3_OFFICIAL Conversions có sàn 0,31 cho hệ thống rỗng, nên không phân biệt tốt (ProStruct 0,326 ≈ ALL_NONE). Điều này củng cố việc chọn PAIR_F1 làm thước đo chính.
  3. CI trên test rộng ±9–11 điểm, nên so sánh phải dùng bootstrap cặp + CV.
- **Bước tiếp:** EXP-003.

## EXP-002b: Bổ sung đối chứng B1-E2E + CGLI (19/10, CPU, NB01 chạy lại PASS)
- **B1-E2E** (heuristic đầu-cuối, luật viết một lần, không chỉnh theo điểm): test PAIR_F1 0,222 [0,109–0,344], @any_imp 0,104; dev 0,088.
- **CGLI 2022** (file dự đoán test công bố, Mayer123/CGLI@9c910de): PAIR_F1 **0,545** [0,466–0,628]; @src_imp 0,452; @tgt_imp 0,091; IMP_ACC DESTROY 0,41 / CREATE 0,20; Q3_OFFICIAL tổng 0,724; vi phạm tồn tại 0.
- **Điều chỉnh nhận định:** khoảng trống chính = CREATE/đích ngầm. Bước tiếp phải so M3 và Qwen **với CGLI**, không chỉ với các hệ thống 2018. `final_analysis.py` đã thêm các so sánh M3−CGLI và Qwen−CGLI.
- MeeT, XPAD: mới chỉ **tìm nhanh bằng GitHub search API** và không ra kết quả (chưa tìm kỹ ở trang bài báo / Papers with Code), nên chỉ trích số bài báo kèm ghi chú “không so trực tiếp”. ProLocal: tên thực thể không khớp đáp án, bỏ qua.

## EXP-003: Qwen2.5-7B-Instruct trên dev (D4) · `03_qwen7b_dev.ipynb` · llm_v1 / prompt_v1 · lượt **K1**
- **Giả thuyết:** H4. Qwen 7B không huấn luyện đạt PAIR_F1 cạnh tranh; đo riêng lát ngầm.
- **Thiết lập (đã khoá trước khi chạy):** `Qwen/Qwen2.5-7B-Instruct@a09a354`; fp16 trên 2×T4 (hoặc nf4 nếu 1 GPU); greedy, repetition_penalty=1.0; 4 chế độ + Q3 hậu xử lý; few-shot train 294, 1156; smoke 3 đoạn train → tự chạy dev nếu Q0 ≥ 2/3 JSON hợp lệ.
- **Trạng thái:** ✅ luồng mã đã kiểm thử cục bộ (mô hình Qwen2 tí hon trọng số ngẫu nhiên + tokenizer thật; smoke tự đánh giá KHÔNG ĐẠT đúng như kỳ vọng; chế độ ép chạy đầy đủ cũng chạy hết, lưu đủ). Độ dài prompt đo bằng tokenizer thật: Q0 ~480, few-shot ~1.700 token. ⏳ **Chờ chạy Kaggle.**
- **Điều kiện hoàn thành:** smoke đạt; dev chạy đủ 4 chế độ; JSON hợp lệ ≥ 90% ở Q0 (nếu thấp hơn thì sửa prompt **trên dev** → prompt_v2 và ghi log).
- **Bước tiếp sau K1:** phân tích lỗi Qwen trên dev → khoá prompt → viết K2 (PP4/M0–M3, CV) → K3 (test một lần).

## EXP-004: Mô hình nhóm PP2/PP3/PP4 + ablation + CV (D5) · `05_pp3_pp4_dev_cv.ipynb` · pp3_v2, pp4_v1, robust_v1 · lượt **K2**
- **Giả thuyết:** H1 (PP2 với tồn tại dự đoán), H2 (E(X,Y) giúp lát ngầm: M3 vs M3-no-E vs M2r), H3 (M3 vs M0c trên lát ngầm, CV).
- **Thiết lập (khoá, protocol v1.2 f–k):**
  - PP4 = RoBERTa-base + BiLSTM + φ (MLP) + ψ = wᵀ[log E, m_i, m_j, d_i, cos, 1].
  - CRF **chính xác** trên 2^n trạng thái tồn tại (n ≤ 8).
  - Mất mát −log p(y_vàng)/số ô.
  - 6 hệ thống × 3 seed; CV 5 phần (M0c, M3).
  - Test và NONCE-test lưu kín.
- **Kiểm chứng đã làm (cục bộ):**
  - CRF: log Z và Viterbi **khớp liệt kê vét cạn** (5/5 kiểm thử). Đoạn lớn nhất (n=8, T=10): nll+backward 0,06 s trên GPU laptop RTX A2000 (chưa đo trên Kaggle).
  - Phép thử học thuộc 8 đoạn: loss 1,36 → 0,002, PAIR_F1 = 1,0 (M0 và CRF).
  - Toàn notebook chạy hết với encoder tí hon + 5 MB WikiHow thật.
  - Trích cặp WikiHow bản regex (pp3_v1) cho nhiều cặp rác, nên đổi sang **pp3_v2 theo từ loại** (NLTK). Thời gian trích toàn bộ WikiHow **ngoại suy** từ mẫu 5 MB (0,6 s) ra khoảng 1–2 phút, chưa đo trên toàn bộ.
  - NONCE giữ nguyên nhãn 43/43 dev, 54/54 test.
- **Bổ sung 19/10 (pp4_v1.1, protocol v1.3):**
  - Checkpoint theo epoch (mô hình + optimizer + scheduler + scaler + RNG).
  - Dừng sớm, patience 3.
  - Khôi phục sang phiên mới qua Add Input output phiên trước.
  - Checkpoint tốt nhất của cả 6 hệ thống.
  - Đã kiểm chứng: ngắt sau epoch 2 rồi tiếp tục cho loss trùng khớp 6 chữ số với chạy liền mạch; trong notebook thật, ngắt giữa chừng thì Run All lại tiếp tục đúng epoch; phiên mới khôi phục 25 file và bỏ qua mọi phần đã xong.
- **Trạng thái:** ⏳ chờ chạy Kaggle.

## EXP-004b: Tự đối chiếu với báo cáo + sửa 2 thiếu sót trước K2 (19/10)
- **Đối chiếu:** `docs/DOI_CHIEU_BAO_CAO.md`, kiểm từng mục §1–§5 và công thức (1)–(28).
  - Kiểm tra mới: E0 63% khi gộp 3 nhãn (BC 59–64%) ⚠️ (theo từng nhãn thì không khớp); đại từ 14–27% trên train ✅; CI (5) 0,1139 ✅.
  - PP1 (8)(9) không tái hiện được (phụ thuộc bộ tách cụm danh từ) ⚠️.
- **Sửa (pp4_v1.2, protocol v1.4):**
  - (1) đầu dự đoán vị trí trên cụm danh từ: phủ 88–89% vị trí vàng; học thuộc đạt PAIR_LOC_F1 = 1,0;
  - (2) PP2 theo xác suất tồn tại (biên CRF khớp vét cạn, 6/6 kiểm thử CRF) + lưới τ1, τ2.
  - Notebook K2 chạy thông cục bộ.
- **Còn thiếu (cần quyết định):** Qwen trong CV (thêm ở K3); tập test thứ hai CUTL/NHK (cần dữ liệu); chưng cất LLM, PP1 (P2); rà tính mới PP3.

## EXP-005: Qwen test khoá + NONCE (D6) · `04_qwen7b_test_locked.ipynb` · lượt **K3**
- **Thiết lập:** dùng nguyên văn mã nạp mô hình/sinh của NB03; test ở mọi chế độ; NONCE (DIRECT, FEWSHOT) trên dev (chấm công khai) và test (lưu kín).
- **Kiểm chứng:** chạy thông cục bộ với mô hình tí hon (lưu kín đủ 9 file).
- **Trạng thái:** ⏳ chạy sau khi khoá cấu hình từ K1.
- **Sau K3:** `tools/final_analysis.py` chấm một lần mọi dự đoán test kín, bootstrap cặp Qwen-chính vs M3 và M3 vs M0c, rồi áp điều kiện quyết định §7.

## Việc song song cho nhóm (không cần GPU)
- Hai người chấm độc lập `results/00_local_cpu_run/gold_audit_sample.csv` (`CAUSAL` / `PROPARA_ONLY` / `UNCLEAR` / `LIKELY_WRONG`), không xem dự đoán mô hình.
- Hỏi giảng viên: (1) ngưỡng 5 điểm PAIR_F1; (2) đồng ý PAIR_F1 là thước đo chính, Q3_OFFICIAL để so tài liệu.
