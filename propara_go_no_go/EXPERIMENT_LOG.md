# EXPERIMENT_LOG — Go/No-Go ProPara

> Gửi file này khi bắt đầu cuộc trò chuyện mới để tiếp tục từ tiến độ đã ghi. Kế hoạch lượt Kaggle: `docs/KAGGLE_RUN_PLAN.md`.

## Tiến độ tổng quát (cập nhật 19/10/2026)

| Giai đoạn | Notebook | Trạng thái | Cổng |
|---|---|---|---|
| D0 Đọc + phản biện | `docs/research_audit.md`, `docs/experiment_protocol.md` (v1.1) | ✅ | G0: protocol đã khoá; ngưỡng 5 điểm **cần GV xác nhận** |
| D1 Dữ liệu | `00_propara_data_audit.ipynb` | ✅ PASS cục bộ (Kaggle: tuỳ chọn) | G1-data ✅ |
| D2 Bộ chấm + heuristic | `01_metrics_heuristics.ipynb` | ✅ PASS cục bộ, 0 FAIL / 0 WARN | **Cửa nghiệm thu 1 ✅** |
| D3 Baseline bài báo | ProStruct, ProGlobal, **CGLI** (dự đoán công bố) + heuristic đầu-cuối B1-E2E đã chấm trong 01; MeeT/XPAD: tìm nhanh chưa thấy mã | ✅ | |
| D4 Qwen dev | `03_qwen7b_dev.ipynb` (K1) | ✅ **Đã chạy trên Kaggle** (2×T4, fp16, 40 phút). Kết quả: `results/kaggle/03.txt` | G2 ✅, G3 ✅ |
| D5 PP2/PP3/PP4 | `05_pp3_pp4_dev_cv.ipynb` (K2) | ✅ **Đã chạy Kaggle** (2,13 giờ); zip đã có | G4 ✅ |
| D4/D5 test khoá | `04_qwen7b_test_locked.ipynb` (K3) + `tools/final_analysis.py` | ✅ K3 đã chạy; **toàn bộ test đã chấm một lần** (`results/final/`) | G5 ✅ |
| D6–D8 | 09–10 | ⬜ | G6–G7 |

**Quyết định (20/10, theo protocol v1.5 §7): KEEP có điều chỉnh**: giữ đề tài, thu hẹp đóng góp về PP4, thiết kế lại PP3. Báo cáo: `reports/go_no_go_final.md`; số liệu: `results/final/`.

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
  1. *(Nhận định lúc đó; đã được điều chỉnh ở EXP-002b: CGLI 2022 có bắt được cặp nguồn ngầm 0,452, chỉ yếu ở đích ngầm 0,091.)* Hai hệ thống 2018 **không bắt được cặp ngầm nào** trên test. Đây là bằng chứng ban đầu ủng hộ khoảng trống đề tài nhắm tới (mới là mốc cũ, còn phải xem CGLI/Qwen).
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
- **Trạng thái:** ✅ **ĐÃ CHẠY TRÊN KAGGLE** (19/10, 2×Tesla T4, fp16 chia 2 GPU, torch 2.11, transformers 5.16.1). Nạp mô hình 29 s, smoke 80 s, tổng 2.375 s. Smoke 3/3 JSON hợp lệ → tự chạy dev.
- **Kết quả dev (43 đoạn), nguồn `results/kaggle/03.txt`:**

| Chế độ | PAIR_F1 [CI95] | @nguồn ngầm | @đích ngầm | IMP_ACC D / C | Vi phạm/thực thể | JSON hợp lệ | s/lời gọi (p50) | VRAM đỉnh |
|---|---|---|---|---|---|---|---|---|
| QWEN_DIRECT (Q0) | 0,217 [0,058–0,352] | 0,095 | 0 | 0,07 / 0,09 | 0,669 | 100% | 4,4 | 8,6 GiB |
| QWEN_DIRECT[cặp LLM tự khai] | 0,370 [0,242–0,487] | 0,167 | 0,267 | — | — | — | — | — |
| **QWEN_FEWSHOT (Q1)** | **0,377 [0,243–0,495]** | 0,231 | 0,118 | 0,33 / 0,09 | 0,509 | 100% | 10,0 | 10,2 GiB |
| QWEN_STRUCTURED (Q2) | 0,318 [0,159–0,462] | 0,083 | 0,105 | 0,07 / 0,09 | 0,737 | **81,4%** | 16,3 | 9,2 GiB |
| QWEN_CONSTRAINED (Q3 = Q0 + luật v1) | **0,033** [0–0,095] | 0 | 0 | 0,07 / 0 | 0 | — | — | — |
| *ORACLE_PREV_STATE (Q4)* | *0,425 [0,287–0,557]* | *0,392* | *0,311* | *0,60 / 0,73* | — | 100% | 1,4 | 8,7 GiB |

- Kiểm định cặp: Q1 − Q0 = **+0,160 [+0,046; +0,289]**; Q2 − Q0 = +0,101 [−0,051; +0,273]; Q3 − Q0 = −0,184 [−0,299; −0,054].
- Điểm quyết định ngầm (18 điểm, dev): Q1 trúng 4 (khớp hẳn 3); Q4 trúng 10 (khớp hẳn 7). (So với heuristic nhắc-gần-nhất không công bằng hoàn toàn: heuristic được biết sẵn vị trí điểm quyết định.)
- **Nhận xét:**
  1. Hệ thống Qwen chính (quy tắc v1.2 (e)) = **QWEN_FEWSHOT**.
  2. Qwen yếu ở phần ngầm, đặc biệt CREATE ngầm (9%).
  3. Khi được cho tồn tại đúng trước câu (Q4), độ chính xác ô ngầm tăng mạnh (D 0,33 → 0,60; C 0,09 → 0,73). Phần lớn lỗi nằm ở **theo dõi tồn tại**, đúng chỗ ràng buộc + giải mã chung của PP4 nhắm tới.
  4. Qwen vi phạm ràng buộc nhiều (0,5–0,7/thực thể). Luật hậu xử lý v1 xoá nhãn nên làm sập PAIR_F1 → đăng ký **luật v2** (sửa tồn tại ban đầu trước, `apply_existence_constraints_v2`). Số của v2 sẽ được tính trên dev từ dự đoán K1 (cần file zip), không gọi lại mô hình.
  5. Q3_OFFICIAL của Qwen 0,53–0,55 ≈ mức sàn ALL_NONE 0,542 trên dev.
  6. Q2 chỉ 81,4% JSON hợp lệ; các đoạn hỏng bị tính là toàn NONE.
- **Quyết định (khoá):** giữ prompt_v1, không chỉnh prompt; Qwen chính = QWEN_FEWSHOT; K3 thêm Qwen chính chạy train (CV 486 đoạn) + hậu xử lý v2.
- **Kiểm chứng output K1 (20/10 ~00:40, `results/kaggle/notebook 03/qwen7b_dev_v1/`):**
  - Có **2 lần chạy K1**: `03.txt` (nạp 28,6 s) và zip (nạp 31,0 s); thời gian khác nhau nhưng **điểm giống hệt** (giải mã tham lam).
  - Tính lại từ đầu ra thô bằng mã hiện tại: Q0 0,2169 · Q1 0,3774 · Q2 0,3182 · oracle 0,4247, **trùng tuyệt đối** với file điểm; nhãn trùng 43/43 đoạn ở cả 3 chế độ. Config ghi `prompt_v1`, `llm_v1`, `core_v1`, `metrics_v1`, revision `a09a354`.
- **Hậu xử lý v2 trên dev (đăng ký trước khi có số):**

| Hệ thống | PAIR_F1 | @nguồn ngầm | @đích ngầm |
|---|---|---|---|
| Q0 + v1 | 0,033 | 0 | 0 |
| Q0 + v2 | 0,179 | 0,095 | 0 |
| Q1 + v1 | 0,167 | 0 | 0 |
| Q1 + v2 | 0,292 | 0,087 | 0 |

  → v2 tốt hơn v1 nhưng **vẫn thấp hơn Q1 không hậu xử lý (0,377)**: ép nhất quán bằng luật đơn giản làm giảm điểm.
- **Phân tích lỗi:**
  - Q0: 109/117 vi phạm là “CREATE khi đang tồn tại”; 97/175 thực thể có sự kiện đầu là CREATE nhưng Qwen khai có vị trí ban đầu.
  - Q1 (có ví dụ): giảm còn 13/175, nhưng vi phạm chuyển sang DESTROY/MOVE khi chưa tồn tại (33 + 33).
  - Q1 trên ô ngầm vàng: DESTROY đúng 5/15 (10 bị bỏ thành NONE); CREATE đúng 1/11 (9 bị bỏ thành NONE).
  - Q2: 8/43 JSON hỏng; 7 do **thiếu dấu đóng `]`/`}`** (lỗi của Qwen, đã kiểm `json.loads`), 1 do chạm giới hạn token. Bộ đọc JSON hiện chỉ sửa lỗi nhỏ (dấu phẩy thừa, nháy đơn, True/False); bộ đọc dễ dãi hơn có thể cứu một phần, ghi là giới hạn.
  - **Phân tích độ nhạy (20/10 ~00:55, không cần GPU):** thử chèn dấu đóng còn thiếu trước khoá cấp cao → cứu được 6/8 đầu ra; nhưng cả 6 đều **không dự đoán cặp D×C nào** (0 cặp dự đoán) → PAIR_F1 của Q2 **không đổi** (0,318 → 0,318). Kết luận: lỗi JSON không ảnh hưởng thước đo chính; **không cần chạy lại K1**. Giữ bộ đọc gốc làm chuẩn.
  - Tên thực thể không khớp danh sách: ít (ví dụ “hormones”, “turbine”).
- **Điều kiện hoàn thành:** smoke đạt; dev chạy đủ 4 chế độ; JSON hợp lệ ≥ 90% ở Q0 (nếu thấp hơn thì sửa prompt **trên dev** → prompt_v2 và ghi log).
- **Bước tiếp sau K1:** phân tích lỗi Qwen trên dev → khoá prompt → viết K2 (PP4/M0–M3, CV) → K3 (test một lần).

## EXP-004: Mô hình nhóm PP2/PP3/PP4 + ablation + CV (D5) · `05_pp3_pp4_dev_cv.ipynb` · pp3_v2, pp4_v1, robust_v1 · lượt **K2**
- **Giả thuyết:** H1 (PP2 với tồn tại dự đoán, đo theo công thức (11) bằng xác suất + ngưỡng τ1/τ2, xem EXP-004b), H2 (E(X,Y) giúp lát ngầm: M3 vs M3-no-E vs M2r), H3 (M3 vs M0c trên lát ngầm, CV).
- **Thiết lập (khoá, protocol v1.2 f–k):**
  - PP4 = RoBERTa-base + BiLSTM + φ (MLP) + ψ = wᵀ[log E, m_i, m_j, d_i, cos, 1].
  - CRF **chính xác** trên 2^n trạng thái tồn tại (n ≤ 8).
  - Mất mát −log p(y_vàng)/số ô.
  - 6 hệ thống × 3 seed; CV 5 phần (M0c, M3).
  - Test và NONCE-test lưu kín.
- **Kiểm chứng đã làm (cục bộ):**
  - CRF: log Z và Viterbi **khớp liệt kê vét cạn** (5/5 kiểm thử lúc đó; nay 6/6 sau khi thêm kiểm thử xác suất tồn tại). Đoạn lớn nhất (n=8, T=10): nll+backward 0,06 s trên GPU laptop RTX A2000 (chưa đo trên Kaggle).
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
- **Trạng thái:** ✅ đã chạy Kaggle.

### KẾT QUẢ K2 (Kaggle 2×T4, 2,13 giờ, `results/kaggle/notebook 05/05.txt`; zip `pp34_v2.zip` CHƯA có)
- Hoàn thành 15/15 lượt (5 biến thể × 3 seed) + CV 10/10 fold; dừng sớm hoạt động (epoch tốt nhất 4–8).
- **PP3 (WikiHow):** 3.683 cặp khác nhau (4.304 lần khớp mẫu). Độ phủ trên 533 cặp vàng: **cặp 6,9%**, nguồn 61,9%, đích 58,7%, cùng danh từ đầu 9,0%.
  - E(X,Y) trên validation WikiHow: AUC 0,692, AP 0,496, AUC so với cặp đảo chiều 0,768. Epoch tốt nhất = 30 = epoch tối đa, nên có thể chưa hội tụ.
  - AUC của log E trên cặp ProPara (dev): **0,612**.
- **Ablation trên DEV** (trung bình ± SD qua 3 seed):

| Hệ thống | PAIR_F1 | @có phía ngầm | PAIR_LOC_F1 | IMP_ACC D / C | Vi phạm |
|---|---|---|---|---|---|
| M0 độc lập | 0,449 ± 0,040 | 0,030 | 0,241 | 0,022 / 0 | 0,291 |
| M0c = M0 + luật | 0,467 ± 0,032 | 0 | 0,276 | 0 / 0 | 0 |
| M3-no-pair (CRF) | 0,495 ± 0,026 | 0 | 0,311 | 0 / 0,030 | 0 |
| M3-no-E | **0,519 ± 0,009** | 0,027 | 0,298 | 0 / 0,030 | 0 |
| M2r (E xáo trộn) | 0,506 ± 0,006 | 0 | 0,286 | 0 / 0 | 0 |
| **M3 (đầy đủ)** | 0,508 ± 0,005 | 0 | 0,281 | 0 / 0 | 0 |

- Δ trên dev (TB 3 seed; CI của seed đầu): M3 − M0c +0,041 [−0,054; +0,124] · M3 − M3-no-E −0,011 [−0,024; +0,012] · M3 − M2r +0,003 [−0,057; +0,056] · M3-no-E − M3-no-pair +0,024 · M3-no-pair − M0c +0,028 [−0,027; +0,132] · M0c − M0 +0,018 [−0,058; +0,078]. **Không chênh lệch nào trên dev có ý nghĩa thống kê.**
- **PP2 (τ1 = 0,1, τ2 = 0,9, dev):** M0c giữ 52,6% ứng viên, phủ 82,8% cặp vàng; M3 giữ 62,1%, phủ 93,1% → **H1 CHƯA ĐẠT** (tiêu chí: phủ ≥ 95% và giữ ≤ 50%).
- **Đổi tên NONCE (dev):** M0 0,449 → 0,324 · M0c 0,467 → 0,381 · M3-no-pair 0,495 → 0,396 · M3-no-E 0,519 → 0,406 · M3 0,508 → 0,421 (giảm 0,09–0,13).
- **CV 5 phần theo chủ đề (seed 13, 488 đoạn):** M0 0,469 (@ngầm 0,058) · M0c 0,472 (@ngầm 0,030) · **M3 0,524** (@nguồn ngầm 0,107, @đích ngầm 0,057, @có phía ngầm 0,096).
  - **ΔM3 − M0c (bootstrap theo đoạn):** tổng **+0,051 [+0,020; +0,085]** · nguồn ngầm **+0,075 [+0,007; +0,145]** · đích ngầm +0,036 [−0,004; +0,093] · có phía ngầm **+0,065 [+0,015; +0,116]**.
- **Nhận xét K2:**
  1. Giải mã chung (PP4) có lợi **có ý nghĩa** so với cùng mô hình cộng luật (M0c) trên CV, kể cả lát ngầm → **H3 được ủng hộ** (về hướng). Nhưng mức tuyệt đối ở lát ngầm vẫn **rất thấp** (0,096).
  2. **PP3 không cho thấy đóng góp:** M3 ≈ M3-no-E ≈ M2r (E xáo trộn); độ phủ cặp WikiHow chỉ 6,9% → **H2 không được ủng hộ** với nguồn WikiHow và cách trích hiện tại.
  3. H1 (PP2) chưa đạt ở điểm vận hành đã khoá.

## EXP-004b: Tự đối chiếu với báo cáo + sửa 2 thiếu sót trước K2 (19/10)
- **Đối chiếu:** `docs/DOI_CHIEU_BAO_CAO.md`, kiểm từng mục §1–§5 và công thức (1)–(28).
  - Kiểm tra mới: E0 63% khi gộp 3 nhãn (BC 59–64%) ⚠️ (theo từng nhãn thì không khớp); đại từ 14–27% trên train ✅; CI (5) 0,1139 ✅.
  - PP1 (8)(9) không tái hiện được (phụ thuộc bộ tách cụm danh từ) ⚠️.
- **Sửa (pp4_v1.2, protocol v1.4):**
  - (1) đầu dự đoán vị trí trên cụm danh từ: phủ 88–89% vị trí vàng; học thuộc đạt PAIR_LOC_F1 = 1,0;
  - (2) PP2 theo xác suất tồn tại (biên CRF khớp vét cạn, 6/6 kiểm thử CRF) + lưới τ1, τ2.
  - Notebook K2 chạy thông cục bộ.
- **Còn thiếu (cần quyết định):** Qwen trong CV (thêm ở K3); tập test thứ hai CUTL/NHK (cần dữ liệu); chưng cất LLM, PP1 (P2); rà tính mới PP3.

## EXP-004c: Rà K2 theo môi trường Kaggle thật (20/10 ~00:15, pp4_v1.3)
- Môi trường đọc từ K1: 2×T4, torch 2.11.0+cu128, transformers 5.16.1.
- Sửa: (1) checkpoint theo epoch không còn kèm bản tốt nhất (trước ghi khoảng 2,5 GB/epoch); bản tốt nhất lưu riêng, chỉ ghi khi cải thiện; (2) tự cài NLTK nếu thiếu.
- Kiểm chứng: ngắt sau epoch 2 rồi tiếp tục → loss **và trọng số bản tốt nhất** trùng khớp với chạy liền mạch (M0/M0c, CRF_NOE); notebook K2 chạy thông, 0 ô lỗi, không còn file tạm.
- Ước tính thời gian K2 sửa thành 3–5 giờ (chưa đo).

## EXP-005: Qwen test khoá + NONCE (D6) · `04_qwen7b_test_locked.ipynb` · lượt **K3**
- **Thiết lập:** dùng nguyên văn mã nạp mô hình/sinh của NB03; test ở mọi chế độ; NONCE (DIRECT, FEWSHOT) trên dev (chấm công khai) và test (lưu kín).
- **Kiểm chứng:** chạy thông cục bộ với mô hình tí hon (lưu kín đủ 9 file).
- **Trạng thái:** ✅ đã chạy Kaggle.

### KẾT QUẢ K3 (Kaggle 2×T4, 2,03 giờ, `results/kaggle/notebook 04/`) + CHẤM TEST MỘT LẦN (phần Qwen: `results/final_partial/`; bản đầy đủ: `results/final/`)
- JSON hợp lệ: test Q0 98,1%, Q1 100%, Q2 92,6%; train Q1 100%; NONCE Q1 100%.
- **TEST (54 đoạn), cùng bộ chấm:**

| Hệ thống | PAIR_F1 [CI95] | @nguồn ngầm | @đích ngầm | IMP_ACC D / C | Vi phạm |
|---|---|---|---|---|---|
| CGLI 2022 (công bố) | **0,545** [0,466–0,628] | 0,452 | 0,091 | 0,41 / 0,20 | 0 |
| **Qwen Q1 có ví dụ (chính)** | **0,411** [0,301–0,521] | 0,383 | **0** | 0,64 / 0,07 | 0,41 |
| Qwen Q0 trực tiếp | 0,350 [0,238–0,457] | 0,077 | 0 | 0,14 / 0,13 | 0,70 |
| Qwen Q2 từng bước | 0,254 [0,149–0,359] | 0,051 | 0,051 | 0,14 / 0,33 | 0,79 |
| Qwen Q1 + hậu xử lý v1 | 0,387 | 0,429 | 0 | — | 0 |
| Qwen Q1 + hậu xử lý v2 | 0,398 | 0,400 | 0 | — | 0 |
| Qwen Q0 + hậu xử lý v1 / v2 | 0,020 / 0,360 | — | — | — | 0 |
| *ORACLE (biết tồn tại đúng trước câu)* | *0,386* | *0,165* | *0,149* | *0,50 / 0,53* | — |

- **Qwen chính − CGLI (test):** tổng **−0,134 [−0,253; −0,014]** (CGLI hơn, có ý nghĩa) · có phía ngầm −0,022 [−0,226; +0,199] · đích ngầm −0,091 [−0,235; 0].
- **Đổi tên NONCE:** test Q1 0,411 → 0,357 (−0,054), Q0 0,350 → 0,342; dev Q1 0,377 → 0,208 (−0,169), Q0 0,217 → 0,308 (+0,091).
- **Qwen Q1 trên 486 đoạn gộp** (train K3 + dev K1 + test K3, bỏ 2 đoạn few-shot): PAIR_F1 **0,438** [0,389–0,484] · @nguồn ngầm 0,315 · @đích ngầm 0,110 · **@có phía ngầm 0,258** · IMP_ACC D 0,43 / C 0,22.
  - So sánh điểm (CHƯA có bootstrap cặp vì thiếu zip K2) với M3 CV: PAIR_F1 0,438 (Qwen) so với 0,524 (M3); @có phía ngầm **0,258 (Qwen) so với 0,096 (M3)**.
- **Chưa làm được (thiếu `pp34_v2.zip`):** chấm test của M3/M0c; M3 so với CGLI, Qwen so với M3 (test + CV 486 đoạn, bootstrap cặp).

- **Sau K3:** `tools/final_analysis.py` chấm một lần mọi dự đoán test kín, bootstrap cặp Qwen-chính vs M3 và M3 vs M0c, rồi áp điều kiện quyết định §7.

## Việc song song cho nhóm (không cần GPU)
- Hai người chấm độc lập `results/00_local_cpu_run/gold_audit_sample.csv` (nay là `annotation/gold_audit_A.xlsx`, `gold_audit_B.xlsx`; 4 nhãn theo PIPELINE §5.5), không xem dự đoán mô hình. Bước T16 (P1), không bắt buộc cho Go/No-Go.
- Hỏi giảng viên: (1) ngưỡng 5 điểm PAIR_F1; (2) đồng ý PAIR_F1 là thước đo chính, Q3_OFFICIAL để so tài liệu.


## Đánh giá tạm theo 3 cửa (20/10, CHƯA phải quyết định cuối)
- **Cửa 1, bằng chứng đáng tin:** gần đủ (dữ liệu, bộ chấm, heuristic, 3 baseline, Qwen dev + test, mô hình nhóm dev + CV). Còn thiếu: test của M3 và các so sánh cặp (cần `pp34_v2.zip`).
- **Cửa 2, Qwen vượt trội rõ? → KHÔNG.**
  - Trên test, Qwen chính **thua CGLI** (−0,134, CI không chứa 0).
  - So theo điểm số (chưa có bootstrap cặp), Qwen thấp hơn M3 ở PAIR_F1 tổng (CV: 0,438 so với 0,524; dev: 0,377 so với 0,508).
  - Nhưng **Qwen hơn hẳn M3 ở lát ngầm** (CV: 0,258 so với 0,096).
- **Cửa 3, PP2/PP3/PP4 còn đóng góp?**
  - **PP4 (giải mã chung): CÓ**, trên CV: +0,051 tổng, +0,065 lát ngầm so với M0c, có ý nghĩa.
  - **PP3 (WikiHow): KHÔNG thấy** (M3 ≈ M3-no-E ≈ E xáo trộn; phủ cặp 6,9%).
  - **PP2:** chưa đạt H1.
- **Theo ma trận đã khoá:**
  - Dòng “PP3 không tăng chất lượng nhưng PP4 vẫn có ích” → **ĐIỀU CHỈNH**.
  - Điều kiện KEEP_a (M3 > M0c có ý nghĩa ở lát ngầm, CV) đạt, nhưng mức tuyệt đối ở lát ngầm còn rất thấp.
- **Hướng điều chỉnh có cơ sở dữ liệu** (đề xuất, chưa kiểm chứng):
  1. Lỗi lớn nhất của cả Qwen lẫn M3 là thay đổi ngầm, mà oracle cho thấy là do **theo dõi tồn tại**.
  2. Qwen giỏi ngầm hơn, còn M3 giỏi phần tường minh và luôn nhất quán. Gợi ý thử: **dùng đề xuất của LLM (ví dụ điểm của Qwen cho cặp/nhãn ngầm) làm đặc trưng ψ hoặc φ trong CRF**, thay cho E(X,Y) từ WikiHow.
  3. Hoặc thay WikiHow bằng nguồn tri thức khoa học (như báo cáo đã dự phòng ở mục 4.5).


## EXP-006: Phân tích cuối + quyết định Go/No-Go (D7–D8) · 20/10 · `tools/final_analysis.py`
- **Đầu vào:** K1 `results/kaggle/notebook 03/qwen7b_dev_v1/qwen7b_dev_v1`, K2 `notebook 05/pp34_v2/pp34_v2`, K3 `notebook 04/qwen7b_test_v1/qwen7b_test_v1`.
  - K2: WikiHow 215.365 dòng, 5.410.591 câu, SHA-256 khớp; bộ trích pp3_v2 (NLTK).
- **Test** (TB 3 seed với mô hình nhóm): CGLI 0,545 · M3 0,497 ± 0,024 · M3 không E 0,493 · M0c 0,483 · M0 0,469 · Qwen có ví dụ 0,411 · Qwen trực tiếp 0,350.
  - M3 − CGLI −0,048 (0/3 seed có ý nghĩa); ở cặp ngầm −0,203 (2/3 seed có ý nghĩa, CGLI hơn).
  - Qwen − CGLI −0,134 [−0,253; −0,014].
- **CV 486–488 đoạn:** M3 0,524 · M0c 0,474 · Qwen 0,438.
  - M3 − M0c +0,050 [+0,018; +0,082]; ở cặp ngầm +0,066 [+0,015; +0,117].
  - Qwen − M3 −0,086 [−0,136; −0,038]; ở cặp ngầm +0,162 [+0,081; +0,242].
- **Đổi tên (test):** M3 0,497 → 0,489; Qwen có ví dụ 0,411 → 0,357; cặp ngầm của Qwen 0,286 → 0,070.
- **Điều kiện quyết định:** KEEP_a ĐÚNG; STOP_1 SAI; STOP_2 SAI. Cộng thêm dòng “PP3 vô ích nhưng PP4 có ích” → **KEEP có điều chỉnh**.
- **Thiếu (ghi là giới hạn):** chi phí của M3; tập test thứ hai; kiểm tra nhãn vàng bởi người; CGLI trên dev/CV.
