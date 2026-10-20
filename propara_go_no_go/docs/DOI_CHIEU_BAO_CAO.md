# ĐỐI CHIẾU PIPELINE VỚI BÁO CÁO GỐC (bao_cao_chi_tiet (1).pdf): làm đúng, làm khác, còn thiếu

**Ngày:** 19/10/2026 · **Người kiểm tra:** trợ lý (tự kiểm tra lại công việc của mình) · **Phạm vi:** mọi mục 1–5 và công thức (1)–(28) của báo cáo, đối chiếu với mã trong `src/` và notebook.

**Ký hiệu:**
- ✅ **ĐÚNG**: làm theo báo cáo và **đã kiểm chứng** (con số tái hiện hoặc có kiểm thử).
- ⚠️ **KHÁC**: làm khác báo cáo; có lý do, đã/ sẽ khai báo trong protocol.
- ❌ **THIẾU**: báo cáo có nhưng pipeline chưa làm.
- 🔧 **ĐANG SỬA**: thiếu sót mình sửa ngay trong đợt này, **trước khi bạn chạy K2**.

---

## A. Kết luận ngắn

| Phần của báo cáo | ✅ Đúng | ⚠️ Khác | ❌ Thiếu | Nhận xét |
|---|---|---|---|---|
| §2 Phân tích dữ liệu (gồm §2.3 heuristic) | 14 | 2 | 0 | Gần như toàn bộ số liệu tái hiện đúng; E0 chỉ khớp khi gộp 3 nhãn |
| §3 Công trình liên quan | 1 | 1 | 1 | CGLI đã chấm; câu “chưa có kho tri thức X→Y” chưa kiểm chứng |
| §4.1 PP1 | 1 | 1 | 1 | PP1 không chạy (kế hoạch xếp ưu tiên thấp) |
| §4.2 PP2 | 3 | 1 | 0 | Đã thêm khảo sát ngưỡng τ1, τ2 (đã sửa) |
| §4.3–4.4 PP4 | 9 | 5 | 0 | Đã thêm đầu dự đoán vị trí (đã sửa); 5 điểm làm khác có khai báo |
| §4.5 PP3 | 8 | 2 | 1 | Thiếu phương án thay nguồn tri thức nếu độ phủ thấp |
| §5 Đánh giá | 6 | 4 | 3 | Thiếu tập test thứ hai (CUTL/NHK), chưng cất LLM; Qwen chưa có trong CV |

*Các con số trong bảng này được **đếm tự động** từ cột trạng thái của các bảng chi tiết bên dưới (sửa ngày 19/10 lúc khoảng 23h; bản trước đếm tay bị sai ở 5/7 dòng).*

**Đánh giá tổng thể:**
- Phần **dữ liệu, bộ chấm, heuristic và đối chứng** (D1–D3) là **đúng và đã kiểm chứng**.
- Phần **phương pháp nhóm** (PP2–PP4) **đúng ở lõi** (CRF khớp công thức (15)–(19), kiểm chứng bằng liệt kê vét cạn), nhưng **chưa đủ so với báo cáo ở hai điểm**:
  1. thiếu đầu dự đoán vị trí;
  2. PP2 chưa được đo bằng ngưỡng xác suất.
  **Cả hai đã sửa ngày 19/10, trước K2**: có kiểm thử (xác suất tồn tại khớp vét cạn; học thuộc đạt PAIR_LOC_F1 = 1,0) và notebook K2 chạy thông.
- Có **4 mục của báo cáo nằm ngoài khả năng làm ngay**: tập test thứ hai, chưng cất LLM, PP1, nguồn tri thức thay thế. Các mục này cần quyết định của nhóm/giảng viên, được ghi ở mục G.

---

## B. §1–§2: Bài toán và phân tích dữ liệu

| # | Báo cáo nói | Pipeline làm | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| B1 | Mỗi thực thể, mỗi câu: nhãn ∈ {CREATE, DESTROY, MOVE, NONE} **và vị trí** | Nhãn ✅. Vị trí ✅ (đầu vị trí đã thêm, xem D9) | `propara_core.build_record`, `PP4Model` | ✅ |
| B2 | Đầu ra cặp (X, Y, t) | `gold_pairs_report` | 533 cặp | ✅ |
| B3 | 488 đoạn, 3.302 câu, 1.915 thực thể | Đếm lại | 488/3302/1915 (NB00, R01) | ✅ |
| B4 | Công thức (1) suy nhãn | `derive_label` | Khớp 100% với `answers.tsv` chính thức; 10 kiểm thử | ✅ |
| B5 | Cờ nhắc m_it (từ nội dung, tên gọi khác, số ít/nhiều) | `is_mentioned(mode="report")` | Bảng 1 khớp 18/18 ô | ✅ |
| B6 | Định nghĩa ngầm: y≠NONE và m=0 | `build_record` → `implicit` | Bảng 1 | ✅ |
| B7 | Bảng 1 (9 ô ngầm + 9 ô tổng) | NB00 R03, R04 | 18/18 | ✅ |
| B8 | Công thức (2) lift + Bảng 2 | `lift_table` | 9/9 ô khớp | ✅ |
| B9 | E0: 59–64% thay đổi ngầm trên test có thực thể khác được nhắc và đổi trạng thái cùng câu | Kiểm tra mới (19/10) | Gộp 3 nhãn: 63% (khớp). Theo từng nhãn: CREATE 80%, DESTROY 86%, MOVE 49% (không nằm trong khoảng). Báo cáo không nói rõ cách tính | ⚠️ chỉ khớp khi gộp |
| B10 | Công thức (3): 533 = 378/58/97 | NB00 R05, R06 | khớp | ✅ |
| B11 | “Bộ đánh giá ProPara suy Q3 theo cùng cách này” | Đọc mã evaluator chính thức | **Sai**: evaluator gộp theo câu, chấm Jaccard mềm, macro theo đoạn; 481 conversion | ⚠️ đã sửa định nghĩa (dùng 2 thước đo) |
| B12 | 57 điểm DESTROY + 76 điểm CREATE | `decision_points` | 57/76 | ✅ |
| B13 | Công thức (4) ngẫu nhiên; Bảng 3 | `decision_point_table` | 50,1/40,1 · 59,6/41,4 · 75,1/32,3 (BC 51/40 · 59/42 · 74/33) | ✅ |
| B14 | Số ứng viên TB 2,6 / 2,9 | NB00 R08 | 2,65 / 2,95 | ✅ |
| B15 | Công thức (5): ±0,114 | Tính lại | 0,1139 (lưu ý: đây là CI của **một** tỉ lệ, không phải của chênh lệch) | ✅ |
| B16 | Đại từ: 14–27% ô ngầm | Kiểm tra mới (19/10) | train 14–27% (khớp; báo cáo đo trên train) | ✅ |

## C. §3: Công trình liên quan

| # | Báo cáo nói | Pipeline | Trạng thái |
|---|---|---|---|
| C1 | CGLI là baseline mạnh | Chấm lại file dự đoán test công bố của CGLI bằng cùng bộ chấm: PAIR_F1 0,545, nguồn ngầm 0,452, đích ngầm 0,091 | ✅ |
| C2 | XPAD | Chỉ mới **tìm nhanh bằng GitHub search API** (từ khoá “xpad procedural”, “allenai xpad”) và không ra kết quả; chưa tìm kỹ (trang bài báo, allenai.org). Theo báo cáo, XPAD không xuất cặp X→Y | ⚠️ chỉ tham khảo |
| C3 | “Chưa tìm thấy mô hình hay kho tri thức nào biểu diễn quan hệ X biến thành Y” | **Chưa kiểm chứng.** Đã xác minh trong phiên: ConceptNet có quan hệ `/r/MadeOf` (mã nguồn conceptnet5); bài “Knowledge-Aware Procedural Text Understanding with Multi-Stage Training” (arXiv 2020) đưa tri thức vào ProPara. Chưa đọc nội dung hai nguồn này để xem có biểu diễn đúng quan hệ “X biến thành Y” hay không | ❌ cần rà tài liệu trước khi viết câu này vào báo cáo cuối |

## D. §4: Phương pháp

### PP1 (§4.1)
| # | Báo cáo | Pipeline | Trạng thái |
|---|---|---|---|
| D1 | Số câu hỏi (6) 52.936, (7) 48.070 | Tái hiện | ✅ |
| D2 | Số câu hỏi (8) 188.794, (9) 42.004 | Thử bằng bộ tách cụm danh từ NLTK: 203.335 / 39.367 | ⚠️ không tái hiện được (phụ thuộc bộ tách cụm danh từ của báo cáo) |
| D3 | Chấm logit Yes/No (10) | **Không chạy** (kế hoạch xếp PP1 vào P2: 331.804 câu hỏi) | ❌ có chủ đích; ghi trong protocol |

### PP2 (§4.2)
| # | Báo cáo | Pipeline | Trạng thái |
|---|---|---|---|
| D4 | Với tồn tại vàng: 11.144/48.070 = 23%, giữ 533/533 | Tái hiện | ✅ |
| D5 | Lọc đồng xuất hiện: 74,7% / 18,9% | Tái hiện (với cách hiểu “cùng được nhắc trong một câu bất kỳ”) | ✅ |
| D6 | Công thức (11): lọc theo **xác suất** tồn tại dự đoán với ngưỡng τ1, τ2 | **Đã sửa:** xác suất tồn tại = biên CRF (forward-backward, khớp liệt kê vét cạn); lưới τ1 × τ2 = 6 × 6 trên dev; điểm vận hành khoá trước τ1=0,1, τ2=0,9; H1 đạt nếu phủ ≥ 95% và giữ ≤ 50% | ✅ (đã sửa 19/10, chờ số thật từ K2) |
| D7 | ψ chỉ khác 0 khi (X, Y) ∈ C_t | Trong CRF chính xác, ràng buộc tồn tại đã bảo đảm X tồn tại và Y chưa tồn tại → C_t nằm sẵn trong giải mã | ⚠️ tương đương về logic; khai báo |

### Kiến trúc + PP4 (§4.3–4.4)
| # | Báo cáo | Pipeline (`propara_pp4.py`) | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| D8 | Encoder RoBERTa + BiLSTM | `PP4Model` (roberta-base + BiLSTM 2×256) | — | ✅ |
| D9 | **Vị trí** dự đoán bằng đầu riêng chấm điểm trên các **cụm danh từ** | **Đã sửa:** ứng viên = cụm danh từ trong đoạn (gán nhãn từ loại NLTK) + lựa chọn '?'; điểm song tuyến giữa h_t^i và biểu diễn ứng viên; L_total = L_PP4 + 1·L_vị_trí; giải mã nhất quán với nhãn | Ứng viên phủ 88–89% vị trí vàng (trần); học thuộc 8 đoạn: PAIR_LOC_F1 = 1,0 | ✅ (đã sửa 19/10) |
| D10 | h_t^i: pooling trên tên thực thể nếu m=1, attention trên câu nếu m=0 | Đúng như vậy | `entity_states` | ✅ |
| D11 | (không nói cách đưa thực thể vào encoder) | Mã hoá riêng từng thực thể: “tên thực thể </s> đoạn văn” (giống TSLM/CGLI) | — | ⚠️ lựa chọn thiết kế, khai báo |
| D12 | φ: MLP 4 nhãn | `self.phi` | — | ✅ |
| D13 | f_t(i,j) = [log E, m_i, m_j, d_i, cos] (13) | Đủ 5 thành phần + **1 hệ số tự do** (bias); d_i chia cho T | — | ⚠️ thêm bias và chuẩn hoá d; khai báo |
| D14 | ψ tuyến tính, chỉ cho cặp (DESTROY, CREATE) (14) | `pair_w` tuyến tính, chỉ cộng khi i=D, j=C | kiểm thử `pair_score_rewards…` | ✅ |
| D15 | Ràng buộc (15) + cập nhật tồn tại | Trạng thái CRF = vector tồn tại | Viterbi luôn hợp lệ (20 lần ngẫu nhiên) | ✅ |
| D16 | S(y) (16), p(y\|x) (17) | `crf_gold_score`, `crf_log_partition` | **log Z khớp liệt kê vét cạn** (sai số < 1e-8) | ✅ |
| D17 | Forward (18)–(19), Viterbi thay tổng bằng max | Có; MOVE/NONE gộp bằng logsumexp (chính xác) | Viterbi khớp vét cạn | ✅ |
| D18 | Cận O(T·4^n(4n+n²)); nhiều thực thể dùng beam search | n ≤ 8 trên ProPara nên giải mã chính xác; đoạn lớn nhất mất 0,06 s (nll + backward) | đo trên **GPU laptop RTX A2000 4 GB** của máy này, chưa đo trên Kaggle | ⚠️ không cần beam trên ProPara; khai báo |
| D19 | (không nói) điểm cho tồn tại ban đầu e_0 | Thêm đầu `init_head` | — | ⚠️ cần thiết để (17) xác định; khai báo |
| D20 | Huấn luyện cực đại log p(y_vàng\|x) | `crf_nll` / số ô | học thuộc 8 đoạn: F1 = 1,0 | ✅ |
| D21 | Mô hình **dự đoán cặp trực tiếp** (§5) | Cặp = D̂×Ĉ suy từ nhãn giải mã chung | — | ⚠️ với vàng D×C hai cách tương đương (audit V2); khai báo |

### PP3 (§4.5)
| # | Báo cáo | Pipeline (`propara_pp3.py`) | Trạng thái |
|---|---|---|---|
| D22 | WikiHow ~215 nghìn bài | `Lajvi/wikihowAll@5e44fed` (SHA-256 khớp, K2). **Đếm được 215.365 dòng, 5.410.591 câu**, khớp con số ~215 nghìn của báo cáo (nhiều khả năng báo cáo đếm theo dòng). Bài gốc (arXiv 1810.09305) ghi “hơn 230.000 cặp bài–tóm tắt”. Việc bản sao là bộ của Koupaee & Wang vẫn là suy luận từ tên file và tên cột | ✅ (số dòng khớp) |
| D23 | 5 nhóm mẫu (bảng trang 6) | 7 mẫu phủ đủ 5 nhóm (+ “made out of” bị động) | ✅ |
| D24 | Rút về danh từ đầu dạng gốc; loại đại từ, từ rỗng | Gán nhãn từ loại NLTK + đưa về số ít bằng luật (không có lemmatizer) | ⚠️ gần đúng; khai báo |
| D25 | (20)–(22) E = σ(uᵀWv + aᵀu + bᵀv + c) | Đúng công thức | ✅ |
| D26 | u_X, v_Y là embedding danh từ | Dùng cùng embedding tĩnh roberta-base cho u và v (W bất đối xứng nên E(X,Y) ≠ E(Y,X)) | ⚠️ khai báo |
| D27 | (23) BCE | `BCEWithLogitsLoss` | ✅ |
| D28 | 3 loại âm: ngẫu nhiên, đảo chiều, cùng bài không khớp mẫu | Đủ 3 loại | ✅ |
| D29 | Tách train/val theo bài | Tách theo bài (băm tiêu đề) | ✅ |
| D30 | Độ phủ (24)–(26) | `coverage` | ✅ |
| D31 | 9% cặp vàng có cùng danh từ đầu | Đo được 9,0% | ✅ |
| D32 | Độ phủ thấp → thay bằng Wikipedia/sách khoa học | **K2 đo được:** phủ cặp 6,9%, nguồn 61,9%, đích 58,7%. Theo quy tắc của báo cáo (phủ cặp thấp nhưng phủ nguồn/đích cao → bộ chấm phải tổng quát hoá), nhưng E không giúp ở ablation (M3 ≈ không E ≈ E xáo trộn). Chưa có nhánh nguồn thay thế | ❌ đề xuất thay nguồn (kết luận tạm: thiết kế lại PP3) |

## E. §5: Đánh giá

| # | Báo cáo | Pipeline | Trạng thái |
|---|---|---|---|
| E1 | CV 5 phần theo chủ đề trên 488 đoạn | `cv5_by_topic.csv` (183 chủ đề, cân bằng) | ✅ |
| E2 | CV dùng cho **các phương pháp** | M0/M0c/M3 có CV (K2); Qwen chính chạy đủ train + dev + test (K3) → so cặp trên 486 đoạn đã làm (`results/final/`). CGLI không có trong CV (chỉ có dự đoán test công bố) | ✅ (trừ CGLI) |
| E3 | **Tập test thứ hai cho ghép cặp (CUTL hoặc NHK)** | Chưa có (chính báo cáo ghi “chưa xác nhận dữ liệu”) | ❌ cần nhóm cung cấp/xác nhận dữ liệu |
| E4 | Baseline CGLI | ✅ (chấm file công bố) | ✅ |
| E5 | Baseline heuristic “nhắc gần nhất” | ✅ trên điểm quyết định + ✅ bản đầu-cuối B1-E2E | ✅ |
| E6 | Baseline **chưng cất LLM vào mô hình nhỏ** | Chưa làm | ❌ (kế hoạch xếp P2) |
| E7 | Thước đo Q3 của ProPara | Có cả Q3 chính thức (mã gốc) và PAIR_F1 (công thức (27) không tính vị trí) + PAIR_LOC | ⚠️ (27) được tách thành 2 thước đo; khai báo |
| E8 | Độ chính xác trên ô ngầm (C/D/M) | `IMPLICIT_ACC` | ✅ |
| E9 | Tỉ lệ vi phạm nhất quán (28) | `VIOLATION_RATE_per_entity` (đúng mẫu số \|E\|) | ✅ |
| E10 | Mốc trên PP1 **kèm phiên bản ProPara đổi tên** | Đổi tên (NONCE) có cho **Qwen và mô hình nhóm**; PP1 không chạy | ⚠️ đổi tên áp cho hệ thống khác PP1 |
| E11 | Chi phí: số lượt gọi LLM / đoạn vs F1 | Có ghi số lời gọi, token, thời gian; chưa vẽ biểu đồ chi phí–chất lượng | ⚠️ thêm vào phân tích cuối |
| E12 | Rò rỉ dữ liệu: đổi tên, nếu điểm giảm mạnh thì nghi ghi nhớ | NONCE (nhãn giữ nguyên 100%); diễn giải thận trọng (mất cả tri thức ngữ nghĩa) | ✅ |
| E13 | Việc còn lại: kiểm tra CGLI có báo cáo Q3 không | Đã tự tính Q3 chính thức từ dự đoán của CGLI (0,724 tổng, Conversions 0,645); **chưa đối chiếu số trong bài báo CGLI** | ⚠️ |

---

## F. Những chỗ mình đã làm KHÁC báo cáo và lý do

1. **Q3 (B11, E7):** báo cáo mô tả sai cách bộ chấm chính thức tính Q3. Pipeline dùng **mã gốc** cho Q3 chính thức, và **PAIR_F1** cho bài toán cặp của đề tài. Không đổi được, vì đây là sửa lỗi.
2. **PP2 nằm sẵn trong CRF (D7):** tương đương logic. PP2 vẫn được đo riêng (D6, sau khi sửa).
3. **Bias trong ψ, chuẩn hoá d, đầu tồn tại ban đầu (D13, D19):** cần thiết về kỹ thuật, không làm thay đổi ý tưởng.
4. **Cặp suy từ nhãn thay vì “dự đoán trực tiếp” (D21):** tương đương với vàng D×C. Muốn đo “ghép cặp đúng ngữ nghĩa” thì cần tập vàng do người chấm (80 cặp mẫu đã chuẩn bị).
5. **Danh từ đầu bằng gán nhãn từ loại + luật số ít (D24), cùng embedding cho u/v (D26):** gần đúng mô tả báo cáo.

## G. Những gì CÒN THIẾU, mức độ, và hướng xử lý

| Mức | Thiếu | Ảnh hưởng tới quyết định Go/No-Go | Xử lý |
|---|---|---|---|
| ~~Cao~~ | Đầu dự đoán vị trí (D9) | — | ✅ **đã sửa** (19/10) |
| ~~Cao~~ | PP2 theo ngưỡng τ1, τ2 (D6) | — | ✅ **đã sửa** (19/10) |
| **Cao** | Qwen chưa có trong CV (E2) | So Qwen với M3 trên lát ngầm chỉ dựa vào test (ít mẫu) | Thêm Qwen-chính chạy trên train ở K3 (sau K1) |
| Trung bình | Kiểm chứng tính mới PP3 (C3) | Cửa 3 (đóng góp) | Rà ConceptNet/KOALA trong báo cáo cuối |
| Trung bình | Nguồn tri thức thay thế (D32) | Chỉ cần nếu K2 cho độ phủ thấp | Quyết định sau K2 |
| Trung bình | Tập test thứ hai CUTL/NHK (E3) | Khái quát | **Cần nhóm xác nhận dữ liệu** |
| Thấp | PP1 (D3), chưng cất LLM (E6) | Mốc trên / baseline phụ | Ngoài phạm vi đợt này (P2 theo kế hoạch) |
| Thấp | Số câu hỏi PP1 (8)(9) không tái hiện được (D2) | Không ảnh hưởng | Ghi chú |
