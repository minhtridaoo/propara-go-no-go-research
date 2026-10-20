# Báo cáo Go/No-Go: “Ghép cặp chuyển đổi thực thể – dự đoán thay đổi trạng thái ngầm”

**Ngày:** 20/10/2026 · **Phiên bản:** final_v1 · **Đề cương áp dụng:** `docs/experiment_protocol.md` v1.5 (khoá trước khi chấm test)
**Nguồn số liệu:** `results/final/final_results.json`, `results/final/final_log.txt`, `results/final/test_seed_aggregates.json` (sinh bởi `tools/final_analysis.py` từ output Kaggle K1/K2/K3 trong `results/kaggle/`). Hình: `results/final/fig_*.png`.
**Quy ước:** PAIR_F1 = thước đo chính (khớp chính xác cặp nguồn→đích→câu). “Cặp ngầm” = cặp có ít nhất một phía không được nhắc trong câu. CI = khoảng tin cậy 95%, bootstrap theo **đoạn**, 10.000 lần. Test chỉ được chấm **một lần**, sau khi mọi cấu hình đã khoá.

---

## 1. Kết luận

**Theo quy tắc đã khoá (protocol §7): KEEP. Cụ thể là giữ đề tài nhưng THU HẸP / ĐIỀU CHỈNH phần đóng góp.**

1. **Qwen2.5-7B không vượt trội**, nên hai điều kiện “cân nhắc bỏ” đều KHÔNG đạt:
   - Trên test, Qwen **thua CGLI** (−0,134 [−0,253; −0,014]).
   - Trên 486 đoạn gộp, Qwen **thua M3** ở PAIR_F1 tổng (−0,086 [−0,136; −0,038]).
2. **Điều kiện KEEP đạt:** trên CV, M3 hơn M0c (cùng mô hình cộng luật tồn tại) có ý nghĩa ở cả PAIR_F1 tổng (+0,050 [+0,018; +0,082]) lẫn **cặp ngầm** (+0,066 [+0,015; +0,117]).
3. **Nhưng đóng góp đến từ PP4 (giải mã chung), không phải PP3:** bản có tri thức WikiHow ≈ bản không có ≈ bản E xáo trộn. Theo ma trận quyết định, dòng “PP3 không giúp nhưng PP4 có ích” → **điều chỉnh: thiết kế lại hoặc bỏ PP3**.
4. **Câu hỏi trung tâm (thay đổi ngầm) chưa được giải:**
   - M3 vẫn rất yếu ở cặp ngầm (CV 0,096).
   - Qwen tốt hơn M3 ở đó (+0,162 [+0,081; +0,242]), nhưng lợi thế này **gần như mất khi đổi tên thực thể**.
   - CGLI tốt hơn M3 ở cặp ngầm trên test.
5. **M3 không hơn CGLI** ở PAIR_F1 tổng trên test (−0,048 trung bình 3 seed; CI của cả 3 seed đều chứa 0).

## 2. Thiết lập (tóm tắt)

| Thành phần | Thiết lập |
|---|---|
| Dữ liệu | ProPara grids.v1 (allenai/propara@51a3ccc): 488 đoạn, chia chính thức 391/43/54; 533 cặp vàng D×C (378/58/97) |
| Thước đo | PAIR_F1 (chính) + lát ngầm; Q3 chính thức (evaluator gốc) để tham khảo |
| Đối chứng | Heuristic (B1, B1-E2E); ProStruct/ProGlobal 2018 và CGLI 2022 (chấm lại file dự đoán test công bố) |
| Qwen | Qwen2.5-7B-Instruct@a09a354, fp16 trên 2×T4, giải mã tham lam, prompt_v1; chế độ chính = có ví dụ (chọn theo dev) |
| Mô hình nhóm | roberta-base + BiLSTM, đầu vị trí, 6 biến thể × 3 seed, chọn epoch theo PAIR_F1 dev, dừng sớm; CV 5 phần theo chủ đề (seed 13) |
| Thống kê | Bootstrap cặp theo đoạn (10.000 lần); CV 486 đoạn gộp (bỏ 2 đoạn dùng làm ví dụ few-shot cho Qwen) |

## 3. Kết quả trên TEST (54 đoạn)

| Hệ thống | PAIR_F1 | Cặp ngầm | Nguồn ngầm | Đích ngầm | PAIR_LOC_F1 | Vi phạm tồn tại |
|---|---:|---:|---:|---:|---:|---:|
| CGLI 2022 (công bố) | **0,545** | **0,308** | 0,452 | 0,091 | 0,364 | 0 |
| **M3 = PP2+PP3+PP4** (TB ± SD 3 seed) | 0,497 ± 0,024 | 0,105 ± 0,089 | 0,155 | 0,029 | 0,237 | 0 |
| M3 không E | 0,493 ± 0,013 | 0,079 | 0,106 | 0,033 | 0,234 | 0 |
| M3 với E xáo trộn | 0,456 ± 0,054 | 0,090 | 0,142 | 0 | 0,197 | 0 |
| CRF không điểm cặp | 0,478 ± 0,030 | 0,106 | 0,147 | 0,037 | 0,233 | 0 |
| M0c = M0 + luật | 0,483 ± 0,023 | 0,091 | 0,153 | 0 | 0,219 | 0 |
| M0 độc lập | 0,469 ± 0,036 | 0,132 | 0,163 | 0,078 | 0,229 | 0,33 |
| **Qwen có ví dụ** | 0,411 [0,301–0,521] | 0,286 | 0,383 | 0 | 0,251 | 0,41 |
| Qwen trực tiếp | 0,350 | 0,048 | 0,077 | 0 | 0,070 | 0,70 |
| ProStruct 2018 | 0,252 | 0 | 0 | 0 | — | 0,22 |
| Heuristic B1-E2E | 0,222 | 0,104 | 0,141 | 0 | — | 0 |

**Chênh lệch trên test** (trung bình qua seed; trong ngoặc: số seed có CI không chứa 0):
- M3 − CGLI: tổng −0,048 (0/3) · cặp ngầm −0,203 (2/3, CGLI hơn).
- Qwen − M3: tổng −0,086 (0/3) · cặp ngầm +0,181 (0/3).
- M3 − M0c: tổng +0,015 (0/3).
- Qwen − CGLI: tổng **−0,134 [−0,253; −0,014]**.

Test chỉ có 54 đoạn và 32 cặp ngầm, nên phần lớn chênh lệch giữa các mô hình học máy **không phân biệt được** trên test. Vì vậy kết luận về cặp ngầm dựa vào CV (mục 4), đúng như protocol đã đăng ký trước.

## 4. Kết quả trên CV 486–488 đoạn (đủ mẫu để kết luận)

| Hệ thống | PAIR_F1 [CI95] | Cặp ngầm | Nguồn ngầm | Đích ngầm | Đúng DESTROY / CREATE ngầm |
|---|---:|---:|---:|---:|---:|
| **M3** (CV) | **0,524** [0,479–0,567] | 0,096 | 0,109 | 0,057 | 0,07 / 0,06 |
| M0c (CV) | 0,474 [0,424–0,524] | 0,031 | 0,033 | 0,021 | 0,02 / 0,03 |
| **Qwen có ví dụ** | 0,438 [0,389–0,484] | **0,258** | 0,315 | 0,110 | 0,43 / 0,22 |

| So sánh (bootstrap cặp theo đoạn) | Δ [CI95] |
|---|---|
| M3 − M0c, tổng | **+0,050 [+0,018; +0,082]** |
| M3 − M0c, cặp ngầm | **+0,066 [+0,015; +0,117]** |
| M3 − M0c, nguồn ngầm / đích ngầm | +0,076 [+0,008; +0,149] / +0,036 [−0,004; +0,090] |
| Qwen − M3, tổng | **−0,086 [−0,136; −0,038]** |
| Qwen − M3, cặp ngầm | **+0,162 [+0,081; +0,242]** |
| Qwen − M3, nguồn ngầm / đích ngầm | +0,206 [+0,094; +0,310] / +0,053 [−0,028; +0,135] |

## 5. Ablation: đóng góp từng thành phần

| Thành phần | Bằng chứng | Kết luận |
|---|---|---|
| **PP4: giải mã chung CRF** | CV: M3 − M0c +0,050 (tổng), +0,066 (ngầm), có ý nghĩa. Dev: CRF không điểm cặp − M0c +0,028 (không có ý nghĩa) | **Có đóng góp** (rõ trên CV) |
| Điểm cặp ψ (không tri thức) | Dev: M3 không E − CRF không điểm cặp +0,024 (không có ý nghĩa); test: 0,493 so với 0,478 | Có thể có, **chưa chứng minh** |
| **PP3: tri thức E(X,Y) từ WikiHow** | Dev: M3 − M3 không E −0,011; M3 − E xáo trộn +0,003. Phủ cặp vàng chỉ 6,9% (nguồn 61,9%, đích 58,7%). AUC của E trên cặp ProPara (dev) 0,61. Các cặp tần suất cao lệch miền (file→format, molehill→mountain…) | **Không thấy đóng góp** (H2 không được ủng hộ) |
| PP2: lọc theo xác suất tồn tại | Dev, τ1 = 0,1, τ2 = 0,9: giữ 62% ứng viên, phủ 93% cặp vàng (tiêu chí: ≥ 95% và ≤ 50%) | **H1 chưa đạt** |
| Ràng buộc tồn tại | M3, M0c: 0 vi phạm theo cấu tạo; Qwen 0,4–0,7 vi phạm/thực thể; sửa Qwen bằng luật sau luôn làm giảm điểm (dev: v1 0,167, v2 0,292 so với 0,377) | Nhất quán là lợi thế vận hành của PP4; không “vá” được cho LLM bằng luật đơn giản |

## 6. Khái quát: đổi tên thực thể bằng từ vô nghĩa (NONCE)

| Hệ thống (test) | PAIR_F1 gốc → đổi tên | Cặp ngầm gốc → đổi tên |
|---|---|---|
| **M3** | 0,497 → **0,489** (−0,008) | 0,105 → 0,122 |
| M3 không E | 0,493 → 0,453 | 0,079 → 0,020 |
| M0c | 0,483 → 0,412 | 0,091 → 0,038 |
| M0 | 0,469 → 0,370 | 0,132 → 0,019 |
| **Qwen có ví dụ** | 0,411 → 0,357 (−0,054) | **0,286 → 0,070** |
| Qwen trực tiếp | 0,350 → 0,342 | 0,048 → 0,093 |

- **M3 bền nhất khi đổi tên.** Lợi thế ở cặp ngầm của Qwen **gần như biến mất** khi đổi tên. Trên dev cũng vậy: Qwen có ví dụ 0,377 → 0,208.
- Theo cảnh báo đã ghi trong kế hoạch (§10.2): đổi tên bằng từ vô nghĩa làm **mất cả tri thức ngữ nghĩa**, nên điểm giảm **không một mình chứng minh** Qwen “ghi nhớ” ProPara. Diễn giải thận trọng: năng lực ngầm của Qwen dựa vào **nghĩa của tên thực thể**.

## 7. Đối chiếu 3 cửa quyết định (protocol §7)

| Cửa | Kết quả |
|---|---|
| **1. Bằng chứng đáng tin?** | **Đạt phần lớn:** cùng dữ liệu, cùng bộ chấm; có heuristic + 3 baseline; mô hình nhóm chạy được với 3 seed; không dùng vàng ở phép thử đầu-cuối; có CI và phân tích lỗi. **Thiếu:** đo chi phí của M3 (độ trễ, VRAM chưa ghi lại); tập test thứ hai (CUTL/NHK); kiểm tra nhãn vàng bởi người (P1) |
| **2. Qwen vượt trội rõ, ổn định?** | **KHÔNG.** STOP_1 sai (Qwen thua M3 ở tổng); STOP_2 sai (trên test không có ý nghĩa; trên CV Qwen hơn ở cặp ngầm nhưng không vững khi đổi tên) |
| **3. PP2–PP4 còn đóng góp?** | **PP4: có.** PP3: không. PP2: chưa đạt H1 |

**Áp quy tắc:**
- Điều kiện KEEP (“M3 hơn M0c có ý nghĩa trên trục ngầm theo CV”) **đạt**.
- Đồng thời điều kiện điều chỉnh (“PP3 vô ích nhưng PP4 có ích”) cũng **đạt**.
- → **KEEP có điều chỉnh**: giữ đề tài, thu hẹp đóng góp về PP4, thiết kế lại PP3.

## 8. Đề xuất bước tiếp theo (có cơ sở từ dữ liệu, chưa kiểm chứng)

1. **Thay nguồn tri thức của PP3.** WikiHow chỉ phủ 6,9% cặp vàng và lệch miền. Báo cáo gốc đã dự phòng: Wikipedia hoặc sách khoa học. Hoặc dùng **tín hiệu từ LLM làm đặc trưng ψ/φ trong CRF**: Qwen giỏi cặp ngầm hơn, CRF bảo đảm nhất quán và bền khi đổi tên.
2. **Tập trung vào CREATE/đích ngầm:** mọi hệ thống đều yếu nhất ở đây (CGLI 0,091; Qwen 0 trên test; M3 0,029).
3. **Theo dõi tồn tại là nút thắt.** Oracle cho thấy khi biết tồn tại đúng trước câu, Qwen gán đúng ô ngầm nhiều hơn hẳn (dev: DESTROY 33% → 60%, CREATE 9% → 73%).
4. **Bổ sung để viết paper:** đo chi phí M3; kiểm tra nhãn vàng bởi người (80 cặp đã chuẩn bị); thêm seed cho CV; so với CGLI bằng cách chạy lại mã CGLI trên dev (hiện chỉ có dự đoán test).

## 9. Giới hạn (phải nêu trong paper)

- **Một LLM, một cấu hình** (Qwen2.5-7B, prompt_v1, giải mã tham lam). Không suy rộng cho mọi LLM.
- **Test nhỏ** (54 đoạn, 32 cặp ngầm). CV chỉ 1 seed. Trên test, mức dao động giữa các seed lớn (ví dụ M2r 0,403–0,512).
- **Cặp vàng D×C suy tự động**, chưa được người xác nhận là quan hệ nhân quả (31% cặp đến từ câu nhiều thực thể).
- **CGLI chỉ có dự đoán test công bố.** Không có dev/CV của CGLI để so cặp trên 486 đoạn.
- **Chưa đo chi phí của M3.** Chỉ biết M3 không gọi LLM khi suy luận; Qwen có ví dụ mất khoảng 10 giây mỗi lời gọi (p50, theo lô, 2×T4).
- PP3 dùng bộ gán nhãn từ loại và luật số ít đơn giản thay cho bộ phân tích cú pháp; E(X,Y) có thể chưa hội tụ (epoch tốt nhất = epoch tối đa 30).
- Đổi tên NONCE làm mất ngữ nghĩa; chưa có tập diễn đạt lại hoặc tập quy trình mới gán nhãn độc lập.
