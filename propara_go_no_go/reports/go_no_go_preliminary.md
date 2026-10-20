> **ĐÃ ĐƯỢC THAY THẾ** bởi `reports/go_no_go_final.md` (20/10, sau khi chấm test M3). Giữ lại để lưu vết.

# Báo cáo Go/No-Go SƠ BỘ: “Ghép cặp chuyển đổi thực thể – dự đoán thay đổi trạng thái ngầm”

**Ngày:** cập nhật 20/10/2026 sau K2 + K3 · **Trạng thái:** sơ bộ, gần cuối, chưa phải quyết định cuối · **Nguồn số liệu:** notebook 00, 01 (chạy CPU, mã và đầu ra trong `propara_go_no_go/`), notebook 03 (Qwen, Kaggle) *[điền sau K1]*.

Ký hiệu nguồn: **[BC]** = báo cáo 16/10 · **[ĐO]** = nhóm tự đo · **[CÔNG BỐ]** = dự đoán do tác giả bài báo công bố, nhóm chấm lại bằng cùng bộ chấm.

---

## 1. Tóm tắt

- **Kết luận hiện tại (tạm, 20/10): ĐIỀU CHỈNH (PIVOT).** Giữ PP4, thiết kế lại PP3. Qwen 7B không vượt trội: thua CGLI trên test và thấp hơn mô hình nhóm ở PAIR_F1 tổng, nhưng mạnh hơn ở thay đổi ngầm. Xem mục 6–7.
- **Những gì đã chắc chắn:**
  1. Toàn bộ phân tích dữ liệu trong báo cáo **tái hiện đúng** (14/14 nhóm số liệu).
  2. Bộ chấm chung **đã kiểm chứng**: tái hiện đúng điểm chính thức của ProStruct (0,545) và ProGlobal (0,519).
  3. Hai phương pháp gốc của ProPara (2018) **không bắt được cặp chuyển đổi ngầm nào** trên test. **CGLI (2022) bắt được một phần**: cặp nguồn ngầm 0,452, nhưng cặp **đích ngầm chỉ 0,091**.
- **Tín hiệu ban đầu:** khoảng trống mà đề tài nhắm tới **còn rõ nhất ở CREATE/đích ngầm** (CGLI 0,091), ít hơn ở nguồn ngầm (CGLI 0,452). Câu hỏi còn mở: Qwen 7B đã lấp khoảng trống này đến đâu *[mục 5, điền sau K1]*.

## 2. Dữ liệu và định nghĩa: đã kiểm chứng

| Hạng mục | [BC] | [ĐO] |
|---|---|---|
| Đoạn / câu / thực thể | 488 / 3.302 / 1.915 | 488 / 3.302 / 1.915 ✅ |
| Cặp vàng D×C (train/dev/test) | 533 (378/58/97) | 533 (378/58/97) ✅ |
| Điểm quyết định ngầm DESTROY / CREATE | 57 / 76 | 57 / 76 ✅ |
| Lift D↔C (Bảng 2) | 9,0 / 6,6 | 9,0 / 6,6 ✅ (9/9 ô) |
| PP2 với tồn tại vàng | 11.144/48.070, giữ 533/533 | ✅ |
| Heuristic nhắc-gần-nhất (DESTROY / CREATE) | 74% / 33% | 75,1% / 32,3% ✅ |

Kiểm tra thêm: nhãn suy ra **trùng 100%** với đáp án của bộ chấm chính thức. 183 chủ đề, **không chủ đề nào bị chia sang hai tập**. Nhãn vàng không vi phạm ràng buộc tồn tại.

## 3. Ba phát hiện về cách đo (ảnh hưởng tới cách lập luận đề tài)

1. **“Q3” trong báo cáo khác Q3 của bộ chấm chính thức.** Bản chính thức gộp mọi thực thể trong một câu thành một conversion, chấm mềm (Jaccard), lấy trung bình theo đoạn: 481 conversion vàng, không phải 533. Q3 chính thức còn có **mức sàn 0,312**: một hệ thống không dự đoán gì vẫn đạt 0,312, gần bằng ProStruct (0,326). → Nhóm dùng **PAIR_F1** (khớp chính xác từng cặp nguồn→đích→câu) làm thước đo chính và báo cáo Q3 chính thức để so với bài báo.
2. **Với định nghĩa vàng “mọi DESTROY × mọi CREATE trong câu”, tập cặp được quyết định hoàn toàn bởi nhãn CREATE/DESTROY.** Vì vậy đóng góp của PP3/PP4 phải được chứng minh qua việc **phát hiện đúng hơn thực thể bị tạo/huỷ ngầm**, không phải qua một bước “ghép cặp” riêng. Chỉ 31% cặp vàng (164/533) đến từ câu có nhiều thực thể.
3. **Tập test có rất ít trường hợp ngầm** (17 cặp nguồn ngầm; 10 điểm quyết định DESTROY). Khoảng tin cậy 95% của PAIR_F1 trên test rộng **±9–11 điểm**. → Kết luận về trường hợp ngầm cần **CV 5 phần theo chủ đề** (đã chia sẵn, cân bằng).

## 4. Các mốc đã có (cùng bộ chấm, tập TEST)

| Hệ thống | Nguồn | PAIR_F1 [CI95] | PAIR_F1 cặp có phía ngầm | Q3 chính thức (Conversions) |
|---|---|---:|---:|---:|
| Không dự đoán gì (ALL_NONE) | [ĐO] | 0,000 | 0,000 | 0,312 |
| ProGlobal (NAACL 2018) | [CÔNG BỐ] | 0,127 [0,056–0,203] | **0,000** | 0,391 |
| ProStruct (EMNLP 2018) | [CÔNG BỐ] | 0,252 [0,147–0,370] | **0,000** | 0,326 |
| Heuristic đầu-cuối B1-E2E (luật + nhắc gần nhất, không dùng vàng) | [ĐO] | 0,222 [0,109–0,344] | 0,104 | 0,411 |
| **CGLI (COLING 2022)** | [CÔNG BỐ] | **0,545** [0,466–0,628] | **0,308** (nguồn ngầm 0,452; đích ngầm 0,091) | 0,645 |
| *Chẩn đoán:* đúng hoàn hảo ở ô **được nhắc**, bỏ qua ô ngầm | [ĐO], dùng vàng | 0,802 [0,705–0,885] | 0,000 | 0,870 |
| *Chẩn đoán:* như trên + heuristic nhắc-gần-nhất cho ô ngầm | [ĐO], dùng vàng | 0,815 | 0,385 | 0,887 |

Đọc bảng:
- **Một hệ thống “mù ngầm” dù hoàn hảo cũng chỉ đạt khoảng 0,80 PAIR_F1.** Phần còn lại (khoảng 20 điểm) và **toàn bộ** lát cặp ngầm là nơi đề tài có thể đóng góp.
- Heuristic đơn giản chỉ lấy lại được một phần của khoảng này.

## 5. Qwen2.5-7B-Instruct trên DEV (đã chạy trên Kaggle, K1)

Thiết lập đã khoá trước khi chạy: `Qwen/Qwen2.5-7B-Instruct@a09a354`, fp16 trên 2×T4, giải mã tham lam, prompt_v1; ví dụ few-shot chỉ lấy từ train; **không** dùng tập test.

| Chế độ (dev, 43 đoạn) | PAIR_F1 [CI95] | Nguồn ngầm | Đích ngầm | Đúng DESTROY / CREATE ngầm | Vi phạm tồn tại/thực thể | JSON hợp lệ | s/đoạn (p50) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mốc: ALL_NONE | 0,000 | 0 | 0 | 0 / 0 | 0 | — | — |
| Mốc: mù ngầm (chẩn đoán, dùng vàng) | 0,779 [0,617–0,897] | 0 | 0 | 0 / 0 | — | — | — |
| Heuristic đầu-cuối B1-E2E | 0,088 [0,017–0,184] | — | — | — | — | — | — |
| Qwen Q0 trực tiếp | 0,217 [0,058–0,352] | 0,095 | 0 | 7% / 9% | 0,67 | 100% | 4,4 |
| **Qwen Q1 có ví dụ (hệ thống Qwen chính)** | **0,377** [0,243–0,495] | 0,231 | 0,118 | 33% / 9% | 0,51 | 100% | 10,0 |
| Qwen Q2 từng bước | 0,318 [0,159–0,462] | 0,083 | 0,105 | 7% / 9% | 0,74 | 81% | 16,3 |
| Qwen Q3 = Q0 + luật tồn tại v1 | 0,033 [0–0,095] | 0 | 0 | 7% / 0% | 0 | — | — |
| Qwen Q1 + luật tồn tại v2 (sửa tồn tại ban đầu trước) | 0,292 | 0,087 | 0 | — | 0 | — | — |
| *ORACLE Q4: biết tồn tại đúng trước câu* | *0,425* [0,287–0,557] | *0,392* | *0,311* | *60% / 73%* | — | 100% | 1,4 |

**Nhận xét (chỉ trên dev, chưa phải kết luận):**
- Ví dụ few-shot giúp rõ: Q1 − Q0 = +0,160 [+0,046; +0,289].
- **Qwen 7B rất yếu ở thay đổi ngầm**, nhất là CREATE ngầm (9%).
- **Khi được cho biết tồn tại đúng trước câu, Qwen gán nhãn ngầm tốt hơn hẳn** (DESTROY 33% → 60%, CREATE 9% → 73%). Phần lớn lỗi của Qwen nằm ở **theo dõi trạng thái tồn tại**. Đây đúng là thứ mà ràng buộc tồn tại + giải mã chung của PP4 nhắm tới. Đây là **tín hiệu ủng hộ** hướng của đề tài, nhưng phải chờ M3 (K2) để biết PP4 có thực sự làm tốt hơn không.
- Qwen vi phạm ràng buộc tồn tại thường xuyên (không có ví dụ: 97/175 thực thể được tạo ra lại bị khai là có sẵn từ đầu). Sửa bằng luật đơn giản luôn làm **giảm** PAIR_F1 (v1: 0,033; v2: 0,292 so với 0,377 khi không sửa). Nghĩa là tính nhất quán không thể “vá” sau, mà cần được mô hình hoá ngay khi giải mã. Đây là luận điểm của PP4, cần kiểm chứng ở K2.
- Chế độ từng bước: 8/43 đầu ra JSON hỏng; 7 trong số đó do Qwen thiếu dấu đóng ngoặc.
- *(Đã có số test của Qwen, xem mục 7: Qwen chính 0,411 so với CGLI 0,545.)*

## 6. Mô hình của nhóm (K2): ablation trên dev, CV 5 phần

| Hệ thống (dev, TB ± SD 3 seed) | PAIR_F1 | Có phía ngầm | PAIR_LOC_F1 | Vi phạm |
|---|---:|---:|---:|---:|
| M0 gán nhãn độc lập | 0,449 ± 0,040 | 0,030 | 0,241 | 0,291 |
| M0c = M0 + luật tồn tại | 0,467 ± 0,032 | 0 | 0,276 | 0 |
| CRF không điểm cặp | 0,495 ± 0,026 | 0 | 0,311 | 0 |
| CRF + điểm cặp, không E | 0,519 ± 0,009 | 0,027 | 0,298 | 0 |
| CRF + E xáo trộn | 0,506 ± 0,006 | 0 | 0,286 | 0 |
| **M3 = PP2+PP3+PP4** | 0,508 ± 0,005 | 0 | 0,281 | 0 |

**CV 5 phần theo chủ đề (488 đoạn):** M3 0,524 so với M0c 0,472.
- ΔM3 − M0c: tổng **+0,051 [+0,020; +0,085]**; có phía ngầm **+0,065 [+0,015; +0,116]**; nguồn ngầm +0,075 [+0,007; +0,145]; đích ngầm +0,036 [−0,004; +0,093].

**PP3:** WikiHow cho 3.683 cặp; phủ **6,9%** cặp vàng (nguồn 62%, đích 59%). M3 ≈ bản không E ≈ bản E xáo trộn → **chưa thấy tri thức WikiHow giúp gì**.

**PP2** (τ1 = 0,1, τ2 = 0,9): giữ 62% ứng viên, phủ 93% cặp vàng → chưa đạt tiêu chí (≥ 95% và ≤ 50%).

**Đổi tên thực thể (dev):** M3 0,508 → 0,421.

## 7. Test (chấm một lần) và đánh giá tạm theo 3 cửa

| Hệ thống (TEST, 54 đoạn) | PAIR_F1 [CI95] | Nguồn ngầm | Đích ngầm |
|---|---:|---:|---:|
| CGLI 2022 (công bố) | **0,545** [0,466–0,628] | 0,452 | 0,091 |
| **Qwen Q1 có ví dụ** | 0,411 [0,301–0,521] | 0,383 | 0 |
| Qwen Q0 trực tiếp | 0,350 [0,238–0,457] | 0,077 | 0 |
| ProStruct 2018 | 0,252 [0,147–0,370] | 0 | 0 |
| B1-E2E heuristic | 0,222 [0,109–0,344] | 0,141 | 0 |
| M3 / M0c của nhóm | *chờ file `pp34_v2.zip`* | | |

- **Qwen − CGLI = −0,134 [−0,253; −0,014]**: CGLI hơn Qwen 7B, có ý nghĩa.
- **Trên 486 đoạn gộp:** Qwen 0,438 (có phía ngầm 0,258) so với M3 CV 0,524 (có phía ngầm 0,096). Đây là so sánh theo điểm số; bootstrap cặp đang chờ file zip.

| Cửa | Trạng thái tạm |
|---|---|
| 1. Bằng chứng đáng tin? | 🟡 Gần đủ; thiếu test M3 và bootstrap cặp M3–Qwen–CGLI |
| 2. Qwen vượt trội rõ? | **Không**: thua CGLI trên test; thấp hơn M3 ở PAIR_F1 tổng (theo điểm số). Nhưng **hơn M3 rõ ở lát ngầm** |
| 3. PP2–PP4 còn đóng góp? | **PP4 có** (CV, có ý nghĩa); **PP3 không thấy**; PP2 chưa đạt H1 |

**Kết luận tạm: ĐIỀU CHỈNH (PIVOT).** Giữ giải mã chung PP4 vì có đóng góp đo được. Thiết kế lại PP3, vì tri thức WikiHow không giúp và độ phủ quá thấp. Vấn đề trung tâm còn mở là **thay đổi ngầm**, nơi mọi hệ thống còn yếu: Qwen tốt hơn M3 nhưng vẫn thấp, CGLI yếu ở đích ngầm.

Hướng thử tiếp theo (đề xuất, chưa kiểm chứng): đưa tín hiệu của LLM vào CRF (thay cho E từ WikiHow), hoặc thay nguồn tri thức bằng văn bản khoa học. **Chưa phải quyết định cuối:** cần chấm test của M3 và các so sánh cặp.

## 8. Giới hạn

- Kết quả Qwen (khi có) chỉ là **dev**, một mô hình, một cấu hình prompt. Không suy rộng cho mọi LLM.
- ProPara công bố từ 2018 nên Qwen có thể đã thấy dữ liệu. Cần phép thử đổi tên/diễn đạt lại (lượt K3).
- Cặp vàng D×C là nhãn **suy từ trạng thái**, chưa được người xác nhận là quan hệ nhân quả. Đã chuẩn bị 80 cặp mẫu để 2 người chấm độc lập.
- Ước tính thời gian/GPU trong báo cáo này **chưa được đo** trên Kaggle.
