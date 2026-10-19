# Báo cáo Go/No-Go SƠ BỘ: “Ghép cặp chuyển đổi thực thể – dự đoán thay đổi trạng thái ngầm”

**Ngày:** 19/10/2026 (cập nhật ~23h30) · **Trạng thái:** sơ bộ, chưa phải quyết định cuối · **Nguồn số liệu:** notebook 00, 01 (chạy CPU, mã và đầu ra trong `propara_go_no_go/`), notebook 03 (Qwen, Kaggle) *[điền sau K1]*.

Ký hiệu nguồn: **[BC]** = báo cáo 16/10 · **[ĐO]** = nhóm tự đo · **[CÔNG BỐ]** = dự đoán do tác giả bài báo công bố, nhóm chấm lại bằng cùng bộ chấm.

---

## 1. Tóm tắt

- **Kết luận hiện tại: INCONCLUSIVE (chưa đủ bằng chứng để giữ hay bỏ).** Lý do: mô hình của nhóm (PP2+PP3+PP4) **chưa được huấn luyện**, và theo quy tắc đã khoá thì không được kết luận bỏ khi phương pháp chưa được thử.
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
- So sánh với CGLI (0,545) **chưa làm được**: CGLI chỉ có số trên test, còn Qwen hiện mới có dev. K3 sẽ cho số test của Qwen.

## 6. Đối chiếu 3 cửa quyết định (protocol §7)

| Cửa | Câu hỏi | Trạng thái |
|---|---|---|
| 1. Bằng chứng đáng tin? | Cùng dữ liệu, cùng bộ chấm, có heuristic + baseline, có kết quả mô hình nhóm | 🟡 Dữ liệu + bộ chấm + heuristic (B1, B1-E2E) + 3 baseline công bố (ProStruct, ProGlobal, CGLI) ✅. **Thiếu kết quả mô hình nhóm** |
| 2. Qwen vượt trội rõ, ổn định? | Hơn ≥ 5 điểm PAIR_F1 và cả ở lát ngầm, CI không chứa 0, vững với đổi tên/diễn đạt | 🟡 Mới có dev: Qwen chính 0,377, rất yếu ở ngầm. Cần test (K3), CV 486 đoạn và M3 (K2) |
| 3. PP2–PP4 còn đóng góp? | Ablation, chi phí, tính mới | ⬜ chưa huấn luyện |

→ Vì chưa qua Cửa 1, **chưa được phép kết luận bỏ**. Theo quy tắc đã khoá, kết luận là **INCONCLUSIVE**.

## 7. Đề xuất hướng đi và kế hoạch

1. **Giữ đề tài, phát biểu lại đóng góp cho đúng với cách đo:** “phát hiện thực thể bị **tạo/huỷ ngầm** nhờ tri thức cặp X→Y + giải mã chung”, đo bằng PAIR_F1 trên lát ngầm. Giải mã chính xác của PP4 **khả thi** trên ProPara (tối đa 8 thực thể/đoạn → ≤ 65.536 cấu hình/câu).
2. **So sánh then chốt cần có:** mô hình nhóm đầy đủ (M3) với mô hình cơ sở **cộng luật ràng buộc** (M0c). Phải so với M0c, không phải M0, để chứng minh cải thiện không chỉ đến từ luật.
3. **Lịch dự kiến** (ước tính): huấn luyện M0–M3 + CV (2–4 ngày làm + GPU Kaggle) → test một lần + biến thể đổi tên/diễn đạt lại → báo cáo cuối.
4. **Cần giảng viên xác nhận:** ngưỡng chênh lệch có ý nghĩa thực tiễn 5 điểm PAIR_F1; việc dùng PAIR_F1 làm thước đo chính.

## 8. Giới hạn

- Kết quả Qwen (khi có) chỉ là **dev**, một mô hình, một cấu hình prompt. Không suy rộng cho mọi LLM.
- ProPara công bố từ 2018 nên Qwen có thể đã thấy dữ liệu. Cần phép thử đổi tên/diễn đạt lại (lượt K3).
- Cặp vàng D×C là nhãn **suy từ trạng thái**, chưa được người xác nhận là quan hệ nhân quả. Đã chuẩn bị 80 cặp mẫu để 2 người chấm độc lập.
- Ước tính thời gian/GPU trong báo cáo này **chưa được đo** trên Kaggle.
