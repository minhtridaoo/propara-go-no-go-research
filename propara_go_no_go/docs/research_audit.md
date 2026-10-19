# research_audit.md — Kiểm tra nghiên cứu (Giai đoạn 0)

**Ngày:** 19/10/2026 · **Phiên bản:** audit_v1 · **Phạm vi:** báo cáo 16/10/2026 (8 trang), PIPELINE v1.0, DIEN_GIAI, MASTER_PROMPT.

**Quy ước nguồn:** [BC] = lấy từ báo cáo · [ĐO] = tự đo trên dữ liệu gốc bằng `propara_core` (notebook 00, chạy thử cục bộ trên CPU, **chưa chạy trên Kaggle**) · [NGOÀI] = đã kiểm tra mã/dữ liệu công khai · [GT] = giả thuyết/nhận định, chưa kiểm chứng.

---

## 1. Dữ liệu dùng là gì

- [NGOÀI] Báo cáo dùng **ProPara bản EMNLP'18 `grids.v1.{train,dev,test}.json`** (allenai/propara, commit `51a3ccc`). Tổng đếm được: **488 đoạn / 3.302 câu / 1.915 thực thể** ([ĐO], khớp [BC] tr.1). Chia 391/43/54 đoạn, khớp `naacl18.partition.paraid.tsv`.
- [ĐO] Hành động suy từ `grids.v1` **trùng khớp 100%** (so dạng đa tập) với `answers.tsv` của **bộ chấm chính thức** (allenai/aristo-leaderboard, commit `60d5be3`) cả 3 tập. Vậy nhãn vàng của nhóm = nhãn vàng của leaderboard.
- [ĐO] **183 chủ đề (prompt), không chủ đề nào nằm ở hơn 1 tập.** Không có đoạn trùng nguyên văn giữa các tập; có 18 câu ngắn chung chung trùng nguyên văn (ví dụ “the magma cools.”). Không phải rò rỉ nhãn.

## 2. Đối chiếu số liệu trong báo cáo

| Con số [BC] | Kết quả [ĐO] | Trạng thái |
|---|---|---|
| 488 đoạn, 3.302 câu, 1.915 thực thể | 488 / 3.302 / 1.915 | ✅ khớp |
| Bảng 1 (tổng nhãn, 9 ô) | 9/9 | ✅ khớp |
| Bảng 1 (số ô ngầm, 9 ô) | 9/9, **chỉ khi** khớp số nhiều kiểu đơn giản (w, w+s, bỏ s) | ✅ khớp (đã xác định đúng quy tắc) |
| 533 cặp = 378/58/97 | 533 = 378/58/97 | ✅ khớp |
| 57 điểm quyết định DESTROY, 76 CREATE | 57 / 76 (train 37/56, dev 10/8, test 10/12) | ✅ khớp |
| Số ứng viên TB 2,6 / 2,9 | 2,65 / 2,95 | ✅ (báo cáo cắt bớt chữ số thay vì làm tròn) |
| Ngẫu nhiên 51% / 40% | 50,6% / 39,8% | ✅ khớp |
| Bảng 2 lift (9 ô + số ô) | 9/9 | ✅ khớp |
| PP1: 52.936 câu hỏi nhãn, 48.070 câu hỏi cặp | 52.936 / 48.070 | ✅ khớp |
| PP2 tồn tại vàng: 11.144/48.070, giữ 533/533 | 11.144 / 533 | ✅ khớp |
| Lọc đồng xuất hiện: 74,7% cặp vàng, 18,9% cặp nguồn ngầm | 74,7 / 18,9 | ⚠️ chỉ khớp nếu “đồng xuất hiện” = **cùng được nhắc trong một câu bất kỳ của đoạn**; nếu hiểu là cùng câu *t* thì 66,6% / 0% |
| n trung bình 3,9 | 3,92 | ✅ |
| PP1 câu hỏi vị trí 188.794, thay thế 42.004 | **CHƯA TÁI HIỆN** (cần bộ tách cụm danh từ mà báo cáo dùng) | ⏸ không quan trọng cho quyết định |
| Heuristic 74% / 33%, ngẫu nhiên-không-nhắc 59% / 42% | **CHƯA CHẠY** (notebook 01) | ⏸ |

**Kết luận phần số liệu:** phân tích dữ liệu trong báo cáo **đáng tin và tái hiện được**. Đây là điểm mạnh của đề tài.

## 3. Các vấn đề phải sửa hoặc làm rõ (xếp theo mức ảnh hưởng tới quyết định Go/No-Go)

### V1 — “Q3” trong báo cáo KHÔNG phải Q3 của bộ chấm chính thức (ảnh hưởng: cao)
- [NGOÀI] Bộ chấm chính thức (`aristo-leaderboard/propara/evaluator`) định nghĩa Q3 “Conversions” như sau:
  1. Mỗi câu cho **tối đa một** conversion: gộp **mọi** thực thể bị huỷ và mọi thực thể được tạo trong câu (`A AND B`), vị trí là hợp các vị trí.
  2. Có **luật ghép câu kế tiếp**: câu *t* chỉ có DESTROY, câu *t+1* chỉ có CREATE thì vẫn tính là một conversion (ghi ở câu *t*). Mã gốc có điều kiện `step < T−1` nên luật này không áp dụng cho cặp câu cuối.
  3. Chấm **mềm**: điểm mỗi cặp dự đoán–vàng = trung bình Jaccard (thực thể huỷ, thực thể tạo, vị trí; vị trí qua Porter stemmer); P/R tính **theo từng đoạn rồi lấy trung bình (macro)**; đoạn không có vàng và không có dự đoán được P=R=1.
- [ĐO] Theo bộ chấm chính thức có **481 conversion vàng (350/49/82)**, không phải 533. Bản sao của nhóm cho 485 (354/49/82), khớp mã gốc ở mọi đoạn dev/test; 4 đoạn train lệch do thực thể trùng tên (xem V5).
- Báo cáo viết “Bộ đánh giá ProPara suy câu hỏi chuyển đổi Q3 theo cùng cách này” (tr.2) và công thức (27) (khớp chính xác tập bộ (nguồn, đích, vị trí, câu), micro). **Cả hai câu đều không đúng với mã chính thức.**
- **Quyết định (đã ghi vào protocol):** dùng hai thước đo, đặt tên khác nhau:
  - `PAIR_F1`: thước đo **chính của đề tài**. P/R/F1 micro trên bộ (đoạn, câu, chỉ số nguồn, chỉ số đích), khớp chính xác, đúng chiều. Biến thể `PAIR_LOC_F1` thêm vị trí (chuẩn hoá) làm thước đo phụ.
  - `Q3_OFFICIAL`: chạy **mã evaluator gốc** (vendored, không sửa) để so với bài báo. Yêu cầu mọi hệ thống xuất đủ lưới hành động (nhãn + vị trí trước/sau cho mọi ô).

### V2 — Với định nghĩa vàng D×C, “cặp X→Y” là hàm của nhãn CREATE/DESTROY (ảnh hưởng: cao, về cách lập luận đóng góp)
- [ĐO] Vàng là `D_t × C_t`, nên khi biết đúng nhãn CREATE/DESTROY của mọi thực thể ở mọi câu thì tập cặp cũng đúng 100%. **Dự đoán cặp “trực tiếp” không thể cho thêm thông tin vượt quá nhãn trên vàng hiện tại.** 369/439 câu có cặp là dạng 1×1 (369/533 cặp); chỉ **164 cặp (31%)** đến từ câu có nhiều thực thể.
- Hệ quả:
  1. Lợi ích của PP3/PP4 trên `PAIR_F1` **chỉ có thể đến từ việc đoán đúng hơn nhãn CREATE/DESTROY**, đặc biệt nhãn ngầm (lift D↔C = 9,0/6,6 cho thấy tín hiệu cặp có ích **cho việc gán nhãn**). Nên phát biểu lại H3: “điểm cặp + giải mã chung giúp phát hiện **nguồn/đích ngầm** tốt hơn mô hình gán nhãn độc lập”.
  2. **Chế độ Qwen Q4 “cho trạng thái vàng rồi ghép cặp” là vô nghĩa**: một luật D×C đạt 100%. Thay bằng `ORACLE_PREV_STATE`: cho tồn tại vàng **trước** câu *t*, yêu cầu dự đoán nhãn/cặp ở câu *t*. Phép thử này cô lập đúng năng lực “suy ra thực thể ngầm bị huỷ/tạo”.
  3. Muốn đánh giá “ghép đúng nguồn–đích” như một năng lực riêng thì cần **tập vàng ngữ nghĩa** (người xác nhận X thực sự tạo ra Y) trên các câu nhiều thực thể. Notebook 00 đã xuất `gold_audit_sample.csv` (80 cặp: 30 nguồn ngầm, 20 đích ngầm, 20 câu nhiều cặp, 10 thường) cho hai người chấm độc lập.

### V3 — Cỡ mẫu phần “ngầm” trên test quá nhỏ để kết luận riêng (ảnh hưởng: cao)
- [ĐO] Test: 97 cặp, trong đó **17 cặp nguồn ngầm, 16 cặp đích ngầm, 32 cặp có ít nhất một phía ngầm**; điểm quyết định test chỉ có **10 DESTROY / 12 CREATE**. Dev: 58 cặp, 21 cặp có phía ngầm.
- Khoảng ±11,4 điểm trong báo cáo (tr.3) là nửa độ rộng khoảng tin cậy của **một tỉ lệ** với n=57 (gộp cả 3 tập). Nó **không phải** khoảng tin cậy của chênh lệch hai phương pháp. So sánh cặp (cùng mẫu) nên dùng bootstrap theo đoạn hoặc McNemar, thường hẹp hơn.
- **Quyết định:** (a) kết luận về trường hợp ngầm dựa trên **5-fold CV theo chủ đề** (`cv5_by_topic.csv`, đã khoá, cân bằng 95–99 đoạn/fold, 10–12 DESTROY-dp và 14–17 CREATE-dp mỗi fold); (b) test chính thức chỉ chạy **một lần** để so với tài liệu.

### V4 — PP2 “giữ 533/533” là tất yếu theo cấu tạo (ảnh hưởng: trung bình)
- [GT→ĐO] Cặp vàng đòi hỏi X tồn tại trước câu (vì bị DESTROY) và Y chưa tồn tại (vì được CREATE). Bộ lọc dùng tồn tại **vàng** nên không thể bỏ sót cặp vàng. Con số 100% là **hằng đúng**, không phải kết quả thực nghiệm. Giá trị thật của PP2 = recall cặp vàng khi dùng tồn tại **dự đoán** + mức giảm ứng viên (thực tế còn 23%).

### V5 — Thực thể trùng tên trong 6 đoạn train (ảnh hưởng: thấp)
- [ĐO] Đoạn 263 (`bales`×2), 1151 (`new element`×2), 893 (`glucose`×2), 901 (`sound`×2) trùng nguyên văn; 1204 (`MRNA`/`mRNA`), 1207 (`Skin`/`skin`) khác hoa thường. Evaluator gốc định danh thực thể bằng chuỗi nên **gộp** 4 đoạn đầu (ghi đè trạng thái). Chỉ ảnh hưởng điểm `Q3_OFFICIAL` trên train, **không ảnh hưởng dev/test**.
- **Quyết định:** giữ nguyên dữ liệu, định danh thực thể bằng **chỉ số**; ghi chú khi báo cáo `Q3_OFFICIAL` trên train.

### V6 — Cờ nhắc (ảnh hưởng: thấp)
- [ĐO] Quy tắc của báo cáo bỏ sót số nhiều −es/−ies (echo/echoes, virus/viruses): 4 ô train bị coi là ngầm dù có được nhắc. Báo cáo đã tự nêu rằng quy tắc khớp này lỏng, nên số ô ngầm có thể bị đếm thiếu (do khớp từ riêng lẻ). Không có đồng tham chiếu đại từ.
- **Quyết định:** giữ quy tắc của báo cáo (`mention_mode="report"`) làm chuẩn để tái hiện; bản `extended` chỉ dùng để đo độ nhạy.

### V7 — Tính mới của PP3 cần kiểm tra kỹ hơn (ảnh hưởng: trung bình, cho Cửa 3)
- [BC] “Chưa tìm thấy mô hình hay kho tri thức nào biểu diễn quan hệ ‘X biến thành Y’.” [GT] Cần kiểm tra trước khi viết điều này: ConceptNet có quan hệ có hướng `/r/MadeOf` (đã xác minh trong mã nguồn conceptnet5); bài “Knowledge-Aware Procedural Text Understanding with Multi-Stage Training” (arXiv 2020) và ProStruct (EMNLP 2018, tiêu đề “…Injecting Commonsense Knowledge”) đưa tri thức vào ProPara. *(Câu trước đây “các bài 2023–2025 dùng LLM sinh tri thức” đã bị xoá vì chưa có nguồn kiểm chứng.)* **Không khẳng định tính mới cho tới D3** (notebook 02 / bảng tài liệu).

### V8 — Nguy cơ Qwen đã thấy ProPara (ảnh hưởng: trung bình)
- [GT] Dữ liệu và đáp án công khai trên GitHub từ 2018; Qwen2.5 được huấn luyện trên lượng dữ liệu web rất lớn. Bắt buộc có phép thử diễn đạt lại / đổi tên giữ nghĩa trước khi kết luận Qwen “suy luận được”.

## 4. Tính khả thi kỹ thuật

| Hạng mục | Nhận định | Cơ sở |
|---|---|---|
| PP4 giải mã chính xác | **Khả thi trên toàn bộ ProPara**: n tối đa = 8 → 4^8 = 65.536 cấu hình/câu, T ≤ 10. Không cần beam search trên ProPara (cần lại nếu áp dụng dữ liệu khác) | [ĐO] |
| PP4 huấn luyện bằng −log p(y_vàng) | Tính Z(x) qua 2^n trạng thái tồn tại; MOVE/NONE gộp được (không đổi tồn tại) nên cận 4^n của báo cáo hợp lý. Cần vector hoá (PyTorch), không lặp Python thuần | [ĐO]+[GT] |
| PP3 WikiHow | Cần xác định nguồn dữ liệu WikiHow cụ thể (giấy phép, kích thước) trước khi lập kế hoạch GPU. Báo cáo: notebook độ phủ “chưa chạy trên toàn bộ WikiHow” | [BC] |
| Qwen2.5-7B-Instruct trên Kaggle | T4 16 GB: bản 4-bit (nf4) **có khả năng** chạy được; fp16 (~15 GB) sát giới hạn 1 GPU. Kiểm tra bằng smoke test (notebook 03); không hứa trước | [GT] |
| PP1 (331.804 câu hỏi) | Không chạy toàn bộ. Nếu cần thì lấy mẫu trên dev sau khi Qwen trực tiếp đã có điểm | [BC] |
| Dữ liệu ProPara trên Kaggle | Notebook 00 tải theo commit ghim (cần Internet) hoặc dùng Dataset dự phòng `propara-raw-v1.zip` | [ĐO] |

## 5. Những gì còn THIẾU để quyết định (theo 3 cửa)

- Cửa 1 (bằng chứng đáng tin): thiếu bộ chấm `PAIR_F1` + `Q3_OFFICIAL` có kiểm thử (01), heuristic (01), một baseline nghiên cứu (02), **mô hình nhóm chạy được** (05–07).
- Cửa 2 (Qwen vượt trội?): chưa có điểm Qwen (03–04).
- Cửa 3 (đóng góp còn không?): chưa có ablation; tính mới PP3 chưa kiểm tra (V7).

**Trạng thái quyết định hiện tại: INCONCLUSIVE.** Lý do: chưa có bất kỳ kết quả mô hình nào. Đây là trạng thái đúng ở Giai đoạn 0, không phải tín hiệu xấu.

---

## 6. Bổ sung sau notebook 01 (19/10/2026, [ĐO] bằng evaluator gốc + bộ chấm đề tài)

- **V9 (đã cập nhật sau khi chấm CGLI):** Hai hệ thống 2018 (ProStruct, ProGlobal) **không bắt được cặp ngầm nào** trên test. **Nhưng CGLI (2022) có**: PAIR_F1 0,545 [0,466–0,628]; @nguồn ngầm 0,452; @đích ngầm chỉ 0,091; độ chính xác ô ngầm DESTROY 41%, CREATE 20%. → Khoảng trống còn lại rõ nhất là **CREATE/đích ngầm**, không phải “mọi trường hợp ngầm”. Heuristic đầu-cuối B1-E2E đạt 0,222 (@nguồn ngầm 0,141). (Câu gốc trước khi chấm CGLI:) Hai hệ thống ProPara 2018 (ProStruct, ProGlobal) không bắt được cặp ngầm nào trên test (PAIR_F1@any_implicit = 0; IMPLICIT_ACC DESTROY 0 / 0,09). Đây là bằng chứng ban đầu rằng khoảng trống “chuyển đổi ngầm” có thật ở các phương pháp gốc. Cần kiểm tra thêm với CGLI/MeeT và Qwen trước khi dùng làm luận điểm.
- **V10: Q3_OFFICIAL (Conversions) có mức sàn 0,312 trên test cho hệ thống không dự đoán gì** (đoạn không có conversion vàng và không có dự đoán được P=R=1). ProStruct chỉ đạt 0,326. Q3_OFFICIAL vì vậy **phân biệt kém** các hệ thống về chuyển đổi; PAIR_F1 không có mức sàn này (ALL_NONE = 0).
- **V11: Trần “mù ngầm”:** một hệ thống đúng hoàn hảo trên mọi ô **được nhắc** nhưng bỏ qua mọi thay đổi ngầm đạt PAIR_F1 0,80 (test) / 0,78 (dev). Khoảng ~20 điểm còn lại chính là phần đề tài nhắm tới. Thay ô ngầm bằng heuristic nhắc-gần-nhất chỉ thêm 1–7 điểm.
- **V12: Độ bất định trên test:** CI95 của PAIR_F1 rộng ±9–11 điểm (54 đoạn). So sánh hai hệ thống bắt buộc dùng **paired bootstrap** (ví dụ ProStruct − ProGlobal = +0,125, CI [+0,023; +0,240]).
