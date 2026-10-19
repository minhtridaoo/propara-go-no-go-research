# experiment_protocol.md — Đề cương thực nghiệm (khoá trước khi chạy mô hình)

**Phiên bản:** protocol_v1.4 · 19/10/2026 · Trạng thái: **KHOÁ cho Giai đoạn 1–4**.

> **v1.1 (19/10, trước khi có bất kỳ điểm Qwen nào):**
> (a) Với LLM, cặp dùng cho `PAIR_F1` chính = **D̂×Ĉ suy từ nhãn LLM** (cùng định nghĩa với vàng, áp dụng như nhau cho mọi hệ thống). Conversions LLM tự khai báo được báo cáo riêng là `PAIR_F1[llm_pairs]`.
> (b) Giải mã Qwen: greedy, `repetition_penalty=1.0` (ghi đè mặc định 1,05 của Qwen để đảm bảo tất định).
> (c) Notebook GPU dùng `RUN_FULL="auto"`: chạy đầy đủ dev **chỉ khi** smoke 3 đoạn train đạt (Q0 ≥ 2/3 JSON hợp lệ). Quy tắc này thay cho việc người dùng tự bật `RUN_FULL`.
> (d) Hệ thống tham chiếu chạy cùng bộ chấm đã có: ALL_NONE, ProStruct 2018 và ProGlobal 2018 (dự đoán công bố, chỉ có trên test).
>
> **v1.2 (19/10, trước khi có bất kỳ điểm Qwen hay mô hình nhóm nào), đăng ký trước cho K2/K3/phân tích cuối:**
> (e) **Hệ thống Qwen chính** = chế độ đầu-cuối có PAIR_F1 **dev** cao nhất trong {DIRECT, FEWSHOT, STRUCTURED, CONSTRAINED}.
> (f) **Mô hình nhóm:** encoder `roberta-base@e2da8e2`, 8 epoch, AdamW (2e-5 encoder / 1e-3 đầu ra), tích luỹ 4 đoạn, seed {13, 42, 2026}. Chọn epoch theo PAIR_F1 dev (hoà thì lấy dev loss thấp hơn). M0c dùng **cùng** mô hình M0, chọn epoch riêng theo PAIR_F1 dev khi giải mã ràng buộc. Báo cáo trung bình ± SD qua seed.
> (g) **CV 5 phần theo chủ đề:** M0/M0c và M3, seed 13, số epoch = trung vị epoch tốt nhất trên dev của biến thể đó. Lấy mô hình cuối, không chọn trên fold kiểm tra. Kiểm định chính cho H3 = paired bootstrap ΔPAIR_F1@any_implicit(M3 − M0c) trên 488 đoạn gộp.
> (h) **Test:** mọi dự đoán test (Qwen, mô hình nhóm, biến thể NONCE) được **lưu kín trong notebook, chấm một lần duy nhất** bằng `tools/final_analysis.py` sau khi K1–K3 xong.
> (i) **PP3:** WikiHow `Lajvi/wikihowAll@5e44fed` (sha256 ghim), 7 mẫu của báo cáo, danh từ đầu theo bộ gán nhãn từ loại NLTK (pp3_v2); E(X,Y) song tuyến trên embedding tĩnh của roberta-base, BCE, chọn theo loss validation (tách theo bài).
> (j) **Khái quát:** biến thể NONCE (`robust_v1`) đổi từ nội dung của tên thực thể thành từ vô nghĩa, nhãn giữ nguyên (đã kiểm tra 43/43 dev, 54/54 test). Đây là phép thử **mất tri thức ngữ nghĩa**, không một mình chứng minh ghi nhớ.
> (l) **v1.3 (19/10, vẫn trước mọi kết quả):** thêm **dừng sớm** cho huấn luyện trên split chính thức: dừng khi PAIR_F1 dev (lấy max của các tên theo dõi) không tăng 3 epoch liên tiếp, tối đa 8 epoch. Thêm checkpoint theo epoch để tiếp tục khi bị ngắt; đã kiểm chứng tiếp tục cho kết quả trùng khớp tuyệt đối với chạy liền mạch. CV không dùng dừng sớm (số epoch cố định theo (g)).
> (m) **v1.4 (19/10, vẫn trước mọi kết quả), sau khi tự đối chiếu với báo cáo (`docs/DOI_CHIEU_BAO_CAO.md`):**
>   - Thêm **đầu dự đoán vị trí** cho mọi biến thể PP4 (§4.3 báo cáo): ứng viên = cụm danh từ trong đoạn + '?', L_total = L_PP4 + 1·L_vị_trí.
>   - **PP2 đo theo công thức (11)** bằng xác suất tồn tại (biên CRF) trên lưới τ1 ∈ {0,01; 0,05; 0,1; 0,2; 0,3; 0,5} × τ2 ∈ {0,5; 0,7; 0,8; 0,9; 0,95; 0,99} trên dev. **Điểm vận hành khoá:** τ1 = 0,1, τ2 = 0,9. H1 đạt nếu phủ cặp vàng ≥ 95% và giữ ≤ 50% ứng viên.
>   - Giới hạn (k) cũ (không có đầu vị trí) **không còn**.
> (k) **Giới hạn đã biết (cũ, đã được (m) thay thế):** mô hình nhóm chưa có đầu dự đoán vị trí (xuất '?'), nên PAIR_LOC_F1 và Q3_OFFICIAL của mô hình nhóm thấp hơn thực chất. Thước đo chính PAIR_F1 không bị ảnh hưởng. Mọi thay đổi phải tăng phiên bản và ghi vào `EXPERIMENT_LOG.md` kèm lý do. **Tuyệt đối không thay đổi sau khi đã thấy điểm test.**

Các mục đánh dấu **[cần GV xác nhận]** là đề xuất của nhóm. Nhóm cứ tiến hành theo đề xuất; nếu giảng viên đổi thì tăng phiên bản trước khi mở test.

---

## 1. Câu hỏi và giả thuyết

**Câu hỏi:** Qwen 7B và các phương pháp hiện có làm tốt bài toán chuyển đổi thực thể ngầm đến đâu? PP2+PP3+PP4 có còn giá trị khoa học hoặc vận hành đủ lớn để tiếp tục không?

| Mã | Giả thuyết (đã điều chỉnh theo research_audit V2, V4) | Phép thử quyết định |
|---|---|---|
| H1 | PP2 với tồn tại **dự đoán** giữ ≥ 95% cặp vàng (dev) trong khi giảm ≥ 50% ứng viên | Đường đánh đổi τ1, τ2 trên dev |
| H2 | E(X,Y) (PP3) phân biệt cặp đúng với cặp đảo chiều/ngẫu nhiên, và **làm tăng F1 nguồn/đích ngầm** khi đưa vào ψ | M3 so với M3-no-E; AUC của E trên cặp ProPara |
| H3 | Điểm cặp ψ + giải mã chung (PP4) **phát hiện CREATE/DESTROY ngầm tốt hơn** mô hình gán nhãn độc lập, kể cả khi mô hình đó **được cộng thêm luật ràng buộc** | M3 so với M0c (M0 + ràng buộc), cùng encoder |
| H4 | Qwen2.5-7B-Instruct không huấn luyện thêm đạt PAIR_F1 đầu-cuối cạnh tranh | Q0–Q3 trên dev, rồi test một lần |
| H5 | Phương pháp nhóm có lợi thế ở ít nhất một trục: ngầm / nhất quán / khái quát / chi phí | Bảng 6 trục (mục 7) |
| H6 | Kết luận không do rò rỉ, nhãn nhập nhằng hay ngẫu nhiên | Kiểm tra nhãn vàng, biến thể diễn đạt, CI bootstrap |

## 2. Dữ liệu và phân chia (đã khoá ở notebook 00)

- Nguồn: `allenai/propara@51a3ccc` `data/emnlp18/grids.v1.*.json`; đáp án chính thức `allenai/aristo-leaderboard@60d5be3`. SHA-256 ghim trong notebook 00.
- Dữ liệu chuẩn: output `propara_audit_v1` (core_v1). Thực thể định danh bằng **chỉ số** trong đoạn.
- **Đánh giá chính A:** split chính thức 391/43/54. Chỉnh sửa (prompt, ngưỡng, siêu tham số) **chỉ trên dev**. Test chạy **một lần** cho mỗi cấu hình đã khoá.
- **Đánh giá chính B (cho trường hợp ngầm):** 5-fold CV theo chủ đề, `cv5_by_topic.csv` (seed 2026). Chọn siêu tham số **bên trong** phần huấn luyện của từng fold (tách một fold con làm validation). Không gộp số CV với số test như một tập độc lập.
- Mô hình chỉ nhận `model_inputs_*.jsonl` (câu + tên thực thể), **không bao giờ** nhận states/labels, trừ chế độ đánh dấu ORACLE.
- Ví dụ few-shot cho Qwen chỉ lấy từ **train**.

## 3. Thước đo (tên cố định trong mã)

| Tên | Định nghĩa | Vai trò |
|---|---|---|
| `PAIR_P/R/F1` | Micro trên tập bộ (paragraph_id, sentence_id, source_idx, target_idx), khớp chính xác, có chiều. Vàng = D_t×C_t (533). Dự đoán = các cặp hệ thống xuất ra (nếu hệ thống chỉ xuất nhãn thì dùng D̂_t×Ĉ_t). Mẫu số 0 → P hoặc R = 0, ghi cảnh báo | **Thước đo chính** |
| `PAIR_LOC_F1` | Như trên + vị trí (vị trí sau của đích, chuẩn hoá: lower, bỏ mạo từ, Porter stem) | Phụ |
| `PAIR_F1@src_implicit`, `@tgt_implicit`, `@any_implicit`, `@both_implicit` | Chia lát theo cờ nhắc **tính từ văn bản** (không cần vàng), áp dụng cho cả dự đoán và vàng, nên P và R đều có nghĩa | **Trọng tâm đề tài** |
| `Q3_OFFICIAL_P/R/F1` | Mã `aristo-leaderboard/propara/evaluator` gốc (Conversions); kèm Inputs/Outputs/Moves/Overall | So với bài báo |
| `LABEL_F1[CREATE/DESTROY/MOVE/NONE]`, macro-F1 3 nhãn thay đổi | Trên mọi ô (thực thể × câu) | Trung gian |
| `IMPLICIT_ACC[CREATE/DESTROY/MOVE]` | Tỉ lệ ô vàng ngầm được gán đúng nhãn (định nghĩa tr.7 báo cáo) | Trung gian |
| `DP_ACC[DESTROY/CREATE]` | Độ chính xác trên điểm quyết định ngầm (57/76), so được với heuristic 74%/33% | Chẩn đoán |
| `EXIST_ACC` | Độ chính xác tồn tại sau mỗi câu, trạng thái **tự tiến hoá** | Trung gian |
| `VIOLATION_RATE` | Công thức (28), trạng thái tự tiến hoá từ nhãn dự đoán | Logic |
| `JSON_VALID_RATE`, `HALLUCINATED_ENTITY_RATE`, `TRUNCATION_RATE` | Chỉ cho LLM | Kỹ thuật |
| `latency_p50/p95_ms`, `peak_vram_gb`, `gpu_hours_per_1000_paragraphs` | Đo cùng phần cứng khi có thể | Chi phí |

## 4. Thống kê

- CI 95% của **chênh lệch** ΔPAIR_F1 bằng **paired bootstrap theo đoạn**, 10.000 lần lấy mẫu, seed 2026. Mỗi lần lấy lại cùng tập đoạn cho cả hai hệ thống.
- Với `DP_ACC`: McNemar chính xác (cặp) cho chênh lệch.
- Mô hình huấn luyện: ≥ 3 seed (ưu tiên 5 nếu đủ quota); báo cáo trung bình ± độ lệch chuẩn. Qwen: 1 lần giải mã tham lam (temperature 0) + độ nhạy với 2 cách diễn đạt prompt đã khoá.
- “Không có ý nghĩa thống kê” ≠ “tương đương”.

## 5. Ngưỡng thực tiễn [cần GV xác nhận]

- Chênh lệch PAIR_F1 **≥ 5 điểm phần trăm** và CI 95% không chứa 0 → coi là chênh lệch **đáng kể**.
- Trên lát ngầm (n nhỏ): chỉ kết luận nếu CI không chứa 0 **trên CV 5-fold**; một mình test thì chỉ mô tả, không kết luận.
- Ngưỡng này dùng để mô tả mức chênh, **không phải luật “Qwen hơn 5 điểm thì bỏ”**.

## 6. Hệ thống và chế độ chạy (đã khoá tên)

| Mã | Hệ thống | Ghi chú |
|---|---|---|
| H-RAND, H-RAND-UNM, H-NEAREST | Heuristic điểm quyết định (tr.3 báo cáo), 20 seed | 01 |
| H-ALLNONE, H-COPYMENTION | Heuristic đầu-cuối xuất đủ lưới hành động (để chạy được Q3_OFFICIAL) | 01, kiểm tra luồng |
| B-CGLI (ưu tiên), B-MeeT, B-ProStruct | Tái hiện nếu có mã + trọng số; nếu không thì ghi “trích bài báo, không so trực tiếp” | 02 |
| QWEN_DIRECT (Q0) | Qwen2.5-7B-Instruct, zero-shot, xuất JSON đủ lưới + cặp | 03→04 |
| QWEN_FEWSHOT (Q1) | + 3 ví dụ cố định từ train | 04 |
| QWEN_STRUCTURED (Q2) | tồn tại trước → nhãn → vị trí → cặp | 04 |
| QWEN_CONSTRAINED (Q3) | Q0 + hậu xử lý luật tồn tại (không dùng vàng) | 04 |
| ORACLE_PREV_STATE (Q4) | Cho tồn tại vàng **trước** câu t; **ORACLE, bảng riêng**. Thay cho “cho trạng thái vàng” (vô nghĩa với vàng D×C, audit V2) | 04 |
| PP1_SAMPLE | Logit Yes/No trên một mẫu dev, chỉ khi còn quota | tuỳ chọn |
| M0 / M0c / M1 / M2 / M2r / M3 / M3-no-E / M3-no-pair | Như PIPELINE §9.2, cùng encoder roberta-base | 05–08 |

Cấu hình Qwen (khoá ở notebook 03 sau smoke test): `Qwen/Qwen2.5-7B-Instruct`, ghi revision hash; 4-bit nf4 (bitsandbytes) nếu fp16 không vừa một GPU; chat template chuẩn; temperature 0; `max_new_tokens` đủ cho đoạn dài nhất (đo trên train). Một đoạn = một lời gọi.

## 7. Quy tắc quyết định (khoá trước test)

Đánh giá theo 6 trục: (1) PAIR_F1 đầu-cuối; (2) PAIR_F1@implicit + IMPLICIT_ACC CREATE/DESTROY; (3) VIOLATION_RATE và lỗi chiều; (4) khái quát (diễn đạt lại / đổi tên giữ nghĩa); (5) ổn định (CI, seed, độ nhạy prompt); (6) chi phí + tính mới.

| Kết luận | Điều kiện (tất cả) |
|---|---|
| **KEEP** | M3 hơn M0c có ý nghĩa trên trục (2) theo CV, **hoặc** M3 không thua Qwen quá 5 điểm PAIR_F1 trong khi rẻ hơn ≥ 10× và V=0 |
| **PIVOT** | Qwen hơn rõ trên (1), nhưng còn nhóm lỗi có hệ thống (ví dụ nguồn ngầm, chiều, nhất quán) mà PP2–PP4 sửa được trong thí nghiệm có kiểm soát; hoặc PP3 vô ích nhưng PP4 có ích |
| **CONSIDER STOP** | Qwen hơn M3 ≥ 5 điểm trên (1) **và** (2), CI không chứa 0, vững trên (4), **và** ablation cho thấy PP2/PP3/PP4 không đóng góp đáng kể, **và** không có lợi thế chi phí/tính mới đáng kể |
| **INCONCLUSIVE** | Mọi trường hợp còn lại, đặc biệt khi M3 chưa huấn luyện đủ hoặc CI rộng |

Kết luận chỉ áp dụng cho Qwen2.5-7B-Instruct với cấu hình đã khoá, không suy rộng cho “mọi LLM”.

## 8. Ngân sách tài nguyên (ước tính, cần kiểm tra trên tài khoản)

- CPU: 00, 01, 05 (phân tích), 09, 10.
- GPU (thứ tự ưu tiên): 03 smoke (< 1 giờ) → 04 dev (Q0/Q2/Q3, Q4) → 07 M0/M3 → 08 ablation/seed → 04 test một lần → biến thể khái quát.
- Không chạy PP1 toàn bộ. Không train PP3 trước khi đo độ phủ WikiHow trên CPU.
- Quota GPU Kaggle mỗi tuần **phải xem trên tài khoản**; không giả định.

## 9. Quy tắc chống rò rỉ

1. Không đọc điểm test cho tới khi cấu hình được khoá và ghi vào log.
2. Few-shot chỉ từ train; ngưỡng τ, prompt, siêu tham số chỉ chọn trên dev / fold con.
3. Mô hình không thấy `states/labels` (trừ ORACLE, đặt trong bảng riêng).
4. Người chấm tập vàng ngữ nghĩa không xem dự đoán của mô hình.
5. Mọi lần chạy lại trên test sau khi sửa → đặt tên thí nghiệm mới và ghi rõ đã dùng thông tin test.
