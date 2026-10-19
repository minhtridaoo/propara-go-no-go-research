# dataset_validation — propara_audit_v1

- GATE_STATUS: **PASS** (FAIL=0, WARN=2); SMOKE_TEST=False
- Nguồn: allenai/propara@51a3ccc (grids.v1), allenai/aristo-leaderboard@60d5be3 (answers.tsv)
- Môi trường: Python 3.14.4, Windows-11-10.0.26300-SP0, on_kaggle=False

| ID | Trạng thái | Kiểm tra | Kỳ vọng | Quan sát | Ghi chú |
|---|---|---|---|---|---|
| S01 | PASS | Kiểm thử đơn vị | 16/16 | 16/16 |  |
| S02 | PASS | SHA-256 file gốc khớp giá trị ghim | 8/8 | 8/8 | lệch = bản dữ liệu khác; các kiểm tra R quyết định |
| S03 | PASS | Không có lỗi cấu trúc (độ dài trạng thái, câu rỗng, trạng thái rỗng) | 0 | 0 |  |
| S04 | WARN | Thực thể trùng tên trong cùng đoạn | 0 | 6 | đoạn ['1151', '1204', '1207', '263', '893', '901']; giữ nguyên, định danh thực thể bằng chỉ số |
| S05 | PASS | Mã đoạn duy nhất, không chéo tập, khớp file phân chia | 0/0/0 | 0/0/0 |  |
| S06 | PASS | Không có chủ đề (prompt) nằm ở >1 tập | 0 | 0 |  |
| S07 | PASS | Không có đoạn văn trùng nguyên văn giữa các tập | 0 | 0 |  |
| S08 | PASS | Câu trùng nguyên văn giữa các tập (thông tin) | ghi nhận | 18 | câu ngắn chung chung kiểu 'the magma cools.'; không phải rò rỉ nhãn |
| S09 | PASS | Nhãn vàng không vi phạm ràng buộc tồn tại | 0 | 0 |  |
| S10 | PASS | Hành động suy từ grids.v1 = answers.tsv chính thức (so đa tập) | 0/0/0 | 0/0/0 |  |
| S11 | PASS | Mọi đoạn có chủ đề (để chia chéo theo chủ đề) | 0 | 0 |  |
| R01 | PASS | Số đoạn / câu / thực thể (tr.1) | 488/3302/1915 | 488/3302/1915 |  |
| R02 | PASS | Số đoạn train/dev/test (file phân chia) | 391/43/54 | 391/43/54 |  |
| R03 | PASS | Bảng 1: tổng số nhãn theo tập (9 ô) | khớp 9/9 | khớp 9/9 |  |
| R04 | PASS | Bảng 1: số thay đổi ngầm (9 ô, cờ nhắc kiểu báo cáo) | khớp 9/9 | khớp 9/9 |  |
| R05 | PASS | Tổng cặp vàng D×C (3) | 533 | 533 |  |
| R06 | PASS | Cặp vàng train/dev/test | 378/58/97 | 378/58/97 |  |
| R07 | PASS | Điểm quyết định DESTROY/CREATE (tr.2) | 57/76 | 57/76 |  |
| R08 | PASS | Số ứng viên TB DESTROY/CREATE (tr.3, làm tròn) | 2.6/2.9 | 2.65/2.95 | báo cáo có vẻ cắt bớt chữ số thay vì làm tròn |
| R09 | PASS | Chọn ngẫu nhiên kỳ vọng DESTROY/CREATE (Bảng 3) | 51%/40% | 50.6%/39.8% |  |
| R10 | PASS | Bảng 2: lift + số ô (9 ô) | khớp 9/9 | khớp 9/9 |  |
| R11 | PASS | PP1: câu hỏi nhãn 4ΣTn / cặp ΣTn(n−1) (6)(7) | 52936/48070 | 52936/48070 |  |
| R12 | PASS | PP2 với tồn tại VÀNG: số cặp còn lại / cặp vàng giữ lại (12) | 11144/533 | 11144/533 |  |
| R13 | PASS | Lọc đồng xuất hiện: % cặp vàng / % cặp nguồn ngầm (tr.4) | 74.7/18.9 | 74.7/18.9 | chỉ khớp nếu hiểu 'đồng xuất hiện' = cùng được nhắc trong MỘT câu bất kỳ của đoạn; nếu hiểu là cùng câu t thì 66.6/0.0 |
| R14 | PASS | Số thực thể TB mỗi đoạn (tr.6) | 3.9 | 3.924 |  |
| D01 | WARN | Số conversion vàng theo bộ chấm CHÍNH THỨC ≠ số cặp D×C | khác 533 (định nghĩa khác) | 485 (354/49/82) | evaluator gộp mọi D,C của một câu thành 1 conversion + luật ghép câu kế tiếp + chấm Jaccard mềm, macro theo đoạn. Bản sao này đếm cả thực thể trùng tên; evaluator gốc gộp chúng ở 4 đoạn train → notebook 01 đối chiếu với mã gốc |
| D02 | PASS | Cặp ở câu 1×1 (cặp được quyết định hoàn toàn bởi nhãn) | ghi nhận | 369/439 câu; 369/533 cặp | với định nghĩa D×C, tập cặp là hàm của nhãn CREATE/DESTROY |
| D03 | PASS | Không có đoạn n>8 (giải mã chính xác 4^n khả thi) | 0 | 0 |  |
| D04 | PASS | Độ nhạy cờ nhắc (extended −es/−ies) — số ô ngầm thay đổi | ghi nhận | 4 | {'train/CREATE': 2, 'train/DESTROY': 1, 'train/MOVE': 1} |
