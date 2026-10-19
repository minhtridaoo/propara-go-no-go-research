"""Kiểm thử đơn vị cho propara_core (chạy được bằng pytest hoặc run_all()).

Dữ liệu ở đây là ví dụ tự dựng (không phải mẫu ProPara thật).
"""
import propara_core as pc


def _toy(states, sents=None, parts=None, pid="toy"):
    n, T = len(states), len(states[0]) - 1
    raw = {"para_id": pid,
           "sentence_texts": sents or [f"Sentence {t}." for t in range(1, T + 1)],
           "participants": parts or [f"entity{i}" for i in range(n)],
           "states": states}
    return pc.build_record(raw, "train", topic="toy topic")


def test_derive_label_all_cases():
    assert pc.derive_label("leaf", "leaf") == pc.NONE
    assert pc.derive_label("-", "-") == pc.NONE
    assert pc.derive_label("?", "?") == pc.NONE
    assert pc.derive_label("-", "leaf") == pc.CREATE
    assert pc.derive_label("-", "?") == pc.CREATE          # tạo ra nhưng không rõ vị trí
    assert pc.derive_label("leaf", "-") == pc.DESTROY
    assert pc.derive_label("?", "-") == pc.DESTROY          # '?' vẫn là tồn tại
    assert pc.derive_label("leaf", "root") == pc.MOVE
    assert pc.derive_label("?", "leaf") == pc.MOVE          # quy ước báo cáo: còn lại là MOVE
    assert pc.derive_label("leaf", "?") == pc.MOVE


def test_exists_question_mark_is_existing():
    assert pc.exists("?") and pc.exists("leaf") and not pc.exists("-")


def test_mention_report_mode_singular_plural():
    assert pc.is_mentioned("plants", "The plant dies.")
    assert pc.is_mentioned("plant", "Plants die.")
    assert pc.is_mentioned("carbon dioxide; CO2", "The leaf absorbs co2.")   # tên gọi khác
    assert not pc.is_mentioned("the water", "The sun shines.")                # 'the' là từ rỗng
    assert not pc.is_mentioned("water", "Waterfalls are big.")                # khớp nguyên từ
    # chế độ report KHÔNG bắt -es / -ies; chế độ extended thì có
    assert not pc.is_mentioned("echo", "Echoes are produced.")
    assert pc.is_mentioned("echo", "Echoes are produced.", mode="extended")
    assert pc.is_mentioned("berry", "Berries grow.", mode="extended")


def test_build_record_labels_and_implicit():
    rec = _toy([["leaf", "leaf", "-"], ["-", "-", "leaf"]],
               sents=["The entity0 sits.", "Sugar is made."], parts=["entity0", "sugar"])
    assert rec["labels"] == [[pc.NONE, pc.DESTROY], [pc.NONE, pc.CREATE]]
    assert rec["implicit"][0] == [False, True]    # entity0 bị huỷ ở câu 2 mà không được nhắc
    assert rec["implicit"][1] == [False, False]   # sugar được nhắc
    assert pc.validate_record(rec) == []


def test_gold_pairs_cross_product_and_direction():
    rec = _toy([["a", "-"], ["b", "-"], ["-", "c"], ["-", "d"], ["e", "e"]])
    pairs = pc.gold_pairs_report(rec)
    got = {(p["source_idx"], p["target_idx"], p["sentence_id"]) for p in pairs}
    assert got == {(0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1)}       # 2 x 2 = 4 cặp
    assert (2, 0, 1) not in got                                       # đúng chiều nguồn -> đích
    assert all(p["pair_gold_type"] == "derived_from_states" for p in pairs)


def test_no_pairs_when_only_destroy():
    rec = _toy([["a", "-"], ["b", "b"]])
    assert pc.gold_pairs_report(rec) == []


def test_official_conversion_same_step_and_next_step_rule():
    # câu 1: huỷ e0, không tạo; câu 2: chỉ tạo e1 -> rule b gán conversion về câu 1
    rec = _toy([["x", "-", "-", "-", "-"], ["-", "-", "y", "y", "y"]])
    conv = pc.official_q3_conversions(rec)
    assert len(conv) == 1 and conv[0]["step_id"] == 1 and conv[0]["rule"] == "next_step_create"
    # cùng câu: một conversion gộp mọi D và C
    rec2 = _toy([["a", "-"], ["b", "-"], ["-", "c"]])
    conv2 = pc.official_q3_conversions(rec2)
    assert len(conv2) == 1 and conv2[0]["destroyed"] == "entity0 AND entity1"
    assert len(pc.gold_pairs_report(rec2)) == 2     # trong khi định nghĩa cặp của báo cáo cho 2 cặp


def test_official_conversion_keeps_evaluator_off_by_one():
    # T=3: huỷ ở câu 2, tạo ở câu 3. Evaluator chỉ xét rule b khi step < T-1 = 2 -> KHÔNG có conversion
    rec = _toy([["x", "x", "-", "-"], ["-", "-", "-", "y"]])
    assert pc.official_q3_conversions(rec) == []


def test_decision_points_definition():
    # câu 1: e0 bị huỷ ngầm; ứng viên DESTROY = các thực thể đang tồn tại (e0, e1) -> 1 điểm quyết định
    rec = _toy([["a", "-"], ["b", "b"], ["-", "-"]],
               sents=["Something happens."], parts=["alpha", "beta", "gamma"])
    dps = pc.decision_points(rec)
    assert len(dps) == 1 and dps[0]["label"] == pc.DESTROY
    assert dps[0]["candidates"] == [0, 1] and dps[0]["gold"] == [0]
    assert abs(dps[0]["random_acc"] - 0.5) < 1e-12


def test_decision_point_excludes_mentioned_gold_and_needs_two_candidates():
    # beta bị huỷ và ĐƯỢC NHẮC -> bị loại khỏi ứng viên; còn alpha (ngầm) + gamma
    rec = _toy([["a", "-"], ["b", "-"], ["c", "c"]],
               sents=["The beta breaks."], parts=["alpha", "beta", "gamma"])
    dps = pc.decision_points(rec)
    assert len(dps) == 1 and dps[0]["candidates"] == [0, 2]
    # chỉ còn 1 ứng viên -> không phải điểm quyết định
    rec2 = _toy([["a", "-"], ["b", "-"]], sents=["The beta breaks."], parts=["alpha", "beta"])
    assert pc.decision_points(rec2) == []


def test_constraint_violations_self_evolving():
    # DESTROY khi chưa tồn tại, rồi CREATE hợp lệ
    bad, tot = pc.count_constraint_violations([[pc.DESTROY, pc.CREATE]], [False])
    assert (bad, tot) == (1, 2)
    # DESTROY hai lần liên tiếp mà không CREATE lại -> lần 2 vi phạm
    bad, _ = pc.count_constraint_violations([[pc.DESTROY, pc.DESTROY]], [True])
    assert bad == 1
    # chuỗi hợp lệ
    bad, _ = pc.count_constraint_violations([[pc.NONE, pc.CREATE, pc.MOVE, pc.DESTROY]], [False])
    assert bad == 0


def test_gold_sequences_never_violate_constraints():
    rec = _toy([["a", "-", "-", "b", "?"], ["-", "x", "y", "-", "-"]])
    init = [rec["existence"][i][0] for i in range(2)]
    assert pc.count_constraint_violations(rec["labels"], init)[0] == 0


def test_validate_record_catches_errors():
    raw = {"para_id": "bad", "sentence_texts": ["One.", ""], "participants": ["a", "A"],
           "states": [["x", "x", "x"], ["x", ""]]}
    issues = pc.validate_record(pc.build_record(raw, "dev"))
    text = " | ".join(issues)
    assert "câu 2 rỗng" in text and "trùng tên" in text and "cần 3" in text


def test_pp2_counts_gold_ceiling():
    rec = _toy([["a", "-"], ["-", "c"], ["e", "e"]])
    c = pc.pp2_counts(rec)
    assert c["all_pairs"] == 1 * 3 * 2
    assert c["kept_gold_existence"] == 2       # (e0->e1), (e2->e1)
    assert c["gold_kept"] == c["gold_pairs"] == 1


def test_compare_action_rows_ignores_order_but_not_content():
    a = [("1", 1, "x", "NONE", "-", "-"), ("1", 1, "y", "CREATE", "-", "leaf")]
    assert pc.compare_action_rows(a, list(reversed(a)))["n_mismatched_steps"] == 0
    b = [("1", 1, "x", "NONE", "-", "-"), ("1", 1, "y", "CREATE", "-", "root")]
    assert pc.compare_action_rows(a, b)["n_mismatched_steps"] == 1


def test_cv_folds_deterministic_and_grouped():
    recs = [_toy([["a", "-"], ["-", "b"]], pid=str(i)) for i in range(10)]
    for i, r in enumerate(recs):
        r["topic"] = f"topic{i % 4}"
    f1 = pc.cv_folds_by_topic(recs, k=3)
    assert f1 == pc.cv_folds_by_topic(recs, k=3)
    for t in range(4):
        assert len({f1[r["paragraph_id"]] for r in recs if r["topic"] == f"topic{t}"}) == 1


def run_all(verbose=True):
    names = sorted(n for n in globals() if n.startswith("test_"))
    failed = []
    for n in names:
        try:
            globals()[n]()
            if verbose:
                print(f"  PASS  {n}")
        except Exception as e:  # noqa: BLE001 — báo lỗi rõ ràng rồi tổng hợp
            failed.append((n, repr(e)))
            print(f"  FAIL  {n}: {e!r}")
    return len(names), failed
