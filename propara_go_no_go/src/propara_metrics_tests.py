"""Kiểm thử bộ chấm (PIPELINE §6.5) — chạy bằng pytest hoặc run_all().

Ví dụ tự dựng: đoạn 2 câu, 4 thực thể.
  câu 1: 'Water and carbon dioxide enter the leaf.'  -> không đổi tồn tại
  câu 2: 'Sugar is made.'  -> water, carbon dioxide bị huỷ (NGẦM); sugar được tạo (được nhắc)
"""
import random

import propara_core as pc
import propara_metrics as pm


def _rec():
    raw = {"para_id": "t1",
           "sentence_texts": ["Water and carbon dioxide enter the leaf.", "Sugar is made."],
           "participants": ["water", "carbon dioxide", "sugar", "oxygen"],
           "states": [["root", "leaf", "-"], ["air", "leaf", "-"], ["-", "-", "leaf"], ["-", "-", "-"]]}
    return pc.build_record(raw, "dev")


def _pred(rec, labels=None, pairs=None):
    p = pm.gold_as_prediction(rec)
    if labels is not None:
        p["labels"] = labels
    if pairs is not None:
        p["pairs"] = pairs
    return p


def _score(rec, pred):
    return pm.evaluate({rec["paragraph_id"]: pred}, [rec])


def test_perfect_prediction_f1_is_one():
    rec = _rec()
    r = _score(rec, pm.gold_as_prediction(rec))
    assert r["PAIR@all"]["F1"] == 1.0 and r["PAIR@all"]["n_gold"] == 2
    assert r["PAIR@src_implicit"]["F1"] == 1.0 and r["PAIR@src_implicit"]["n_gold"] == 2
    assert r["VIOLATION_RATE_per_entity"] == 0 and r["EXIST_ACC"] == 1.0 and r["PAIR_LOC"]["F1"] == 1.0


def test_no_correct_pair_gives_zero():
    rec = _rec()
    r = _score(rec, pm.all_none_prediction(rec))
    assert r["PAIR@all"]["F1"] == 0.0 and r["PAIR@all"]["n_pred"] == 0 and r["PAIR@all"]["R"] == 0.0


def test_swapped_direction_is_wrong():
    rec = _rec()
    r = _score(rec, _pred(rec, pairs=[{"sentence_id": 2, "source_idx": 2, "target_idx": 0}]))
    assert r["PAIR@all"]["tp"] == 0 and r["PAIR@all"]["n_pred"] == 1


def test_wrong_sentence_is_wrong():
    rec = _rec()
    r = _score(rec, _pred(rec, pairs=[{"sentence_id": 1, "source_idx": 0, "target_idx": 2}]))
    assert r["PAIR@all"]["tp"] == 0


def test_correct_pair_wrong_location_only_hurts_pair_loc():
    rec = _rec()
    p = pm.gold_as_prediction(rec)
    p["locations_after"][2] = ["-", "root"]          # sugar tạo ở 'root' thay vì 'leaf'
    r = _score(rec, p)
    assert r["PAIR@all"]["F1"] == 1.0 and r["PAIR_LOC"]["tp"] == 0


def test_duplicate_pairs_counted_once_and_empty_sets_safe():
    rec = _rec()
    dup = [{"sentence_id": 2, "source_idx": 0, "target_idx": 2}] * 3
    r = _score(rec, _pred(rec, pairs=dup))
    assert r["PAIR@all"]["n_pred"] == 1 and r["PAIR@all"]["P"] == 1.0 and r["PAIR@all"]["R"] == 0.5
    assert pm.prf(0, 0, 0)["F1"] == 0.0          # không chia cho 0


def test_destroy_twice_without_create_is_violation():
    rec = _rec()
    labels = [list(r) for r in rec["labels"]]
    labels[0] = [pc.DESTROY, pc.DESTROY]
    p = _pred(rec, labels=labels)
    p["exists_initial"] = [True, True, False, False]
    r = _score(rec, p)
    assert r["VIOLATION_RATE_per_entity"] == 1 / 4


def test_unknown_entity_index_raises():
    rec = _rec()
    try:
        _score(rec, _pred(rec, pairs=[{"sentence_id": 2, "source_idx": 0, "target_idx": 9}]))
    except pm.PredictionError:
        return
    raise AssertionError("phải báo PredictionError")


def test_bad_label_and_shape_raise():
    rec = _rec()
    for bad in ([["NONE", "BURN"]] + [["NONE", "NONE"]] * 3, [["NONE"]] * 4):
        try:
            _score(rec, _pred(rec, labels=bad))
        except pm.PredictionError:
            continue
        raise AssertionError("phải báo PredictionError")


def test_scoring_is_deterministic():
    rec = _rec()
    p = _pred(rec, pairs=[{"sentence_id": 2, "source_idx": 1, "target_idx": 2}])
    a, b = _score(rec, p), _score(rec, p)
    assert a["PAIR@all"] == b["PAIR@all"]


def test_missing_paragraph_strict():
    rec = _rec()
    try:
        pm.evaluate({}, [rec])
    except pm.PredictionError:
        return
    raise AssertionError("thiếu đoạn phải báo lỗi khi strict=True")


def test_slices_use_text_mentions_for_predictions():
    rec = _rec()
    # dự đoán cặp oxygen -> sugar: oxygen không được nhắc ở câu 2 nên thuộc lát src_implicit
    r = _score(rec, _pred(rec, pairs=[{"sentence_id": 2, "source_idx": 3, "target_idx": 2}]))
    assert r["PAIR@src_implicit"]["n_pred"] == 1 and r["PAIR@both_explicit"]["n_pred"] == 0


def test_label_derived_pairs_when_no_pairs_field():
    rec = _rec()
    p = pm.gold_as_prediction(rec)
    assert pm.predicted_pairs(p, rec) == {(2, 0, 2), (2, 1, 2)}


def test_action_rows_are_valid_even_for_violations():
    rec = _rec()
    labels = [[pc.DESTROY, pc.DESTROY], [pc.MOVE, pc.NONE], [pc.CREATE, pc.CREATE], [pc.NONE, pc.NONE]]
    rows = pm.to_official_action_rows({"paragraph_id": "t1", "labels": labels}, rec)
    for _, _, _, y, b, a in rows:
        if y == pc.NONE:
            assert b == a
        if y == pc.CREATE:
            assert b == "-" and a not in ("", "-")
        if y == pc.DESTROY:
            assert b not in ("", "-") and a == "-"
        if y == pc.MOVE:
            assert b not in ("", "-") and a not in ("", "-")


def test_gold_roundtrip_through_action_rows():
    rec = _rec()
    rows = pm.to_official_action_rows(pm.gold_as_prediction(rec), rec)
    assert pc.compare_action_rows(pc.to_action_rows(rec), rows)["n_mismatched_steps"] == 0
    back = pm.action_rows_to_prediction(rows, rec)
    assert back["labels"] == rec["labels"]


def test_paired_bootstrap_identical_systems_zero():
    rec = _rec()
    r = _score(rec, pm.gold_as_prediction(rec))
    bs = pm.paired_bootstrap_pair_f1(r, r, n_boot=200)
    assert bs["delta"] == 0 and bs["ci_low"] == 0 and bs["ci_high"] == 0


def test_mcnemar():
    assert pm.mcnemar_exact([True] * 10, [True] * 10)["p"] == 1.0
    assert pm.mcnemar_exact([True] * 10 + [False] * 0, [False] * 10)["p"] < 0.01


def run_all(verbose=True):
    names = sorted(n for n in globals() if n.startswith("test_"))
    failed = []
    for n in names:
        try:
            globals()[n]()
            if verbose:
                print(f"  PASS  {n}")
        except Exception as e:  # noqa: BLE001
            failed.append((n, repr(e)))
            print(f"  FAIL  {n}: {e!r}")
    return len(names), failed
