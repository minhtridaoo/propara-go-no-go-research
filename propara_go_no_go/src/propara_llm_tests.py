"""Kiểm thử propara_llm (không cần GPU)."""
import json

import propara_core as pc
import propara_llm as pl
import propara_metrics as pm


def _rec(split="train"):
    raw = {"para_id": "t1",
           "sentence_texts": ["Water and carbon dioxide enter the leaf.", "Sugar is made.", "The sugar moves to the root."],
           "participants": ["water", "carbon dioxide; CO2", "sugar", "oxygen"],
           "states": [["root", "leaf", "-", "-"], ["air", "leaf", "-", "-"], ["-", "-", "leaf", "root"], ["-", "-", "-", "-"]]}
    return pc.build_record(raw, split)


def test_gold_answer_roundtrip_gives_perfect_scores():
    rec = _rec()
    pred, diag = pl.parse_to_prediction(pl.gold_compact_answer(rec), rec, "QWEN_DIRECT")
    assert diag["json_status"] == "valid" and diag["hallucinated"] == 0
    assert pred["labels"] == rec["labels"]
    r = pm.evaluate({"t1": pl.as_eval_prediction(pred)}, [rec])
    assert r["PAIR@all"]["F1"] == 1.0 and r["EXIST_ACC"] == 1.0
    r2 = pm.evaluate({"t1": pl.as_eval_prediction(pred, "llm")}, [rec])
    assert r2["PAIR@all"]["F1"] == 1.0


def test_json_in_code_fence_and_trailing_text():
    obj, st = pl.extract_json('Sure!\n```json\n{"a": 1}\n```\nHope this helps')
    assert obj == {"a": 1} and st == "valid"


def test_truncated_and_repaired_json():
    assert pl.extract_json('{"changes": [{"sentence": 1')[1] == "truncated"
    obj, st = pl.extract_json('{"changes": [1, 2,], "x": True}')
    assert st == "repaired" and obj == {"changes": [1, 2], "x": True}
    assert pl.extract_json("no json here")[1] == "invalid"


def test_alias_and_fuzzy_entity_resolution_and_hallucination():
    rec = _rec()
    text = json.dumps({"initial": {"water": "root", "CO2": "air", "Sugar": "-", "oxygen": "-"},
                       "changes": [{"sentence": 2, "entity": "the waters", "action": "DESTROY", "location_after": "-"},
                                   {"sentence": 2, "entity": "glucose", "action": "CREATE", "location_after": "leaf"}],
                       "conversions": [{"sentence": 2, "source": "co2", "target": "sugar"}]})
    pred, diag = pl.parse_to_prediction(text, rec, "QWEN_DIRECT")
    assert pred["labels"][0][1] == pc.DESTROY            # 'the waters' -> water
    assert diag["hallucinated"] == 1                     # glucose không có trong danh sách
    assert pred["pairs_llm"] == [{"sentence_id": 2, "source_idx": 1, "target_idx": 2}]
    assert pred["exists_initial"] == [True, True, False, False]


def test_invalid_output_gives_all_none_not_crash():
    rec = _rec()
    pred, diag = pl.parse_to_prediction("I cannot answer.", rec, "QWEN_DIRECT")
    assert diag["json_status"] == "invalid" and all(y == pc.NONE for row in pred["labels"] for y in row)
    pm.evaluate({"t1": pl.as_eval_prediction(pred)}, [rec])    # vẫn chấm được


def test_bad_sentence_and_action_counted():
    rec = _rec()
    text = json.dumps({"initial": {}, "changes": [{"sentence": 9, "entity": "water", "action": "DESTROY"},
                                                  {"sentence": 1, "entity": "water", "action": "EVAPORATE"}]})
    _, diag = pl.parse_to_prediction(text, rec, "QWEN_DIRECT")
    assert diag["bad_sentence"] == 1 and diag["bad_action"] == 1 and diag["missing_initial"] == 4


def test_structured_mode_parsing():
    rec = _rec()
    steps = []
    for t in range(1, 4):
        rows = []
        for i, e in enumerate(rec["entities"]):
            rows.append({"entity": e.split(";")[0], "exists_before": rec["existence"][i][t - 1],
                         "action": rec["labels"][i][t - 1], "exists_after": rec["existence"][i][t],
                         "location_after": rec["states"][i][t]})
        steps.append({"sentence": t, "entities": rows})
    pred, diag = pl.parse_to_prediction(json.dumps({"initial": {}, "steps": steps, "conversions": []}), rec, "QWEN_STRUCTURED")
    assert pred["labels"] == rec["labels"] and pred["exists_initial"] == [True, True, False, False]


def test_constraint_postprocessing_without_gold():
    rec = _rec()
    pred = {"paragraph_id": "t1", "labels": [[pc.DESTROY, pc.DESTROY, pc.NONE], [pc.NONE] * 3,
                                             [pc.MOVE, pc.CREATE, pc.MOVE], [pc.NONE] * 3],
            "exists_initial": [True, True, False, False],
            "pairs_llm": [{"sentence_id": 2, "source_idx": 0, "target_idx": 2}]}
    out, fixed = pl.apply_existence_constraints(pred)
    assert fixed == 2                                   # DESTROY lần 2 + MOVE khi chưa tồn tại
    assert out["labels"][0] == [pc.DESTROY, pc.NONE, pc.NONE]
    assert out["labels"][2] == [pc.NONE, pc.CREATE, pc.MOVE]
    assert out["pairs_llm"] == []                       # nguồn không còn DESTROY ở câu 2
    assert pc.count_constraint_violations(out["labels"], out["exists_initial"])[0] == 0


def test_oracle_prompt_contains_only_state_before_t():
    rec = _rec()
    msg = pl.build_messages(rec, "ORACLE_PREV_STATE", t=2)[-1]["content"]
    assert "water: exists, location leaf" in msg          # trạng thái sau câu 1 = trước câu 2
    assert "sugar: does not exist" in msg                 # chưa lộ việc sugar được tạo ở câu 2


def test_oracle_assembly():
    rec = _rec()
    steps = {}
    for t in range(1, 4):
        ch, pr, _ = pl.parse_oracle_step(json.dumps({"changes": [{"entity": "sugar", "action": "CREATE", "location_after": "leaf"}]
                                                      if t == 2 else [], "conversions": []}), rec, t)
        steps[t] = (ch, pr)
    pred = pl.assemble_oracle(rec, steps)
    assert pred["labels"][2] == [pc.NONE, pc.CREATE, pc.NONE]


def test_fewshot_only_from_train():
    try:
        pl.select_fewshot([_rec("dev")])
    except ValueError:
        return
    raise AssertionError("phải từ chối ví dụ không thuộc train")


def test_fewshot_prompt_has_examples_then_target():
    rec = _rec()
    msgs = pl.build_messages(rec, "QWEN_FEWSHOT", fewshot=[rec])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


def test_constraint_v2_fixes_initial_existence_first():
    rec = _rec()
    # Qwen khai sugar tồn tại từ đầu nhưng lại CREATE ở câu 2 → v1 xoá CREATE, v2 sửa tồn tại ban đầu và giữ CREATE
    pred = {"paragraph_id": "t1", "labels": [[pc.NONE, pc.DESTROY, pc.NONE], [pc.NONE, pc.DESTROY, pc.NONE],
                                             [pc.NONE, pc.CREATE, pc.MOVE], [pc.NONE] * 3],
            "exists_initial": [True, True, True, True]}
    v1, _ = pl.apply_existence_constraints(pred)
    v2, _ = pl.apply_existence_constraints_v2(pred)
    assert v1["labels"][2][1] == pc.NONE and v2["labels"][2][1] == pc.CREATE
    assert v2["exists_initial"][2] is False
    assert pc.count_constraint_violations(v2["labels"], v2["exists_initial"])[0] == 0


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
