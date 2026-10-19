"""propara_llm — prompt, phân tích đầu ra và chuyển đầu ra LLM sang định dạng dự đoán chuẩn.

Phiên bản: llm_v1 (prompt_v1). Không phụ thuộc torch/transformers (kiểm thử được trên CPU).

Chế độ (tên khoá theo experiment_protocol):
  QWEN_DIRECT      (Q0) zero-shot, xuất các THAY ĐỔI + conversions (gọn)
  QWEN_FEWSHOT     (Q1) như Q0 + 2 ví dụ cố định lấy từ TRAIN
  QWEN_STRUCTURED  (Q2) xuất bảng đầy đủ: mỗi câu × mỗi thực thể: exists_before, action, exists_after, location_after
  QWEN_CONSTRAINED (Q3) = hậu xử lý luật tồn tại trên đầu ra Q0 (không gọi lại mô hình, không dùng vàng)
  ORACLE_PREV_STATE(Q4) ORACLE: một lời gọi cho mỗi câu t, cho biết trạng thái VÀNG trước câu t, hỏi thay đổi ở câu t
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import propara_core as pc

LLM_VERSION = "llm_v1"
PROMPT_VERSION = "prompt_v1"
MODES = ("QWEN_DIRECT", "QWEN_FEWSHOT", "QWEN_STRUCTURED", "ORACLE_PREV_STATE")
POSTPROC_MODES = {"QWEN_CONSTRAINED": "QWEN_DIRECT"}

SYSTEM = ("You are a careful annotator of procedural texts. You track the existence and location of given "
          "entities through each step of a process and answer ONLY with valid JSON.")

GUIDE = """Task: the paragraph describes a process, one numbered sentence per step. For EACH listed entity and EACH sentence, decide what happens to the entity in that sentence:
- CREATE: the entity did not exist before the sentence and exists after it (it is produced/formed).
- DESTROY: the entity existed before the sentence and no longer exists after it (it is consumed/used up/converted into something else).
- MOVE: the entity exists before and after but its location changes.
- NONE: nothing changes.
Important: an entity can be created or destroyed in a sentence even if the sentence does NOT mention it by name (for example "Sugar is made in the leaf." destroys carbon dioxide and water). Read the whole paragraph to infer such implicit changes.
Locations are short phrases copied from the paragraph when possible. Use "?" if the entity exists but its location is unknown, and "-" if it does not exist.
A conversion is a pair (source, target, sentence): the source entity is destroyed in that sentence to produce the target entity created in the same sentence.
Use ONLY the entity names exactly as listed. Do not invent entities."""

SCHEMA_COMPACT = """Answer with ONE JSON object and nothing else:
{"initial": {"<entity>": "<location before sentence 1, or ? or ->", ...},
 "changes": [{"sentence": <int>, "entity": "<entity>", "action": "CREATE|DESTROY|MOVE", "location_after": "<location or ? or ->"}, ...],
 "conversions": [{"sentence": <int>, "source": "<entity>", "target": "<entity>"}, ...]}
List only changes (omit NONE). Every entity must appear in "initial"."""

SCHEMA_STRUCTURED = """Answer with ONE JSON object and nothing else. Work step by step: for every sentence and EVERY entity give one row.
{"initial": {"<entity>": "<location before sentence 1, or ? or ->", ...},
 "steps": [{"sentence": <int>, "entities": [{"entity": "<entity>", "exists_before": true|false, "action": "CREATE|DESTROY|MOVE|NONE", "exists_after": true|false, "location_after": "<location or ? or ->"}, ...]}, ...],
 "conversions": [{"sentence": <int>, "source": "<entity>", "target": "<entity>"}, ...]}"""

SCHEMA_ORACLE = """Answer with ONE JSON object and nothing else, about sentence {t} ONLY:
{{"changes": [{{"entity": "<entity>", "action": "CREATE|DESTROY|MOVE", "location_after": "<location or ? or ->"}}, ...],
 "conversions": [{{"source": "<entity>", "target": "<entity>"}}, ...]}}
List only entities that change in sentence {t} (omit NONE)."""


def _paragraph_block(rec: dict) -> str:
    sents = "\n".join(f"{i}. {s}" for i, s in enumerate(rec["sentences"], 1))
    ents = "\n".join(f"- {e}" for e in rec["entities"])
    return f"Paragraph:\n{sents}\n\nEntities:\n{ents}"


def gold_compact_answer(rec: dict) -> str:
    """Đáp án mẫu (chỉ dùng cho ví dụ few-shot lấy từ TRAIN)."""
    init = {e: rec["states"][i][0] for i, e in enumerate(rec["entities"])}
    changes = []
    for t in range(1, len(rec["sentences"]) + 1):
        for i, e in enumerate(rec["entities"]):
            y = rec["labels"][i][t - 1]
            if y != pc.NONE:
                changes.append({"sentence": t, "entity": e, "action": y, "location_after": rec["states"][i][t]})
    conv = [{"sentence": g["sentence_id"], "source": g["source"], "target": g["target"]} for g in pc.gold_pairs_report(rec)]
    return json.dumps({"initial": init, "changes": changes, "conversions": conv}, ensure_ascii=False)


def select_fewshot(train_recs: List[dict], k: int = 2) -> List[dict]:
    """Chọn tất định k ví dụ TRAIN: 3–4 thực thể, ≤ 6 câu, có ≥1 cặp có nguồn ngầm; xếp theo băm ổn định."""
    pool = []
    for r in train_recs:
        if r["split"] != "train":
            raise ValueError("Ví dụ few-shot chỉ được lấy từ train")
        if 3 <= len(r["entities"]) <= 4 and len(r["sentences"]) <= 6:
            gp = pc.gold_pairs_report(r)
            if any(g["source_implicit"] for g in gp):
                pool.append(r)
    pool.sort(key=lambda r: pc.stable_hash("fewshot:" + r["paragraph_id"]))
    return pool[:k]


def build_messages(rec: dict, mode: str, fewshot: Optional[List[dict]] = None, t: Optional[int] = None) -> List[Dict[str, str]]:
    if mode in ("QWEN_DIRECT", "QWEN_FEWSHOT"):
        msgs = [{"role": "system", "content": SYSTEM}]
        if mode == "QWEN_FEWSHOT":
            for ex in fewshot or []:
                msgs.append({"role": "user", "content": f"{GUIDE}\n\n{_paragraph_block(ex)}\n\n{SCHEMA_COMPACT}"})
                msgs.append({"role": "assistant", "content": gold_compact_answer(ex)})
        msgs.append({"role": "user", "content": f"{GUIDE}\n\n{_paragraph_block(rec)}\n\n{SCHEMA_COMPACT}"})
        return msgs
    if mode == "QWEN_STRUCTURED":
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{GUIDE}\n\n{_paragraph_block(rec)}\n\n{SCHEMA_STRUCTURED}"}]
    if mode == "ORACLE_PREV_STATE":
        if t is None:
            raise ValueError("ORACLE_PREV_STATE cần t")
        state = "\n".join(
            f"- {e}: " + ("does not exist" if rec["states"][i][t - 1] == pc.NO_LOC else
                          f"exists, location {rec['states'][i][t - 1]}" if rec["states"][i][t - 1] != pc.UNK_LOC else
                          "exists, location unknown")
            for i, e in enumerate(rec["entities"]))
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{GUIDE}\n\n{_paragraph_block(rec)}\n\n"
                                            f"Correct state of every entity just BEFORE sentence {t} (given to you):\n{state}\n\n"
                                            + SCHEMA_ORACLE.format(t=t)}]
    raise ValueError(mode)


def max_new_tokens(rec: dict, mode: str) -> int:
    n, T = len(rec["entities"]), len(rec["sentences"])
    if mode == "QWEN_STRUCTURED":
        return min(4096, 200 + 45 * n * T)
    if mode == "ORACLE_PREV_STATE":
        return min(768, 120 + 40 * n)
    return min(2048, 200 + 18 * n * T)


# ----------------------------------------------------------------------------- phân tích JSON
def extract_json(text: str) -> Tuple[Optional[dict], str]:
    """Trả về (obj, trạng_thái) với trạng_thái ∈ {valid, repaired, truncated, invalid}."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    start = s.find("{")
    if start < 0:
        return None, "invalid"
    depth, in_str, esc, end = 0, False, False, None
    for k in range(start, len(s)):
        ch = s[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = k + 1
                break
    if end is None:
        return None, "truncated"
    block = s[start:end]
    try:
        return json.loads(block), "valid"
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", block)                     # dấu phẩy thừa
    fixed = re.sub(r"(?<=[{,])\s*'([^']*)'\s*:", r'"\1":', fixed)    # khoá dùng nháy đơn
    fixed = fixed.replace("True", "true").replace("False", "false")
    try:
        return json.loads(fixed), "repaired"
    except json.JSONDecodeError:
        return None, "invalid"


def _norm_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", s))


def entity_resolver(rec: dict):
    """Ánh xạ tên do LLM viết → chỉ số thực thể. Khớp chính xác, rồi theo tên gọi khác, rồi chuẩn hoá nhẹ."""
    exact = {}
    norm = {}
    for i, e in enumerate(rec["entities"]):
        exact.setdefault(e.strip().lower(), i)
        for a in pc.split_aliases(e):
            exact.setdefault(a.strip().lower(), i)
            norm.setdefault(_norm_name(a), i)
        norm.setdefault(_norm_name(e), i)

    def resolve(name) -> Optional[int]:
        if not isinstance(name, str):
            return None
        k = name.strip().lower()
        if k in exact:
            return exact[k]
        return norm.get(_norm_name(name))
    return resolve


def _clean_loc(v) -> str:
    if v is None:
        return "?"
    v = str(v).strip()
    return v if v else "?"


def parse_to_prediction(text: str, rec: dict, mode: str) -> Tuple[dict, dict]:
    """Đầu ra thô → (dự đoán chuẩn, chẩn đoán). Không bao giờ dùng vàng.
    Thực thể lạ → bỏ qua và đếm 'hallucinated'. Nhãn lạ → bỏ qua, đếm 'bad_action'."""
    n, T = len(rec["entities"]), len(rec["sentences"])
    obj, status = extract_json(text)
    diag = {"json_status": status, "hallucinated": 0, "bad_action": 0, "bad_sentence": 0, "missing_initial": 0,
            "n_changes": 0, "n_conversions": 0}
    labels = [[pc.NONE] * T for _ in range(n)]
    loc_after = [[None] * T for _ in range(n)]
    init_loc = [None] * n
    pairs = []
    if obj is None or not isinstance(obj, dict):
        pred = {"paragraph_id": rec["paragraph_id"], "labels": labels}
        return pred, diag
    res = entity_resolver(rec)

    init = obj.get("initial") if isinstance(obj.get("initial"), dict) else {}
    for name, loc in init.items():
        i = res(name)
        if i is None:
            diag["hallucinated"] += 1
        else:
            init_loc[i] = _clean_loc(loc)
    diag["missing_initial"] = sum(v is None for v in init_loc)

    def put(i, t, action, loc):
        if action not in (pc.CREATE, pc.DESTROY, pc.MOVE, pc.NONE):
            diag["bad_action"] += 1
            return
        labels[i][t - 1] = action
        loc_after[i][t - 1] = _clean_loc(loc) if action != pc.DESTROY else pc.NO_LOC

    if mode == "QWEN_STRUCTURED":
        for st in obj.get("steps") or []:
            t = st.get("sentence") if isinstance(st, dict) else None
            if not isinstance(t, int) or not 1 <= t <= T:
                diag["bad_sentence"] += 1
                continue
            for row in st.get("entities") or []:
                i = res(row.get("entity")) if isinstance(row, dict) else None
                if i is None:
                    diag["hallucinated"] += 1
                    continue
                act = str(row.get("action", "")).upper()
                put(i, t, act, row.get("location_after"))
                if act != pc.NONE:
                    diag["n_changes"] += 1
                if t == 1 and init_loc[i] is None and isinstance(row.get("exists_before"), bool):
                    init_loc[i] = "?" if row["exists_before"] else pc.NO_LOC
    else:
        for ch in obj.get("changes") or []:
            if not isinstance(ch, dict):
                continue
            t = ch.get("sentence")
            if isinstance(t, str) and t.isdigit():
                t = int(t)
            if not isinstance(t, int) or not 1 <= t <= T:
                diag["bad_sentence"] += 1
                continue
            i = res(ch.get("entity"))
            if i is None:
                diag["hallucinated"] += 1
                continue
            put(i, t, str(ch.get("action", "")).upper(), ch.get("location_after"))
            diag["n_changes"] += 1

    for cv in obj.get("conversions") or []:
        if not isinstance(cv, dict):
            continue
        t = cv.get("sentence")
        if isinstance(t, str) and t.isdigit():
            t = int(t)
        x, y = res(cv.get("source")), res(cv.get("target"))
        if not isinstance(t, int) or not 1 <= t <= T:
            diag["bad_sentence"] += 1
            continue
        if x is None or y is None:
            diag["hallucinated"] += 1
            continue
        if x != y:
            pairs.append({"sentence_id": t, "source_idx": x, "target_idx": y})
    diag["n_conversions"] = len(pairs)
    return _assemble(rec, labels, loc_after, init_loc, pairs), diag


def _assemble(rec, labels, loc_after, init_loc, pairs) -> dict:
    n, T = len(rec["entities"]), len(rec["sentences"])
    exists_initial = []
    for i in range(n):
        if init_loc[i] is not None:
            exists_initial.append(init_loc[i] != pc.NO_LOC)
        else:
            first = next((y for y in labels[i] if y != pc.NONE), pc.NONE)
            exists_initial.append(first != pc.CREATE)
    la = []
    for i in range(n):
        cur = (init_loc[i] if init_loc[i] is not None else ("?" if exists_initial[i] else pc.NO_LOC))
        row = []
        for t in range(T):
            y = labels[i][t]
            if y == pc.NONE:
                row.append(cur)
            elif y == pc.DESTROY:
                cur = pc.NO_LOC
                row.append(cur)
            else:
                cur = loc_after[i][t] if loc_after[i][t] not in (None, pc.NO_LOC) else "?"
                row.append(cur)
        la.append(row)
    return {"paragraph_id": rec["paragraph_id"], "labels": labels, "exists_initial": exists_initial,
            "location_initial": [init_loc[i] if init_loc[i] is not None else ("?" if exists_initial[i] else pc.NO_LOC) for i in range(n)],
            "locations_after": la, "pairs_llm": pairs}


def parse_oracle_step(text: str, rec: dict, t: int) -> Tuple[Dict[int, Tuple[str, str]], List[Tuple[int, int]], dict]:
    obj, status = extract_json(text)
    diag = {"json_status": status, "hallucinated": 0, "bad_action": 0}
    changes, pairs = {}, []
    if not isinstance(obj, dict):
        return changes, pairs, diag
    res = entity_resolver(rec)
    for ch in obj.get("changes") or []:
        if not isinstance(ch, dict):
            continue
        i = res(ch.get("entity"))
        act = str(ch.get("action", "")).upper()
        if i is None:
            diag["hallucinated"] += 1
        elif act not in (pc.CREATE, pc.DESTROY, pc.MOVE, pc.NONE):
            diag["bad_action"] += 1
        else:
            changes[i] = (act, _clean_loc(ch.get("location_after")))
    for cv in obj.get("conversions") or []:
        if isinstance(cv, dict):
            x, y = res(cv.get("source")), res(cv.get("target"))
            if x is None or y is None:
                diag["hallucinated"] += 1
            elif x != y:
                pairs.append((x, y))
    return changes, pairs, diag


def assemble_oracle(rec: dict, per_step: Dict[int, Tuple[dict, list]]) -> dict:
    """Ghép các lời gọi theo câu thành một dự đoán (ORACLE: trạng thái ban đầu lấy từ vàng)."""
    n, T = len(rec["entities"]), len(rec["sentences"])
    labels = [[pc.NONE] * T for _ in range(n)]
    loc_after = [[None] * T for _ in range(n)]
    pairs = []
    for t, (changes, prs) in per_step.items():
        for i, (act, loc) in changes.items():
            labels[i][t - 1] = act
            loc_after[i][t - 1] = loc if act != pc.DESTROY else pc.NO_LOC
        pairs += [{"sentence_id": t, "source_idx": x, "target_idx": y} for x, y in prs]
    init_loc = [rec["states"][i][0] for i in range(n)]
    return _assemble(rec, labels, loc_after, init_loc, pairs)


# ----------------------------------------------------------------------------- Q3: hậu xử lý luật tồn tại (không dùng vàng)
def apply_existence_constraints(pred: dict) -> Tuple[dict, int]:
    """Đi theo thứ tự câu với tồn tại tự tiến hoá; nhãn không hợp lệ → NONE.
    Trả về (dự đoán mới, số ô bị sửa). Cặp LLM tự khai báo giữ nguyên nếu nguồn/đích còn nhãn D/C tương ứng."""
    labels = [list(r) for r in pred["labels"]]
    init = [bool(x) for x in pred.get("exists_initial") or [True] * len(labels)]
    fixed = 0
    for i, row in enumerate(labels):
        e = init[i]
        for t, y in enumerate(row):
            if y not in pc.VALID_LABELS[e]:
                row[t] = pc.NONE
                fixed += 1
                continue
            if y == pc.CREATE:
                e = True
            elif y == pc.DESTROY:
                e = False
    out = dict(pred, labels=labels)
    if pred.get("pairs_llm") is not None:
        out["pairs_llm"] = [p for p in pred["pairs_llm"]
                            if labels[p["source_idx"]][p["sentence_id"] - 1] == pc.DESTROY
                            and labels[p["target_idx"]][p["sentence_id"] - 1] == pc.CREATE]
    return out, fixed


def apply_existence_constraints_v2(pred: dict) -> Tuple[dict, int]:
    """QWEN_CONSTRAINED_V2 (đăng ký 19/10 ~23h50, SAU khi thấy v1 làm sập PAIR_F1 trên dev, TRƯỚC khi có số của v2;
    vẫn không dùng vàng):
    (1) sửa TỒN TẠI BAN ĐẦU trước: thực thể có sự kiện đầu tiên là CREATE → ban đầu không tồn tại; là DESTROY/MOVE → tồn tại;
    (2) sau đó mới áp luật v1 (nhãn còn không hợp lệ → NONE).
    Lý do: phần lớn vi phạm của Qwen là khai 'initial' sai (sản phẩm được gán vị trí ban đầu) chứ không phải nhãn sai."""
    labels = pred["labels"]
    init = list(pred.get("exists_initial") or [True] * len(labels))
    for i, row in enumerate(labels):
        first = next((y for y in row if y != pc.NONE), None)
        if first == pc.CREATE:
            init[i] = False
        elif first in (pc.DESTROY, pc.MOVE):
            init[i] = True
    out, fixed = apply_existence_constraints(dict(pred, exists_initial=init))
    out["exists_initial"] = init
    if pred.get("location_initial") is not None:
        out["location_initial"] = [("-" if not e else (v if v != "-" else "?")) for e, v in zip(init, pred["location_initial"])]
    return out, fixed


def as_eval_prediction(pred: dict, pair_source: str = "labels") -> dict:
    """pair_source='labels': cặp = D̂×Ĉ (thước đo CHÍNH, tương thích định nghĩa vàng).
       pair_source='llm':    cặp = conversions LLM tự khai báo (thước đo phụ)."""
    p = {k: v for k, v in pred.items() if k != "pairs_llm"}
    if pair_source == "llm":
        p["pairs"] = pred.get("pairs_llm") or []
    return p
