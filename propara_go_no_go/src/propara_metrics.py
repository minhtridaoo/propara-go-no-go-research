"""propara_metrics — bộ chấm chung cho mọi hệ thống (heuristic, Qwen, mô hình nhóm).

Phiên bản: metrics_v1. Phụ thuộc: propara_core (core_v1). Chỉ dùng thư viện chuẩn.

ĐỊNH DẠNG DỰ ĐOÁN (một dict cho mỗi đoạn):
  {
    "paragraph_id": "1002",
    "labels": [[nhãn câu 1..T] cho thực thể 0], ..., [..] cho thực thể n-1]   # BẮT BUỘC, n x T
    "exists_initial": [bool]*n            # tuỳ chọn; thiếu thì suy từ nhãn (xem infer_initial_existence)
    "location_initial": [str]*n           # tuỳ chọn; '-' không tồn tại, '?' không rõ
    "locations_after": [[str]*T]*n        # tuỳ chọn; vị trí SAU câu t
    "pairs": [{"sentence_id", "source_idx", "target_idx"}]   # tuỳ chọn; thiếu thì = D̂_t x Ĉ_t
  }
Thực thể định danh bằng CHỈ SỐ trong danh sách thực thể của đoạn (như dữ liệu chuẩn của notebook 00).
"""
from __future__ import annotations

import io
import math
import random
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import propara_core as pc

METRICS_VERSION = "metrics_v1"
SLICES = ("all", "src_implicit", "tgt_implicit", "any_implicit", "both_implicit", "both_explicit")


class PredictionError(ValueError):
    """Lỗi cấu trúc của dự đoán (sai kích thước, nhãn lạ, chỉ số thực thể ngoài danh sách...)."""


# ----------------------------------------------------------------------------- kiểm tra cấu trúc
def validate_prediction(pred: dict, rec: dict) -> None:
    n, T = len(rec["entities"]), len(rec["sentences"])
    pid = rec["paragraph_id"]
    if str(pred.get("paragraph_id")) != pid:
        raise PredictionError(f"{pid}: paragraph_id dự đoán = {pred.get('paragraph_id')!r}")
    labs = pred.get("labels")
    if not isinstance(labs, list) or len(labs) != n:
        raise PredictionError(f"{pid}: labels phải có {n} hàng (số thực thể)")
    for i, row in enumerate(labs):
        if not isinstance(row, list) or len(row) != T:
            raise PredictionError(f"{pid}: labels[{i}] phải có {T} phần tử (số câu)")
        for y in row:
            if y not in pc.LABELS:
                raise PredictionError(f"{pid}: nhãn lạ {y!r} ở thực thể {i}")
    for key, length in (("exists_initial", n), ("location_initial", n)):
        if key in pred and pred[key] is not None and len(pred[key]) != length:
            raise PredictionError(f"{pid}: {key} phải có {length} phần tử")
    if pred.get("locations_after") is not None:
        la = pred["locations_after"]
        if len(la) != n or any(len(r) != T for r in la):
            raise PredictionError(f"{pid}: locations_after phải là {n} x {T}")
    for p in pred.get("pairs") or []:
        s, x, y = p.get("sentence_id"), p.get("source_idx"), p.get("target_idx")
        if not (isinstance(s, int) and 1 <= s <= T):
            raise PredictionError(f"{pid}: sentence_id ngoài [1,{T}]: {s!r}")
        for v in (x, y):
            if not (isinstance(v, int) and 0 <= v < n):
                raise PredictionError(f"{pid}: chỉ số thực thể ngoài danh sách: {v!r}")
        if x == y:
            raise PredictionError(f"{pid}: cặp tự thân ({x}->{y})")


def infer_initial_existence(labels_row: Sequence[str]) -> bool:
    """Khi hệ thống không cho tồn tại ban đầu: tồn tại trừ khi thay đổi đầu tiên là CREATE."""
    for y in labels_row:
        if y != pc.NONE:
            return y != pc.CREATE
    return True


def pred_initial_existence(pred: dict) -> List[bool]:
    if pred.get("exists_initial") is not None:
        return [bool(v) for v in pred["exists_initial"]]
    if pred.get("location_initial") is not None:
        return [v != pc.NO_LOC for v in pred["location_initial"]]
    return [infer_initial_existence(r) for r in pred["labels"]]


def gold_as_prediction(rec: dict) -> dict:
    return {"paragraph_id": rec["paragraph_id"], "labels": [list(r) for r in rec["labels"]],
            "exists_initial": [e[0] for e in rec["existence"]],
            "location_initial": [s[0] for s in rec["states"]],
            "locations_after": [s[1:] for s in rec["states"]]}


# ----------------------------------------------------------------------------- cặp
def predicted_pairs(pred: dict, rec: dict) -> set:
    if pred.get("pairs") is not None:
        return {(p["sentence_id"], p["source_idx"], p["target_idx"]) for p in pred["pairs"]}
    out = set()
    n = len(rec["entities"])
    for t in range(1, len(rec["sentences"]) + 1):
        D = [i for i in range(n) if pred["labels"][i][t - 1] == pc.DESTROY]
        C = [i for i in range(n) if pred["labels"][i][t - 1] == pc.CREATE]
        out.update((t, x, y) for x in D for y in C)
    return out


def gold_pairs(rec: dict) -> set:
    return {(g["sentence_id"], g["source_idx"], g["target_idx"]) for g in pc.gold_pairs_report(rec)}


def pair_slice(rec: dict, t: int, x: int, y: int) -> Dict[str, bool]:
    """Lát cắt theo cờ nhắc TÍNH TỪ VĂN BẢN (không cần vàng) nên áp dụng được cho dự đoán."""
    sx = not rec["mentions"][x][t - 1]
    sy = not rec["mentions"][y][t - 1]
    return {"all": True, "src_implicit": sx, "tgt_implicit": sy, "any_implicit": sx or sy,
            "both_implicit": sx and sy, "both_explicit": (not sx) and (not sy)}


def pair_counts(pred: dict, rec: dict) -> Dict[str, Tuple[int, int, int]]:
    """{slice: (tp, n_pred, n_gold)} cho một đoạn."""
    P, G = predicted_pairs(pred, rec), gold_pairs(rec)
    out = {s: [0, 0, 0] for s in SLICES}
    for (t, x, y) in P:
        sl = pair_slice(rec, t, x, y)
        for s in SLICES:
            if sl[s]:
                out[s][1] += 1
                if (t, x, y) in G:
                    out[s][0] += 1
    for (t, x, y) in G:
        sl = pair_slice(rec, t, x, y)
        for s in SLICES:
            if sl[s]:
                out[s][2] += 1
    return {s: tuple(v) for s, v in out.items()}


def prf(tp: int, npred: int, ngold: int) -> Dict[str, float]:
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": p, "R": r, "F1": f, "tp": tp, "n_pred": npred, "n_gold": ngold}


def _norm_loc(s: str) -> str:
    s = s.strip().lower()
    for a in ("the ", "a ", "an ", "its ", "their "):
        if s.startswith(a):
            s = s[len(a):]
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", s))


def pair_loc_counts(pred: dict, rec: dict) -> Tuple[int, int, int]:
    """PAIR_LOC: cặp + vị trí sau của đích (chuẩn hoá nhẹ). Cần locations_after; thiếu thì vị trí = '?'."""
    la = pred.get("locations_after")
    P = {(t, x, y, _norm_loc(la[y][t - 1]) if la else "?") for (t, x, y) in predicted_pairs(pred, rec)}
    G = {(t, x, y, _norm_loc(rec["states"][y][t])) for (t, x, y) in gold_pairs(rec)}
    return len(P & G), len(P), len(G)


# ----------------------------------------------------------------------------- nhãn / tồn tại / ràng buộc
def cell_metrics(pred: dict, rec: dict) -> Dict[str, object]:
    n, T = len(rec["entities"]), len(rec["sentences"])
    conf = Counter()          # (gold, pred)
    imp = Counter()           # (gold_label, correct?) trên ô vàng ngầm
    exist_ok = exist_tot = 0
    init = pred_initial_existence(pred)
    for i in range(n):
        e = init[i]
        for t in range(1, T + 1):
            g, y = rec["labels"][i][t - 1], pred["labels"][i][t - 1]
            conf[(g, y)] += 1
            if rec["implicit"][i][t - 1]:
                imp[(g, g == y)] += 1
            if y == pc.CREATE:
                e = True
            elif y == pc.DESTROY:
                e = False
            exist_ok += int(e == rec["existence"][i][t])
            exist_tot += 1
    viol, tot = pc.count_constraint_violations(pred["labels"], init)
    return {"confusion": conf, "implicit": imp, "exist_ok": exist_ok, "exist_tot": exist_tot,
            "viol": viol, "viol_cells": tot, "n_entities": n}


# ----------------------------------------------------------------------------- tổng hợp
def evaluate(preds: Dict[str, dict], recs: List[dict], strict: bool = True) -> Dict[str, object]:
    """Chấm một hệ thống trên danh sách đoạn. strict=True: thiếu đoạn -> lỗi.
    strict=False: thiếu đoạn -> coi như dự đoán toàn NONE và ghi vào 'missing'."""
    tot = {s: [0, 0, 0] for s in SLICES}
    loc = [0, 0, 0]
    conf, imp = Counter(), Counter()
    ex_ok = ex_tot = viol = viol_cells = n_ent = 0
    missing, per_para = [], {}
    for rec in recs:
        pid = rec["paragraph_id"]
        pred = preds.get(pid)
        if pred is None:
            if strict:
                raise PredictionError(f"Thiếu dự đoán cho đoạn {pid}")
            missing.append(pid)
            pred = all_none_prediction(rec)
        validate_prediction(pred, rec)
        pcnt = pair_counts(pred, rec)
        for s in SLICES:
            for k in range(3):
                tot[s][k] += pcnt[s][k]
        lc = pair_loc_counts(pred, rec)
        for k in range(3):
            loc[k] += lc[k]
        cm = cell_metrics(pred, rec)
        conf.update(cm["confusion"])
        imp.update(cm["implicit"])
        ex_ok += cm["exist_ok"]; ex_tot += cm["exist_tot"]
        viol += cm["viol"]; viol_cells += cm["viol_cells"]; n_ent += cm["n_entities"]
        per_para[pid] = {"pairs": pcnt, "viol": cm["viol"]}
    res = {"metrics_version": METRICS_VERSION, "n_paragraphs": len(recs), "missing": missing}
    for s in SLICES:
        res[f"PAIR@{s}"] = prf(*tot[s])
    res["PAIR_LOC"] = prf(*loc)
    lab = {}
    for L in pc.LABELS:
        tp = conf[(L, L)]
        npred = sum(v for (g, y), v in conf.items() if y == L)
        ngold = sum(v for (g, y), v in conf.items() if g == L)
        lab[L] = prf(tp, npred, ngold)
    res["LABEL"] = lab
    res["LABEL_MACRO_F1_CDM"] = sum(lab[L]["F1"] for L in (pc.CREATE, pc.DESTROY, pc.MOVE)) / 3
    res["IMPLICIT_ACC"] = {L: {"correct": imp[(L, True)], "n": imp[(L, True)] + imp[(L, False)],
                               "acc": (imp[(L, True)] / (imp[(L, True)] + imp[(L, False)]))
                               if imp[(L, True)] + imp[(L, False)] else float("nan")}
                           for L in (pc.CREATE, pc.DESTROY, pc.MOVE)}
    res["EXIST_ACC"] = ex_ok / ex_tot if ex_tot else float("nan")
    res["VIOLATION_RATE_per_entity"] = viol / n_ent if n_ent else float("nan")   # đúng công thức (28)
    res["VIOLATION_RATE_per_cell"] = viol / viol_cells if viol_cells else float("nan")
    res["_per_paragraph"] = per_para
    return res


def all_none_prediction(rec: dict) -> dict:
    n, T = len(rec["entities"]), len(rec["sentences"])
    return {"paragraph_id": rec["paragraph_id"], "labels": [[pc.NONE] * T for _ in range(n)]}


def summary_row(name: str, res: dict, extra: Optional[dict] = None) -> Dict[str, object]:
    r = {"system": name,
         "PAIR_P": res["PAIR@all"]["P"], "PAIR_R": res["PAIR@all"]["R"], "PAIR_F1": res["PAIR@all"]["F1"],
         "PAIR_F1@src_imp": res["PAIR@src_implicit"]["F1"], "PAIR_F1@tgt_imp": res["PAIR@tgt_implicit"]["F1"],
         "PAIR_F1@any_imp": res["PAIR@any_implicit"]["F1"], "PAIR_LOC_F1": res["PAIR_LOC"]["F1"],
         "F1_CREATE": res["LABEL"]["CREATE"]["F1"], "F1_DESTROY": res["LABEL"]["DESTROY"]["F1"],
         "F1_MOVE": res["LABEL"]["MOVE"]["F1"], "IMP_ACC_CREATE": res["IMPLICIT_ACC"]["CREATE"]["acc"],
         "IMP_ACC_DESTROY": res["IMPLICIT_ACC"]["DESTROY"]["acc"], "EXIST_ACC": res["EXIST_ACC"],
         "VIOL_per_entity": res["VIOLATION_RATE_per_entity"]}
    r.update(extra or {})
    return r


# ----------------------------------------------------------------------------- xuất action file cho evaluator chính thức
def to_official_action_rows(pred: dict, rec: dict) -> List[Tuple[str, int, str, str, str, str]]:
    """Chuyển dự đoán thành các dòng hợp lệ theo luật của evaluator (README aristo-leaderboard).
    Vị trí thiếu -> '?'. Dự đoán vi phạm ràng buộc vẫn được ghi ra dòng HỢP LỆ (ví dụ DESTROY khi
    đang không tồn tại -> before='?'), vì định dạng chính thức không biểu diễn được vi phạm.
    VIOLATION_RATE phải đọc từ evaluate(), không từ file này."""
    n, T = len(rec["entities"]), len(rec["sentences"])
    init_e = pred_initial_existence(pred)
    li = pred.get("location_initial")
    la = pred.get("locations_after")
    rows = []
    for i, ent in enumerate(rec["entities"]):
        if not init_e[i]:
            cur = "-"
        else:
            cur = li[i] if li and li[i] not in (None, "", "-") else "?"
        for t in range(1, T + 1):
            y = pred["labels"][i][t - 1]
            want = (la[i][t - 1] if la else None)
            if y == pc.NONE:
                # evaluator chỉ dùng cột 'after' (và 'before' của câu 1); giữ vị trí hệ thống đã dự đoán nếu có
                if want not in (None, "") and (want == pc.NO_LOC) == (cur == pc.NO_LOC):
                    cur = want
                before = after = cur
            elif y == pc.CREATE:
                before = "-"
                after = want if want not in (None, "", "-") else "?"
            elif y == pc.DESTROY:
                before = cur if cur != "-" else "?"
                after = "-"
            else:  # MOVE
                before = cur if cur != "-" else "?"
                after = want if want not in (None, "", "-") else "?"
            rows.append((rec["paragraph_id"], t, ent, y, before, after))
            cur = after
    return rows


def write_action_file(path: str, rows: Iterable[tuple]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def official_q3(evaluator_dir: str, answers_path: str, predictions_path: str) -> Dict[str, object]:
    """Gọi đúng chuỗi hàm trong evaluator.main() của mã gốc (không sửa mã gốc), trả về P/R/F1
    từng câu hỏi + điểm từng đoạn (để bootstrap). Lỗi định dạng của evaluator (sys.exit(2))
    được đổi thành PredictionError kèm thông báo gốc."""
    import contextlib
    import importlib
    import sys
    if evaluator_dir not in sys.path:
        sys.path.insert(0, evaluator_dir)
    ActionFile = importlib.import_module("process").ActionFile
    QuestionScores = importlib.import_module("scoring").QuestionScores
    Evaluation = importlib.import_module("evaluation").Evaluation
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            predictions = ActionFile.from_file(predictions_path)
            answers = ActionFile.from_file(answers_path)
            diff = answers.diff_participants(predictions)
            if diff:
                raise PredictionError("Thực thể không khớp evaluator: " + "; ".join(diff[:5]))
            ps, asum = predictions.summarize(), answers.summarize()
            scores = {}
            for pid, a in asum.items():
                if pid not in ps:
                    raise PredictionError(f"Evaluator: thiếu dự đoán cho đoạn {pid}")
                scores[pid] = QuestionScores.from_summaries(a, ps[pid])
            ev = Evaluation(scores)
    except SystemExit as e:
        raise PredictionError(f"Evaluator gốc từ chối file (exit {e.code}): {buf.getvalue().strip()}")
    out = {}
    for q in ("inputs", "outputs", "conversions", "moves", "overall"):
        m = getattr(ev, q)
        out[q] = {"P": m.precision, "R": m.recall, "F1": m.F1()}
    out["overall_rounded"] = {"precision": round(ev.overall.precision, 3), "recall": round(ev.overall.recall, 3),
                              "f1": round(ev.overall.F1(), 3)}
    out["_per_paragraph"] = {str(pid): {q: (getattr(s, q).precision, getattr(s, q).recall)
                                        for q in ("inputs", "outputs", "conversions", "moves")}
                             for pid, s in scores.items()}
    return out


# ----------------------------------------------------------------------------- bootstrap theo đoạn
def _f1_from(tp, npred, ngold):
    return prf(tp, npred, ngold)["F1"]


def paired_bootstrap_pair_f1(res_a: dict, res_b: dict, slice_name: str = "all", n_boot: int = 10000,
                             seed: int = 2026) -> Dict[str, float]:
    """CI 95% của ΔF1 = F1(A) − F1(B), lấy mẫu lặp lại THEO ĐOẠN, cùng mẫu cho A và B."""
    pa, pb = res_a["_per_paragraph"], res_b["_per_paragraph"]
    pids = sorted(set(pa) & set(pb))
    if set(pa) != set(pb):
        raise ValueError("Hai hệ thống phải được chấm trên cùng tập đoạn")
    A = [pa[p]["pairs"][slice_name] for p in pids]
    B = [pb[p]["pairs"][slice_name] for p in pids]
    rng = random.Random(seed)
    N = len(pids)

    def f1(arr, idx):
        return _f1_from(sum(arr[i][0] for i in idx), sum(arr[i][1] for i in idx), sum(arr[i][2] for i in idx))

    full = list(range(N))
    delta = f1(A, full) - f1(B, full)
    ds = []
    for _ in range(n_boot):
        idx = [rng.randrange(N) for _ in range(N)]
        ds.append(f1(A, idx) - f1(B, idx))
    ds.sort()
    lo, hi = ds[int(0.025 * n_boot)], ds[int(0.975 * n_boot) - 1]
    p_le0 = sum(d <= 0 for d in ds) / n_boot
    return {"delta": delta, "ci_low": lo, "ci_high": hi, "p_boot_delta_le_0": p_le0, "n_paragraphs": N,
            "n_boot": n_boot, "slice": slice_name}


def bootstrap_ci_pair_f1(res: dict, slice_name: str = "all", n_boot: int = 10000, seed: int = 2026):
    pp = res["_per_paragraph"]
    pids = sorted(pp)
    A = [pp[p]["pairs"][slice_name] for p in pids]
    rng = random.Random(seed)
    N = len(pids)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(N) for _ in range(N)]
        vals.append(_f1_from(sum(A[i][0] for i in idx), sum(A[i][1] for i in idx), sum(A[i][2] for i in idx)))
    vals.sort()
    return {"ci_low": vals[int(0.025 * n_boot)], "ci_high": vals[int(0.975 * n_boot) - 1]}


def mcnemar_exact(a_correct: List[bool], b_correct: List[bool]) -> Dict[str, float]:
    """McNemar chính xác (hai phía) cho hai bộ chọn trên CÙNG các điểm quyết định."""
    b = sum(x and not y for x, y in zip(a_correct, b_correct))
    c = sum(y and not x for x, y in zip(a_correct, b_correct))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p": min(1.0, 2 * p)}


# ----------------------------------------------------------------------------- đọc action file của hệ thống khác
def action_rows_to_prediction(rows: List[tuple], rec: dict) -> dict:
    """Đổi action file (ví dụ dự đoán ProStruct/ProGlobal công bố) sang định dạng dự đoán chuẩn.
    Ghép theo (câu, tên thực thể); thực thể trùng tên được gán theo thứ tự xuất hiện."""
    n, T = len(rec["entities"]), len(rec["sentences"])
    by_step = defaultdict(list)
    for r in rows:
        if str(r[0]) == rec["paragraph_id"]:
            by_step[int(r[1])].append(r)
    labels = [[pc.NONE] * T for _ in range(n)]
    loc_after = [["?"] * T for _ in range(n)]
    loc_init = ["?"] * n
    for t in range(1, T + 1):
        used = set()
        for r in by_step.get(t, []):
            cand = [i for i, e in enumerate(rec["entities"]) if e == r[2] and i not in used]
            if not cand:
                raise PredictionError(f"{rec['paragraph_id']}: thực thể lạ {r[2]!r} ở câu {t}")
            i = cand[0]
            used.add(i)
            labels[i][t - 1] = r[3]
            before, after = (r[4] or "?"), (r[5] or "?")
            loc_after[i][t - 1] = after
            if t == 1:
                loc_init[i] = before
    exists_initial = [loc_init[i] != pc.NO_LOC if any(by_step.values()) else True for i in range(n)]
    return {"paragraph_id": rec["paragraph_id"], "labels": labels, "exists_initial": exists_initial,
            "location_initial": loc_init, "locations_after": loc_after}
