"""propara_heuristics — các heuristic mốc (mục 2.3 báo cáo) + hệ thống đầu-cuối đơn giản.

Phiên bản: heur_v1. Phụ thuộc propara_core, propara_metrics.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import propara_core as pc

HEUR_VERSION = "heur_v1"


# ----------------------------------------------------------------------------- bộ chọn trên điểm quyết định
def _unmentioned(rec: dict, t: int, cands: List[int]) -> List[int]:
    return [i for i in cands if not rec["mentions"][i][t - 1]]


def last_mention_before(rec: dict, i: int, t: int) -> Optional[int]:
    """Câu gần nhất TRƯỚC câu t có nhắc thực thể i (None nếu chưa từng nhắc)."""
    for k in range(t - 1, 0, -1):
        if rec["mentions"][i][k - 1]:
            return k
    return None


def expected_acc(dp: dict, rec: dict, method: str) -> float:
    """Độ chính xác KỲ VỌNG (tính chính xác, không mô phỏng) của bộ chọn tại một điểm quyết định."""
    U, G = dp["candidates"], set(dp["gold"])
    t = dp["sentence_id"]
    if method == "random":
        pool = U
    elif method == "random_unmentioned":
        pool = _unmentioned(rec, t, U) or U
    elif method == "nearest_mention":
        last = {i: last_mention_before(rec, i, t) for i in U}
        seen = [i for i in U if last[i] is not None]
        if not seen:
            pool = U
        else:
            best = max(last[i] for i in seen)
            pool = [i for i in seen if last[i] == best]       # hoà -> chọn ngẫu nhiên trong nhóm hoà
    else:
        raise ValueError(method)
    return sum(i in G for i in pool) / len(pool)


def simulate_choice(dp: dict, rec: dict, method: str, rng: random.Random) -> int:
    U, t = dp["candidates"], dp["sentence_id"]
    if method == "random":
        return rng.choice(U)
    if method == "random_unmentioned":
        return rng.choice(_unmentioned(rec, t, U) or U)
    if method == "nearest_mention":
        last = {i: last_mention_before(rec, i, t) for i in U}
        seen = [i for i in U if last[i] is not None]
        if not seen:
            return rng.choice(U)
        best = max(last[i] for i in seen)
        return rng.choice([i for i in seen if last[i] == best])
    raise ValueError(method)


def decision_point_table(dps: List[dict], recs_by_pid: Dict[str, dict], methods=("random", "random_unmentioned", "nearest_mention"),
                         seeds: int = 20, seed0: int = 0) -> Dict[str, Dict[str, object]]:
    """Trả về {method: {label: {expected, sim_mean, n, per_dp_correct_seed0}}}."""
    out = {}
    for m in methods:
        out[m] = {}
        for lab in (pc.DESTROY, pc.CREATE):
            sub = [d for d in dps if d["label"] == lab]
            exp = [expected_acc(d, recs_by_pid[d["paragraph_id"]], m) for d in sub]
            sims = []
            first = None
            for s in range(seeds):
                rng = random.Random(seed0 + s)
                corr = [simulate_choice(d, recs_by_pid[d["paragraph_id"]], m, rng) in set(d["gold"]) for d in sub]
                sims.append(sum(corr) / len(corr) if corr else float("nan"))
                if first is None:
                    first = corr
            out[m][lab] = {"n": len(sub), "expected": sum(exp) / len(exp) if exp else float("nan"),
                           "sim_mean_20seeds": sum(sims) / len(sims) if sims else float("nan"),
                           "expected_per_dp": exp}
    return out


# ----------------------------------------------------------------------------- hệ thống đầu-cuối (để chạy toàn bộ luồng chấm)
def all_none(rec: dict) -> dict:
    n, T = len(rec["entities"]), len(rec["sentences"])
    return {"paragraph_id": rec["paragraph_id"], "labels": [[pc.NONE] * T for _ in range(n)],
            "exists_initial": [True] * n}


def oracle_mentioned_only(rec: dict) -> dict:
    """CHẨN ĐOÁN (dùng vàng): giữ nhãn vàng ở ô được nhắc, đặt NONE ở mọi ô ngầm.
    Cho biết hệ thống 'hoàn hảo trên ô được nhắc nhưng mù với thay đổi ngầm' đạt bao nhiêu → trần
    của các phương pháp không suy luận ngầm, và khoảng trống mà đề tài nhắm tới."""
    labels = [[(y if not rec["implicit"][i][t] else pc.NONE) for t, y in enumerate(row)]
              for i, row in enumerate(rec["labels"])]
    return {"paragraph_id": rec["paragraph_id"], "labels": labels,
            "exists_initial": [e[0] for e in rec["existence"]],
            "location_initial": [s[0] for s in rec["states"]],
            "locations_after": [s[1:] for s in rec["states"]]}


def oracle_mentioned_plus_nearest(rec: dict, dps_of_para: List[dict], rng: random.Random) -> dict:
    """CHẨN ĐOÁN (dùng vàng cho ô được nhắc + VỊ TRÍ điểm quyết định): tại mỗi điểm quyết định,
    gán nhãn bằng heuristic nhắc-gần-nhất cho đúng số thực thể vàng ngầm. Đo heuristic ở mức cặp."""
    pred = oracle_mentioned_only(rec)
    for d in dps_of_para:
        U = list(d["candidates"])
        k = len(d["gold"])
        chosen = []
        for _ in range(k):
            dd = dict(d, candidates=[i for i in U if i not in chosen])
            if not dd["candidates"]:
                break
            chosen.append(simulate_choice(dd, rec, "nearest_mention", rng))
        for i in chosen:
            pred["labels"][i][d["sentence_id"] - 1] = d["label"]
    return pred


# ----------------------------------------------------------------------------- B1-E2E: heuristic đầu-cuối KHÔNG dùng vàng
CREATE_CUES = set("""form forms formed forming produce produces produced producing create creates created creating make makes made
making become becomes became develop develops developed generate generates generated grow grows grew release releases released""".split())
DESTROY_CUES = set("""use uses used using consume consumes consumed consuming decompose decomposes decomposed decomposing burn burns
burned burnt burning die dies died dying melt melts melted melting evaporate evaporates evaporated evaporating dissolve dissolves
dissolved eat eats ate eaten break breaks broke broken""".split())
TRANSFORM = set("turn turns turned turning change changes changed convert converts converted become becomes became form forms formed".split())


def rule_nearest_e2e(rec: dict) -> dict:
    """Heuristic đầu-cuối (B1-E2E), luật cố định viết TRƯỚC khi xem điểm dev/test:
    (1) Thực thể ĐƯỢC NHẮC: trước 'into' trong câu có động từ biến đổi → DESTROY, sau 'into' → CREATE;
        nếu không: câu có từ gợi tạo → CREATE, câu có từ gợi huỷ → DESTROY.
    (2) Tồn tại ban đầu = không (sự kiện đầu tiên của thực thể là CREATE).
    (3) Ngầm (heuristic 'nhắc gần nhất' của báo cáo, B1c): câu có CREATE mà không DESTROY → huỷ thực thể đang tồn tại,
        không được nhắc, được nhắc gần nhất trước đó; ngược lại có DESTROY mà không CREATE → tạo thực thể chưa tồn tại,
        không được nhắc, được nhắc gần nhất trước đó. Không có ứng viên từng được nhắc → bỏ qua (không đoán bừa).
    (4) Áp ràng buộc tồn tại khi đi qua các câu."""
    n, T = len(rec["entities"]), len(rec["sentences"])
    words = [pc.tokenize(s) for s in rec["sentences"]]
    raw = [[pc.NONE] * T for _ in range(n)]
    for t in range(1, T + 1):
        ws = words[t - 1]
        has_into = "into" in ws and any(w in TRANSFORM for w in ws)
        into_pos = ws.index("into") if "into" in ws else -1
        cue_c = any(w in CREATE_CUES for w in ws)
        cue_d = any(w in DESTROY_CUES for w in ws)
        for i, ent in enumerate(rec["entities"]):
            if not rec["mentions"][i][t - 1]:
                continue
            pos = [k for k, w in enumerate(ws) if pc.is_mentioned(ent, w)]
            if has_into and pos:
                raw[i][t - 1] = pc.DESTROY if min(pos) < into_pos else pc.CREATE
            elif cue_c and not cue_d:
                raw[i][t - 1] = pc.CREATE
            elif cue_d and not cue_c:
                raw[i][t - 1] = pc.DESTROY
    init = [next((y for y in raw[i] if y != pc.NONE), pc.NONE) != pc.CREATE for i in range(n)]
    labels = [[pc.NONE] * T for _ in range(n)]
    e = list(init)
    for t in range(1, T + 1):
        for i in range(n):
            y = raw[i][t - 1]
            if y in pc.VALID_LABELS[e[i]] and y != pc.NONE:
                labels[i][t - 1] = y
        D = [i for i in range(n) if labels[i][t - 1] == pc.DESTROY]
        C = [i for i in range(n) if labels[i][t - 1] == pc.CREATE]
        def nearest(cands):
            last = {i: last_mention_before(rec, i, t) for i in cands}
            seen = [i for i in cands if last[i] is not None]
            return max(seen, key=lambda i: (last[i], -i)) if seen else None
        if C and not D:
            j = nearest([i for i in range(n) if e[i] and not rec["mentions"][i][t - 1] and labels[i][t - 1] == pc.NONE])
            if j is not None:
                labels[j][t - 1] = pc.DESTROY
        elif D and not C:
            j = nearest([i for i in range(n) if not e[i] and not rec["mentions"][i][t - 1] and labels[i][t - 1] == pc.NONE])
            if j is not None:
                labels[j][t - 1] = pc.CREATE
        for i in range(n):
            if labels[i][t - 1] == pc.CREATE:
                e[i] = True
            elif labels[i][t - 1] == pc.DESTROY:
                e[i] = False
    return {"paragraph_id": rec["paragraph_id"], "labels": labels, "exists_initial": init}


def rule_only_e2e(rec: dict) -> dict:
    """Như rule_nearest_e2e nhưng BỎ bước (3) suy ngầm → đo riêng đóng góp của heuristic ngầm."""
    p = rule_nearest_e2e(rec)
    n, T = len(rec["entities"]), len(rec["sentences"])
    labels = [[(y if rec["mentions"][i][t] else pc.NONE) for t, y in enumerate(p["labels"][i])] for i in range(n)]
    return {"paragraph_id": rec["paragraph_id"], "labels": labels, "exists_initial": p["exists_initial"]}
