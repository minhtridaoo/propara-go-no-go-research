"""propara_core — chuẩn hoá ProPara, suy nhãn, cờ nhắc, cặp vàng, điểm quyết định.

Phiên bản: core_v1 (notebook 00_propara_data_audit). Chỉ dùng thư viện chuẩn của Python.

Quy ước chỉ số:
  * states[i] có độ dài T+1: states[i][0] là trạng thái ban đầu, states[i][t] là trạng thái SAU câu t.
  * labels[i], mentions[i], implicit[i] có độ dài T; phần tử [t-1] ứng với câu t (đánh số từ 1).
  * sentence_id luôn đánh số từ 1 như bộ chấm chính thức của ProPara.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

CORE_VERSION = "core_v1"

NONE, CREATE, DESTROY, MOVE = "NONE", "CREATE", "DESTROY", "MOVE"
LABELS = (CREATE, DESTROY, MOVE, NONE)
NO_LOC = "-"   # không tồn tại
UNK_LOC = "?"  # tồn tại nhưng không rõ vị trí

# Từ rỗng bỏ qua khi xét cờ nhắc. Danh sách này được KHOÁ cùng core_v1;
# với danh sách này + MENTION_MODE="report" ta tái hiện đúng Bảng 1 của báo cáo.
STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or its it this that these those "
    "is are be into onto some any other another their his her your our".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ----------------------------------------------------------------------------- tiện ích
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def split_aliases(participant: str) -> List[str]:
    """'water vapor ; vapor' -> ['water vapor', 'vapor'] (giống evaluator: tách theo ';')."""
    return [p.strip() for p in participant.split(";") if p.strip()]


# ----------------------------------------------------------------------------- nhãn
def derive_label(before: str, after: str) -> str:
    """Công thức (1) của báo cáo."""
    if before == after:
        return NONE
    if before == NO_LOC and after != NO_LOC:
        return CREATE
    if before != NO_LOC and after == NO_LOC:
        return DESTROY
    return MOVE


def exists(state: str) -> bool:
    """'?' = tồn tại nhưng không rõ vị trí; chỉ '-' là không tồn tại."""
    return state != NO_LOC


VALID_LABELS = {False: {CREATE, NONE}, True: {DESTROY, MOVE, NONE}}  # công thức (15)


def count_constraint_violations(labels_by_entity: List[List[str]],
                                init_exists: List[bool]) -> Tuple[int, int]:
    """Số ô vi phạm ràng buộc tồn tại khi trạng thái TỰ TIẾN HOÁ theo nhãn dự đoán (công thức 28).

    Trả về (số ô vi phạm, số ô). Khi vi phạm, trạng thái vẫn cập nhật theo nhãn
    (CREATE -> tồn tại, DESTROY -> không tồn tại) để không che lỗi tiếp theo.
    """
    bad = total = 0
    for labs, e in zip(labels_by_entity, init_exists):
        for y in labs:
            total += 1
            if y not in VALID_LABELS[e]:
                bad += 1
            if y == CREATE:
                e = True
            elif y == DESTROY:
                e = False
    return bad, total


# ----------------------------------------------------------------------------- cờ nhắc
def _variants(word: str, mode: str) -> set:
    v = {word, word + "s"}
    if word.endswith("s"):
        v.add(word[:-1])
    if mode == "extended":  # nhạy hơn với số nhiều -es/-ies (dùng cho phân tích độ nhạy)
        v.add(word + "es")
        if word.endswith("es"):
            v.add(word[:-2])
        if word.endswith("ies"):
            v.add(word[:-3] + "y")
        if word.endswith("y"):
            v.add(word[:-1] + "ies")
    return v


def is_mentioned(participant: str, sentence: str, mode: str = "report") -> bool:
    """m_it = 1 nếu có ít nhất một từ nội dung của tên thực thể (mọi tên gọi khác,
    dạng số ít/số nhiều) xuất hiện trong câu.

    mode="report": khớp w, w+s, w bỏ 's' — tái hiện đúng Bảng 1 của báo cáo.
    mode="extended": thêm -es/-ies (echo/echoes, virus/viruses) — chỉ để đo độ nhạy.
    """
    if mode not in ("report", "extended"):
        raise ValueError(f"mode không hợp lệ: {mode}")
    sent_tokens = set(tokenize(sentence))
    for alias in split_aliases(participant):
        for w in tokenize(alias):
            if w in STOPWORDS:
                continue
            if _variants(w, mode) & sent_tokens:
                return True
    return False


# ----------------------------------------------------------------------------- bản ghi chuẩn
def build_record(raw: dict, split: str, topic: Optional[str] = None,
                 mention_mode: str = "report") -> dict:
    """Chuyển một dòng grids.v1.*.json thành bản ghi chuẩn (không sửa nhãn gốc)."""
    sents = list(raw["sentence_texts"])
    parts = list(raw["participants"])
    states = [list(s) for s in raw["states"]]
    T = len(sents)
    labels, mentions, implicit, existence = [], [], [], []
    for i, p in enumerate(parts):
        st = states[i]
        labs = [derive_label(st[t - 1], st[t]) for t in range(1, len(st))]
        men = [is_mentioned(p, sents[t - 1], mention_mode) for t in range(1, T + 1)] if len(st) == T + 1 else []
        labels.append(labs)
        mentions.append(men)
        implicit.append([(y != NONE) and (not m) for y, m in zip(labs, men)])
        existence.append([exists(s) for s in st])
    return {
        "paragraph_id": str(raw["para_id"]),
        "split": split,
        "topic": topic,
        "sentences": sents,
        "entities": parts,
        "entity_aliases": [split_aliases(p) for p in parts],
        "states": states,
        "existence": existence,
        "labels": labels,
        "mentions": mentions,
        "implicit": implicit,
        "mention_mode": mention_mode,
        "core_version": CORE_VERSION,
    }


def validate_record(rec: dict) -> List[str]:
    """Kiểm tra cấu trúc; trả về danh sách lỗi (rỗng = hợp lệ)."""
    issues = []
    pid = rec["paragraph_id"]
    T, n = len(rec["sentences"]), len(rec["entities"])
    if T == 0:
        issues.append(f"{pid}: không có câu")
    if n == 0:
        issues.append(f"{pid}: không có thực thể")
    if len(rec["states"]) != n:
        issues.append(f"{pid}: số hàng states ({len(rec['states'])}) != số thực thể ({n})")
    for t, s in enumerate(rec["sentences"], 1):
        if not s or not s.strip():
            issues.append(f"{pid}: câu {t} rỗng")
    lowered = [p.strip().lower() for p in rec["entities"]]
    for p, c in Counter(lowered).items():
        if c > 1:
            issues.append(f"{pid}: thực thể trùng tên '{p}'")
    for i, st in enumerate(rec["states"]):
        if len(st) != T + 1:
            issues.append(f"{pid}: thực thể {i} có {len(st)} trạng thái, cần {T + 1}")
        for t, v in enumerate(st):
            if not isinstance(v, str) or v.strip() == "":
                issues.append(f"{pid}: trạng thái rỗng tại thực thể {i}, bước {t}")
            elif v != v.strip():
                issues.append(f"{pid}: trạng thái có khoảng trắng thừa '{v}' (thực thể {i}, bước {t})")
    return issues


# ----------------------------------------------------------------------------- cặp vàng
def destroyed_created_at(rec: dict, t: int) -> Tuple[List[int], List[int]]:
    D = [i for i in range(len(rec["entities"])) if rec["labels"][i][t - 1] == DESTROY]
    C = [i for i in range(len(rec["entities"])) if rec["labels"][i][t - 1] == CREATE]
    return D, C


def gold_pairs_report(rec: dict) -> List[dict]:
    """Công thức (3): GoldPairs(t) = D_t x C_t. Đây là nhãn SUY TỪ TRẠNG THÁI,
    không phải quan hệ nhân quả đã được người xác nhận (pair_gold_type)."""
    out = []
    for t in range(1, len(rec["sentences"]) + 1):
        D, C = destroyed_created_at(rec, t)
        for x in D:
            for y in C:
                out.append({
                    "paragraph_id": rec["paragraph_id"],
                    "split": rec["split"],
                    "sentence_id": t,
                    "source": rec["entities"][x],
                    "target": rec["entities"][y],
                    "source_idx": x,
                    "target_idx": y,
                    "source_location_before": rec["states"][x][t - 1],
                    "target_location_after": rec["states"][y][t],
                    "source_implicit": bool(rec["implicit"][x][t - 1]),
                    "target_implicit": bool(rec["implicit"][y][t - 1]),
                    "n_destroy_in_sentence": len(D),
                    "n_create_in_sentence": len(C),
                    "pair_gold_type": "derived_from_states",
                })
    return out


def official_q3_conversions(rec: dict) -> List[dict]:
    """Bản sao logic Process.conversions() của evaluator chính thức
    (allenai/aristo-leaderboard/propara, commit 60d5be3), GIỮ NGUYÊN cả điều kiện
    `step_id < num_steps - 1`. Mỗi câu cho tối đa MỘT conversion, gộp mọi thực thể
    bị huỷ/tạo bằng ' AND '; có luật ghép với câu kế tiếp (rule b).
    Notebook 01 sẽ đối chiếu hàm này với mã evaluator gốc."""
    T = len(rec["sentences"])
    names = [" OR ".join(split_aliases(p)) for p in rec["entities"]]

    def loc(v):  # evaluator đổi '?' -> 'unk', '-' -> 'null'
        return {"?": "unk", "-": "null"}.get(v, v)

    def created(t):
        idx = [i for i in range(len(names)) if rec["states"][i][t - 1] == NO_LOC and rec["states"][i][t] != NO_LOC]
        return [names[i] for i in idx], [loc(rec["states"][i][t]) for i in idx]

    def destroyed(t):
        idx = [i for i in range(len(names)) if rec["states"][i][t - 1] != NO_LOC and rec["states"][i][t] == NO_LOC]
        return [names[i] for i in idx], [loc(rec["states"][i][t - 1]) for i in idx]

    conv = []
    for t in range(1, T + 1):
        c, cl = created(t)
        d, dl = destroyed(t)
        if c and d:
            conv.append({"step_id": t, "destroyed": " AND ".join(d), "created": " AND ".join(c),
                         "locations": sorted(set(cl + dl)), "rule": "same_step"})
        elif d and t < T - 1:
            c2, cl2 = created(t + 1)
            d2, _ = destroyed(t + 1)
            cnd = [x for x in c2 if x not in set(d)]
            if not d2 and cnd:
                conv.append({"step_id": t, "destroyed": " AND ".join(d), "created": " AND ".join(sorted(set(cnd))),
                             "locations": sorted(set(cl2 + dl)), "rule": "next_step_create"})
        elif c and t < T - 1:
            c2, _ = created(t + 1)
            d2, dl2 = destroyed(t + 1)
            dnc = [x for x in d2 if x not in set(c)]
            if not c2 and dnc:
                conv.append({"step_id": t, "destroyed": " AND ".join(sorted(set(dnc))), "created": " AND ".join(c),
                             "locations": sorted(set(cl + dl2)), "rule": "next_step_destroy"})
    return conv


def to_action_rows(rec: dict) -> List[Tuple[str, int, str, str, str, str]]:
    """Dạng 'action file' của evaluator: (process_id, step, participant, action, before, after)."""
    rows = []
    for t in range(1, len(rec["sentences"]) + 1):
        for i, p in enumerate(rec["entities"]):
            rows.append((rec["paragraph_id"], t, p, rec["labels"][i][t - 1],
                         rec["states"][i][t - 1], rec["states"][i][t]))
    return rows


# ----------------------------------------------------------------------------- điểm quyết định ngầm
def decision_points(rec: dict) -> List[dict]:
    """Mục 2.2: bộ (đoạn, câu, nhãn) có >=1 ô vàng ngầm và >=2 ứng viên.
    Ứng viên DESTROY = thực thể đang tồn tại trước câu; CREATE = chưa tồn tại;
    thực thể vàng ĐƯỢC NHẮC bị loại khỏi tập ứng viên."""
    out = []
    n = len(rec["entities"])
    for t in range(1, len(rec["sentences"]) + 1):
        for lab in (DESTROY, CREATE):
            need_exist = lab == DESTROY
            gold = [i for i in range(n) if rec["labels"][i][t - 1] == lab]
            gold_implicit = [i for i in gold if rec["implicit"][i][t - 1]]
            gold_mentioned = [i for i in gold if not rec["implicit"][i][t - 1]]
            if not gold_implicit:
                continue
            cands = [i for i in range(n)
                     if rec["existence"][i][t - 1] == need_exist and i not in gold_mentioned]
            if len(cands) < 2:
                continue
            out.append({
                "paragraph_id": rec["paragraph_id"], "split": rec["split"], "topic": rec.get("topic"),
                "sentence_id": t, "label": lab,
                "candidates": cands, "gold": gold_implicit,
                "n_candidates": len(cands), "n_gold": len(gold_implicit),
                "random_acc": len(gold_implicit) / len(cands),
            })
    return out


# ----------------------------------------------------------------------------- PP2 / PP1 (đếm)
def pp2_counts(rec: dict) -> Dict[str, int]:
    """Công thức (7) và (12): mọi cặp có thứ tự (X != Y) mỗi câu, và số cặp còn lại
    sau lọc bằng TỒN TẠI VÀNG (X tồn tại trước câu, Y chưa tồn tại). Đây là MỨC TRẦN."""
    T, n = len(rec["sentences"]), len(rec["entities"])
    total = T * n * (n - 1)
    kept = 0
    gold = {(g["source_idx"], g["target_idx"], g["sentence_id"]) for g in gold_pairs_report(rec)}
    gold_kept = 0
    for t in range(1, T + 1):
        ex = [rec["existence"][i][t - 1] for i in range(n)]
        for x in range(n):
            for y in range(n):
                if x != y and ex[x] and not ex[y]:
                    kept += 1
                    if (x, y, t) in gold:
                        gold_kept += 1
    return {"all_pairs": total, "kept_gold_existence": kept,
            "gold_pairs": len(gold), "gold_kept": gold_kept, "label_questions": 4 * T * n}


def cooccurrence_filter_coverage(rec: dict) -> Dict[str, int]:
    """Thử tái hiện câu 'lọc theo đồng xuất hiện trong câu chỉ giữ 74,7% cặp vàng và 18,9%
    cặp có nguồn ngầm' (mục 4.2). Báo cáo KHÔNG định nghĩa rõ, nên đo hai cách hiểu:
      same_sentence: X và Y cùng được nhắc trong chính câu t;
      any_sentence:  X và Y cùng được nhắc trong ít nhất một câu nào đó của đoạn."""
    out = Counter()
    T = len(rec["sentences"])
    for g in gold_pairs_report(rec):
        x, y, t = g["source_idx"], g["target_idx"], g["sentence_id"]
        same = rec["mentions"][x][t - 1] and rec["mentions"][y][t - 1]
        anys = any(rec["mentions"][x][k] and rec["mentions"][y][k] for k in range(T))
        out["gold"] += 1
        out["same_sentence"] += int(same)
        out["any_sentence"] += int(anys)
        if g["source_implicit"]:
            out["gold_src_implicit"] += 1
            out["same_sentence_src_implicit"] += int(same)
            out["any_sentence_src_implicit"] += int(anys)
    return dict(out)


# ----------------------------------------------------------------------------- lift (Bảng 2)
def lift_table(records: Iterable[dict]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Công thức (2). Ô hợp lệ: CREATE -> thực thể chưa tồn tại trước câu;
    DESTROY, MOVE -> đang tồn tại. Chỉ xét ô KHÔNG được nhắc (m_it = 0).
    B = 'trong câu t có thực thể j != i được nhắc và nhận nhãn B'."""
    cnt = defaultdict(Counter)
    for rec in records:
        n = len(rec["entities"])
        for t in range(1, len(rec["sentences"]) + 1):
            for i in range(n):
                if rec["mentions"][i][t - 1]:
                    continue
                y = rec["labels"][i][t - 1]
                e = rec["existence"][i][t - 1]
                for A in (CREATE, DESTROY, MOVE):
                    if (A == CREATE) == e:  # ô không hợp lệ cho nhãn A
                        continue
                    for B in (CREATE, DESTROY, MOVE):
                        sig = any(j != i and rec["mentions"][j][t - 1] and rec["labels"][j][t - 1] == B
                                  for j in range(n))
                        cnt[(A, B)][("sig" if sig else "nosig", y == A)] += 1
    res = {}
    for (A, B), c in cnt.items():
        a1, n1 = c[("sig", True)], c[("sig", True)] + c[("sig", False)]
        a0, n0 = c[("nosig", True)], c[("nosig", True)] + c[("nosig", False)]
        p1 = a1 / n1 if n1 else float("nan")
        p0 = a0 / n0 if n0 else float("nan")
        res[(A, B)] = {"lift": (p1 / p0) if p0 and p0 == p0 else float("nan"),
                       "n_implicit_A_with_B": a1, "p_sig": p1, "p_nosig": p0}
    return res


# ----------------------------------------------------------------------------- nạp dữ liệu
def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_topics_from_naacl_grid(path: str) -> Dict[str, str]:
    """gold-full-grids.v3.tsv: dòng có cột 3 bắt đầu 'PROMPT:' cho biết chủ đề quy trình."""
    topics = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > 2 and c[2].startswith("PROMPT:"):
                topics[c[0].strip()] = c[2][len("PROMPT:"):].strip()
    return topics


def load_action_file(path: str) -> List[Tuple[str, int, str, str, str, str]]:
    """answers.tsv của leaderboard: (pid, step, participant, action, before, after).
    Trả về DANH SÁCH (không dùng dict) để không âm thầm gộp các thực thể trùng tên."""
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            c = line.rstrip("\n\r").split("\t")
            if len(c) < 6:
                raise ValueError(f"{path}:{ln}: cần 6 cột, có {len(c)}")
            out.append((c[0], int(c[1]), c[2], c[3], c[4], c[5]))
    return out


def compare_action_rows(official: List[tuple], derived: List[tuple]) -> Dict[str, object]:
    """So sánh dạng đa tập theo (pid, step): bỏ qua thứ tự dòng, không gộp trùng tên."""
    a, b = defaultdict(list), defaultdict(list)
    for r in official:
        a[(str(r[0]), int(r[1]))].append(tuple(r[2:6]))
    for r in derived:
        b[(str(r[0]), int(r[1]))].append(tuple(r[2:6]))
    keys = set(a) | set(b)
    bad = sorted(k for k in keys if sorted(a[k]) != sorted(b[k]))
    return {"n_official_rows": len(official), "n_derived_rows": len(derived),
            "n_mismatched_steps": len(bad), "examples": bad[:5]}


# ----------------------------------------------------------------------------- chia chéo & mẫu kiểm tra
def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def cv_folds_by_topic(records: List[dict], k: int = 5, seed: int = 2026) -> Dict[str, int]:
    """Chia k phần theo CHỦ ĐỀ (prompt), tất định. Cân bằng tham lam đồng thời
    số điểm quyết định ngầm, số cặp vàng và số đoạn (mỗi đại lượng chuẩn hoá theo tổng).
    Trả về {paragraph_id: fold}."""
    by_topic = defaultdict(list)
    for r in records:
        by_topic[r.get("topic") or f"__no_topic__{r['paragraph_id']}"].append(r)
    stats = []
    for tp, rs in by_topic.items():
        v = (sum(len(decision_points(r)) for r in rs), sum(len(gold_pairs_report(r)) for r in rs), len(rs))
        stats.append((v, stable_hash(f"{seed}:{tp}"), tp))
    tot = [max(sum(s[0][d] for s in stats), 1) for d in range(3)]
    stats.sort(key=lambda x: (-sum(x[0][d] / tot[d] for d in range(3)), x[1]))
    load = [[0, 0, 0] for _ in range(k)]
    topic_fold = {}
    for v, _, tp in stats:
        f = min(range(k), key=lambda j: (max((load[j][d] + v[d]) / tot[d] for d in range(3)),
                                         sum(load[j][d] / tot[d] for d in range(3)), j))
        topic_fold[tp] = f
        for d in range(3):
            load[f][d] += v[d]
    return {r["paragraph_id"]: topic_fold[r.get("topic") or f"__no_topic__{r['paragraph_id']}"] for r in records}


def gold_audit_sample(pairs: List[dict], quotas: Dict[str, int], seed: int = 2026) -> List[dict]:
    """Lấy mẫu có chủ đích, tất định, cho người kiểm tra quan hệ X->Y.
    Nhóm (ưu tiên theo thứ tự): source_implicit, target_implicit, multi_pair, regular."""
    def group(p):
        if p["source_implicit"]:
            return "source_implicit"
        if p["target_implicit"]:
            return "target_implicit"
        if p["n_destroy_in_sentence"] * p["n_create_in_sentence"] > 1:
            return "multi_pair"
        return "regular"
    out = []
    for g, q in quotas.items():
        pool = [p for p in pairs if group(p) == g]
        pool.sort(key=lambda p: stable_hash(f"{seed}:{p['paragraph_id']}:{p['sentence_id']}:{p['source_idx']}:{p['target_idx']}"))
        for p in pool[:q]:
            out.append(dict(p, audit_group=g))
    return out
