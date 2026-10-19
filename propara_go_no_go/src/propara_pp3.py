"""propara_pp3 — PP3: khai thác cặp chuyển đổi X→Y từ WikiHow bằng mẫu cố định + bộ chấm có hướng E(X,Y) (mục 4.5).

Phiên bản: pp3_v1. Phần trích mẫu chỉ dùng thư viện chuẩn; phần huấn luyện cần torch.

Rút gọn so với báo cáo (ghi rõ để không nhầm): báo cáo rút cụm về "danh từ đầu ở dạng gốc"; ở đây
KHÔNG có bộ phân tích cú pháp, danh từ đầu = từ cuối của cụm sau khi cắt ở từ ranh giới, đưa về số ít
bằng luật đơn giản. Các mẫu được KHOÁ trước khi chạy (PATTERNS_V1).
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import propara_core as pc

PP3_VERSION = "pp3_v2"

# ----------------------------------------------------------------------------- mẫu (khoá trước)
_V_TURN_T = r"(?:turn|turns|turned|turning|convert|converts|converted|converting|transform|transforms|transformed|transforming)"
_V_INTR = (r"(?:turns|turned|turn|turning|changes|changed|change|changing|breaks down|broke down|break down|breaking down|"
           r"decomposes|decomposed|decompose|decomposing|melts|melted|melt|melting)")
_V_PASS = r"(?:is|are|was|were|be|been|being|gets|get|got|getting)\s+(?:turned|converted|transformed)"
_V_MAKE = r"(?:make|makes|made|making)"
_BE = r"(?:is|are|was|were|be|been|being|can be|will be)"

# Mỗi mẫu: (tên, regex với nhóm có tên L (bên trái), R (bên phải), hướng): hướng "LR" nghĩa là X=L, Y=R.
PATTERNS_V1 = [
    ("turn_X_into_Y", re.compile(rf"\b{_V_TURN_T}\s+(?P<L>[^.,;:!?]+?)\s+into\s+(?P<R>[^.,;:!?]+)"), "LR"),
    ("X_turns_into_Y", re.compile(rf"(?P<L>[^.,;:!?]+?)\s+{_V_INTR}\s+into\s+(?P<R>[^.,;:!?]+)"), "LR"),
    ("X_is_turned_into_Y", re.compile(rf"(?P<L>[^.,;:!?]+?)\s+{_V_PASS}\s+into\s+(?P<R>[^.,;:!?]+)"), "LR"),
    ("make_Y_from_X", re.compile(rf"\b{_V_MAKE}\s+(?P<L>[^.,;:!?]+?)\s+(?:from|out of)\s+(?P<R>[^.,;:!?]+)"), "RL"),
    ("Y_is_made_from_X", re.compile(rf"(?P<L>[^.,;:!?]+?)\s+{_BE}\s+made\s+(?:from|out of)\s+(?P<R>[^.,;:!?]+)"), "RL"),
    ("Y_is_formed_from_X", re.compile(rf"(?P<L>[^.,;:!?]+?)\s+{_BE}\s+formed\s+from\s+(?P<R>[^.,;:!?]+)"), "RL"),
    ("Y_forms_from_X", re.compile(rf"(?P<L>[^.,;:!?]+?)\s+(?:forms|form|formed|forming)\s+from\s+(?P<R>[^.,;:!?]+)"), "RL"),
]

# Từ ranh giới: cụm bên phải bị cắt TRƯỚC từ đầu tiên thuộc tập này; cụm bên trái chỉ giữ phần SAU từ cuối thuộc tập này.
BOUNDARY = set("""when if and or but by with in on at for to that which who because while so as using until
after before then than of about over under through during without within into onto from is are was were be been
can will would should could may might must do does did has have had not you your yourself it they them this these
those we i he she its their his her our my me us also just simply easily quickly slowly""".split())
BAD_HEADS = set("""it them this that these those one ones thing things something anything everything stuff way ways
you yourself he she they we i what which who whatever part parts piece pieces bit bits kind type lot lots amount
amounts number numbers form forms shape shapes use uses order sure time times step steps sense good bad""".split())


IRREGULAR = {"leaves": "leaf", "halves": "half", "loaves": "loaf", "shelves": "shelf", "calves": "calf", "wolves": "wolf",
             "knives": "knife", "lives": "life", "wives": "wife", "mice": "mouse", "geese": "goose", "teeth": "tooth",
             "feet": "foot", "children": "child", "people": "person", "men": "man", "women": "woman", "oxen": "ox"}
QUANTIFIERS = set("""layer layers pile piles bunch bunches cup cups handful handfuls piece pieces bit bits lot lots amount amounts
drop drops bag bags sheet sheets slice slices kind kinds type types sort sorts mixture""".split())


NO_SINGULAR = {"canvas", "gas", "atlas", "lens", "species", "series", "news", "physics", "mathematics", "chaos", "bias", "alias"}


def singularize(w: str) -> str:
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w in NO_SINGULAR:
        return w
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes", "zes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _words(s: str) -> List[str]:
    return re.findall(r"[a-z][a-z\-']*", s.lower())


def head_right(phrase: str, max_words: int = 5) -> Optional[str]:
    """Cụm bên phải: cắt ở từ ranh giới đầu tiên, lấy từ cuối còn lại."""
    out = []
    ws = _words(phrase)
    while len(ws) >= 3 and ws[0] in ("a", "an", "the") and ws[1] in QUANTIFIERS and ws[2] == "of":
        ws = ws[3:]
    while len(ws) >= 2 and ws[0] in QUANTIFIERS and ws[1] == "of":       # 'layers of sediment' -> sediment
        ws = ws[2:]
    for w in ws[:max_words + 3]:
        if w in BOUNDARY and out:
            break
        if w in ("a", "an", "the", "some", "any", "more", "small", "large", "new"):
            continue
        out.append(w)
        if len(out) >= max_words:
            break
    return _clean_head(out[-1]) if out else None


def head_left(phrase: str) -> Optional[str]:
    """Cụm bên trái: chỉ giữ phần sau từ ranh giới cuối cùng, lấy từ cuối (sát động từ)."""
    ws = _words(phrase)
    keep = []
    for w in reversed(ws):
        if w in BOUNDARY:
            break
        keep.append(w)
        if len(keep) >= 5:
            break
    keep = [w for w in reversed(keep) if w not in ("a", "an", "the", "some", "any")]
    return _clean_head(keep[-1]) if keep else None


def _clean_head(w: str) -> Optional[str]:
    w = w.strip("-'").replace("'s", "")
    w = singularize(w)
    if len(w) < 3 or w in BAD_HEADS or w in BOUNDARY:
        return None
    return w


def extract_pairs_from_sentence(sentence: str) -> List[Tuple[str, str, str]]:
    """→ [(X, Y, tên_mẫu)] với X→Y."""
    s = sentence.strip()
    out = []
    for name, rx, direction in PATTERNS_V1:
        for mt in rx.finditer(s.lower()):
            # cụm L là tân ngữ đứng SAU động từ (turn X into, make Y from) hoặc chủ ngữ đứng TRƯỚC động từ (các mẫu còn lại)
            L = head_right(mt.group("L")) if name in ("turn_X_into_Y", "make_Y_from_X") else head_left(mt.group("L"))
            R = head_right(mt.group("R"))
            if not L or not R or L == R:
                continue
            X, Y = (L, R) if direction == "LR" else (R, L)
            out.append((X, Y, name))
    return out


# ----------------------------------------------------------------------------- pp3_v2: trích theo TỪ LOẠI (POS)
# Cùng 7 mẫu của báo cáo, nhưng danh từ đầu phải được gán nhãn NN*/NNS (bộ gán nhãn perceptron của NLTK).
TURN_T = {"turn", "turns", "turned", "turning", "convert", "converts", "converted", "converting",
          "transform", "transforms", "transformed", "transforming"}
INTR_1 = {"turns", "turned", "turn", "turning", "changes", "changed", "change", "changing", "decomposes", "decomposed",
          "decompose", "decomposing", "melts", "melted", "melt", "melting"}
BREAK = {"breaks", "broke", "break", "breaking"}
BE_GET = {"is", "are", "was", "were", "be", "been", "being", "gets", "get", "got", "getting"}
PASS_V = {"turned", "converted", "transformed"}
MAKE = {"make", "makes", "made", "making"}
FORM_V = {"forms", "form", "formed", "forming"}
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
NP_TAGS = NOUN_TAGS | {"DT", "PDT", "PRP$", "JJ", "JJR", "JJS", "CD", "POS", "VBN", "VBG"}
_TOK = re.compile(r"[A-Za-z][A-Za-z\-']*|[.,;:!?()]")


def _np_after(toks, tags, p):
    """Cụm danh từ bắt đầu tại p; trả về (danh từ đầu, vị trí kết thúc) hoặc (None, p)."""
    j, head, hpos = p, None, None
    while j < len(toks) and tags[j] in NP_TAGS:
        if tags[j] in NOUN_TAGS:
            head, hpos = toks[j], j
        elif head is not None and tags[j] not in NOUN_TAGS:
            break
        j += 1
    if head is not None and head.lower() in QUANTIFIERS and hpos + 1 < len(toks) and toks[hpos + 1].lower() == "of":
        return _np_after(toks, tags, hpos + 2)
    return head, j


def _np_before(toks, tags, e):
    """Danh từ đầu của cụm kết thúc tại e (ngay trước động từ); phải là danh từ."""
    if e < 0 or tags[e] not in NOUN_TAGS:
        return None
    return toks[e]


def _norm_head(w):
    return _clean_head(w.lower()) if w else None


def extract_pairs_pos(sentence: str, tagger) -> List[Tuple[str, str, str]]:
    toks = _TOK.findall(sentence)
    if len(toks) < 4:
        return []
    tags = [t for _, t in tagger(toks)]
    low = [t.lower() for t in toks]
    out = []
    n = len(toks)
    for i, w in enumerate(low):
        try:
            if w in TURN_T and i + 1 < n and low[i + 1] != "into":
                obj, j = _np_after(toks, tags, i + 1)
                if obj and j < n and low[j] == "into":
                    tgt, _ = _np_after(toks, tags, j + 1)
                    out.append((_norm_head(obj), _norm_head(tgt), "turn_X_into_Y"))
            if (w in INTR_1 and i + 1 < n and low[i + 1] == "into") or (w in BREAK and i + 2 < n and low[i + 1] == "down" and low[i + 2] == "into"):
                k = i + 1 if w in INTR_1 else i + 2
                if i > 0 and low[i - 1] in BE_GET:
                    continue                                    # 'is turned into' xử lý ở mẫu bị động
                src = _np_before(toks, tags, i - 1)
                tgt, _ = _np_after(toks, tags, k + 1)
                out.append((_norm_head(src), _norm_head(tgt), "X_turns_into_Y"))
            if w in BE_GET and i + 2 < n and low[i + 1] in PASS_V and low[i + 2] == "into":
                src = _np_before(toks, tags, i - 1)
                tgt, _ = _np_after(toks, tags, i + 3)
                out.append((_norm_head(src), _norm_head(tgt), "X_is_turned_into_Y"))
            if w in MAKE and not (i > 0 and low[i - 1] in BE_GET):
                obj, j = _np_after(toks, tags, i + 1)
                if obj and j < n and (low[j] == "from" or (low[j] == "out" and j + 1 < n and low[j + 1] == "of")):
                    src, _ = _np_after(toks, tags, j + (1 if low[j] == "from" else 2))
                    out.append((_norm_head(src), _norm_head(obj), "make_Y_from_X"))
            if w in BE_GET and i + 2 < n and low[i + 1] == "made" and (low[i + 2] == "from" or (low[i + 2] == "out" and i + 3 < n and low[i + 3] == "of")):
                tgt = _np_before(toks, tags, i - 1)
                src, _ = _np_after(toks, tags, i + (3 if low[i + 2] == "from" else 4))
                out.append((_norm_head(src), _norm_head(tgt), "Y_is_made_from_X"))
            if w in BE_GET and i + 2 < n and low[i + 1] == "formed" and low[i + 2] == "from":
                tgt = _np_before(toks, tags, i - 1)
                src, _ = _np_after(toks, tags, i + 3)
                out.append((_norm_head(src), _norm_head(tgt), "Y_is_formed_from_X"))
            if w in FORM_V and i + 1 < n and low[i + 1] == "from" and not (i > 0 and low[i - 1] in BE_GET):
                tgt = _np_before(toks, tags, i - 1)
                src, _ = _np_after(toks, tags, i + 2)
                out.append((_norm_head(src), _norm_head(tgt), "Y_forms_from_X"))
        except IndexError:
            continue
    return [(x, y, nm) for x, y, nm in out if x and y and x != y]


_TRIGGER = re.compile(r"\b(?:into|from|out of)\b")
_VERB_HINT = re.compile(r"\b(?:turn|convert|transform|chang|break|broke|decompos|melt|made|make|form)")


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    return [s for s in _SENT_SPLIT.split(text or "") if len(s) > 5]


def article_key(title: str) -> str:
    return re.sub(r"\d+$", "", (title or "").strip()).lower()


def mine_corpus(rows: Iterable[Tuple[str, str]], log_every: int = 50000, log=print, tagger=None):
    """rows: (title, text). Trả về (Counter (X,Y)->số lần, {(X,Y)->set bài}, Counter mẫu, thống kê, {bài->set danh từ đầu})."""
    pair_cnt, pair_art, pat_cnt = Counter(), defaultdict(set), Counter()
    art_nouns = defaultdict(set)
    n_rows = n_sent = 0
    for title, text in rows:
        n_rows += 1
        art = article_key(title)
        for s in split_sentences(text):
            n_sent += 1
            low = s.lower()
            if not any(k in low for k in (" into ", " from ", " out of ")):
                continue
            if tagger is not None and not _VERB_HINT.search(low):
                continue
            found = extract_pairs_pos(s, tagger) if tagger is not None else extract_pairs_from_sentence(s)
            for X, Y, name in found:
                pair_cnt[(X, Y)] += 1
                pair_art[(X, Y)].add(art)
                pat_cnt[name] += 1
                art_nouns[art].update((X, Y))
        if log_every and n_rows % log_every == 0:
            log(f"  đã quét {n_rows} dòng, {n_sent} câu, {len(pair_cnt)} cặp khác nhau")
    return pair_cnt, pair_art, pat_cnt, {"rows": n_rows, "sentences": n_sent}, art_nouns


# ----------------------------------------------------------------------------- độ phủ trên ProPara
def entity_heads(entity: str) -> List[str]:
    """Danh từ đầu của mọi tên gọi khác của thực thể ProPara (cùng luật với phía WikiHow)."""
    hs = []
    for a in pc.split_aliases(entity):
        h = head_right(a) or head_left(a)
        if h and h not in hs:
            hs.append(h)
    return hs


def coverage(gold_pairs: List[dict], K: set) -> Dict[str, float]:
    """Công thức (24)–(26): phủ cặp / nguồn / đích + tỉ lệ cặp cùng danh từ đầu (không khai thác được bằng mẫu)."""
    sources = {x for x, _ in K}
    targets = {y for _, y in K}
    n = len(gold_pairs)
    c_pair = c_src = c_tgt = same = 0
    for g in gold_pairs:
        hx, hy = entity_heads(g["source"]), entity_heads(g["target"])
        if set(hx) & set(hy):
            same += 1
        c_pair += any((x, y) in K for x in hx for y in hy)
        c_src += any(x in sources for x in hx)
        c_tgt += any(y in targets for y in hy)
    return {"n_gold_pairs": n, "cover_pair": c_pair / max(n, 1), "cover_source": c_src / max(n, 1),
            "cover_target": c_tgt / max(n, 1), "same_head_pairs": same / max(n, 1)}


# ----------------------------------------------------------------------------- tập huấn luyện E(X,Y)
def build_training_pairs(pair_cnt: Counter, pair_art: dict, art_nouns: dict, val_frac: float = 0.1, min_count: int = 1,
                         seed: int = 2026, max_pos: int = 200000):
    """Dương = cặp khớp mẫu. Âm (ba loại như báo cáo): đích ngẫu nhiên, đảo chiều, cùng xuất hiện trong bài nhưng không khớp mẫu
    (âm GIẢ ĐỊNH, có thể nhiễu). Tách train/val theo BÀI: một cặp thuộc val nếu mọi bài chứa nó đều thuộc val."""
    rng = random.Random(seed)
    K = {p for p, c in pair_cnt.items() if c >= min_count}
    pos = sorted(K, key=lambda p: -pair_cnt[p])[:max_pos]
    is_val_art = lambda a: (pc.stable_hash(f"{seed}:{a}") % 1000) < val_frac * 1000
    targets = [y for _, y in pos]
    data = {"train": [], "val": []}
    for (x, y) in pos:
        split = "val" if all(is_val_art(a) for a in pair_art[(x, y)]) else "train"
        data[split].append((x, y, 1, "pos"))
        for _ in range(5):                                   # đích ngẫu nhiên
            y2 = rng.choice(targets)
            if (x, y2) not in K and y2 != x:
                data[split].append((x, y2, 0, "neg_random")); break
        if (y, x) not in K:                                  # đảo chiều
            data[split].append((y, x, 0, "neg_reverse"))
        arts = sorted(pair_art[(x, y)])
        nouns = sorted(art_nouns.get(rng.choice(arts), set()) - {x, y})
        if nouns:                                            # cùng bài, không khớp mẫu
            z = rng.choice(nouns)
            if (x, z) not in K:
                data[split].append((x, z, 0, "neg_cooccur"))
    return data, K


def train_pair_scorer(data: dict, embed: Callable[[List[str]], "torch.Tensor"], epochs: int = 30, lr: float = 1e-3,
                      wd: float = 1e-4, batch: int = 1024, seed: int = 2026, device: str = "cpu", log=print, patience: int = 5):
    """E(X,Y) = σ(u_X^T W v_Y + a^T u_X + b^T v_Y + c), u = v = embedding cố định (công thức 20–22), mất mát BCE (23).
    Chọn epoch theo loss VALIDATION. Trả về (model, lịch sử)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    vocab = sorted({w for sp in data.values() for x, y, _, _ in sp for w in (x, y)})
    idx = {w: i for i, w in enumerate(vocab)}
    emb = embed(vocab).float().to(device)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d = emb.shape[1]

    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.zeros(d, d))
            self.a = nn.Linear(d, 1)
            self.b = nn.Linear(d, 1, bias=False)
        def forward(self, u, v):
            return (u @ self.W * v).sum(-1) + self.a(u).squeeze(-1) + self.b(v).squeeze(-1)

    model = Scorer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    def tens(split):
        rows = data[split]
        return (torch.tensor([idx[x] for x, _, _, _ in rows], device=device), torch.tensor([idx[y] for _, y, _, _ in rows], device=device),
                torch.tensor([z for _, _, z, _ in rows], dtype=torch.float, device=device), [k for *_, k in rows])
    tr, va = tens("train"), tens("val")
    lossf = nn.BCEWithLogitsLoss()
    hist, best, best_state, bad = [], float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr[0]), device=device)
        tot = 0.0
        for s in range(0, len(perm), batch):
            j = perm[s:s + batch]
            loss = lossf(model(emb[tr[0][j]], emb[tr[1][j]]), tr[2][j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        model.eval()
        with torch.no_grad():
            lv = model(emb[va[0]], emb[va[1]])
            vloss = float(lossf(lv, va[2])) if len(lv) else tot / max(len(perm), 1)   # val rỗng → dùng train loss (ghi chú)
            m = _rank_metrics(lv.cpu(), va[2].cpu(), va[3])
        row = {"epoch": ep, "train_loss": tot / max(len(perm), 1), "val_loss": vloss, **m}
        hist.append(row)
        log(f"  PP3 ep{ep} train={row['train_loss']:.4f} val={vloss:.4f} AUC={m['auc']:.3f} AP={m['ap']:.3f} "
            f"AUC_vs_reverse={m['auc_vs_reverse']:.3f}")
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def _auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, r = 0.0, 1
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _rank_metrics(logits, labels, kinds):
    v = logits.tolist(); z = labels.tolist()
    pos = [a for a, b in zip(v, z) if b == 1]
    neg = [a for a, b in zip(v, z) if b == 0]
    rev = [a for a, k in zip(v, kinds) if k == "neg_reverse"]
    order = sorted(range(len(v)), key=lambda i: -v[i])
    hits, ap = 0, 0.0
    for r, i in enumerate(order, 1):
        if z[i] == 1:
            hits += 1
            ap += hits / r
    return {"auc": _auc(pos, neg), "ap": ap / max(hits, 1), "auc_vs_reverse": _auc(pos, rev), "n_val": len(v)}


def logE_matrix(model, embed, rec: dict, device: str = "cpu", floor: float = -10.0) -> List[List[float]]:
    """log E(X_i, X_j) cho mọi cặp thực thể của một đoạn (lấy max trên các tên gọi khác)."""
    import torch
    import torch.nn.functional as F
    n = len(rec["entities"])
    names = [entity_heads(e) or [e.lower()] for e in rec["entities"]]
    flat = sorted({h for hs in names for h in hs})
    E = embed(flat).float().to(device)
    E = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pos = {h: k for k, h in enumerate(flat)}
    out = [[0.0] * n for _ in range(n)]
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                best = floor
                for hx in names[i]:
                    for hy in names[j]:
                        s = model(E[pos[hx]][None], E[pos[hy]][None])
                        best = max(best, float(F.logsigmoid(s)))
                out[i][j] = max(best, floor)
    return out
