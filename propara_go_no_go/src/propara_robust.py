"""propara_robust — biến thể kiểm tra khái quát (Tập D trong PIPELINE §10.2).

Phiên bản: robust_v1.
NONCE: thay mọi TỪ NỘI DUNG trong tên thực thể bằng từ vô nghĩa, nhất quán trong toàn đoạn (câu, tên thực thể,
vị trí). Nhãn không đổi (đã kiểm tra lại sau khi thay). Đây là phép thử MẤT TRI THỨC NGỮ NGHĨA: điểm giảm KHÔNG
một mình chứng minh mô hình 'chỉ ghi nhớ' (xem PIPELINE §10.2, sai lầm #10).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import propara_core as pc

ROBUST_VERSION = "robust_v1"
_SYL_A = ["bl", "dr", "fr", "gl", "kr", "pl", "sn", "tr", "v", "z", "m", "n", "t", "d", "k", "p"]
_SYL_B = ["a", "e", "i", "o", "u", "ai", "ou"]
_SYL_C = ["x", "rb", "nd", "mp", "sk", "lt", "rk", "v", "z", "t", "p", "g"]


def nonce_word(k: int) -> str:
    a = _SYL_A[k % len(_SYL_A)]
    b = _SYL_B[(k // len(_SYL_A)) % len(_SYL_B)]
    c = _SYL_C[(k // (len(_SYL_A) * len(_SYL_B))) % len(_SYL_C)]
    return a + b + c + ("o" if k >= len(_SYL_A) * len(_SYL_B) * len(_SYL_C) else "")


def _content_words(entity: str) -> List[str]:
    return [w for a in pc.split_aliases(entity) for w in pc.tokenize(a) if w not in pc.STOPWORDS and not w.isdigit()]


def nonce_variant(raw: dict, vocab: Optional[set] = None, offset: int = 0) -> Tuple[Optional[dict], Dict[str, str]]:
    """Trả về (raw mới, ánh xạ) hoặc (None, ...) nếu nhãn bị thay đổi (khi đó bỏ đoạn khỏi biến thể)."""
    words = []
    for e in raw["participants"]:
        for w in _content_words(e):
            base = w[:-1] if w.endswith("s") and len(w) > 3 else w
            if base not in words:
                words.append(base)
    vocab = vocab or set()
    mapping, k = {}, offset
    for w in words:
        nw = nonce_word(k)
        while nw in vocab or nw in mapping.values():
            k += 1
            nw = nonce_word(k)
        mapping[w] = nw
        k += 1

    def sub(text: str) -> str:
        def rep(mt):
            tokw = mt.group(0)
            low = tokw.lower()
            stem, plural = (low[:-1], True) if low.endswith("s") and low[:-1] in mapping else (low, False)
            if stem not in mapping:
                return tokw
            out = mapping[stem] + ("s" if plural else "")
            return out.capitalize() if tokw[0].isupper() else out
        return re.sub(r"[A-Za-z]+", rep, text)

    new = {"para_id": raw["para_id"], "sentence_texts": [sub(s) for s in raw["sentence_texts"]],
           "participants": [sub(p) for p in raw["participants"]],
           "states": [[v if v in (pc.NO_LOC, pc.UNK_LOC) else sub(v) for v in st] for st in raw["states"]]}
    old_l = [[pc.derive_label(st[t - 1], st[t]) for t in range(1, len(st))] for st in raw["states"]]
    new_l = [[pc.derive_label(st[t - 1], st[t]) for t in range(1, len(st))] for st in new["states"]]
    if old_l != new_l:
        return None, mapping
    return new, mapping


def nonce_records(raws_by_split: Dict[str, List[dict]], topics: Dict[str, str]) -> Dict[str, List[dict]]:
    vocab = set()
    for rows in raws_by_split.values():
        for r in rows:
            for s in r["sentence_texts"]:
                vocab.update(pc.tokenize(s))
    out = {}
    for sp, rows in raws_by_split.items():
        recs = []
        for r in rows:
            nr, _ = nonce_variant(r, vocab)
            if nr is not None:
                rec = pc.build_record(nr, sp, topics.get(str(r["para_id"])))
                rec["variant"] = "nonce_v1"
                recs.append(rec)
        out[sp] = recs
    return out
