"""propara_pp4 — PP4: RoBERTa + BiLSTM, điểm đơn φ, điểm cặp ψ, CRF giải mã chung (mục 4.3–4.4 báo cáo).

Phiên bản: pp4_v1. Cần torch; phần mô hình cần transformers (nạp muộn).

CRF chính xác trên không gian TỒN TẠI: trạng thái e ∈ {0,1}^n (2^n ≤ 256 trên ProPara vì n ≤ 8).
Chuyển e → e' xác định loại của từng thực thể:
   (0,0) chưa tồn tại, giữ nguyên → NONE
   (0,1) CREATE
   (1,0) DESTROY
   (1,1) đang tồn tại, giữ nguyên → MOVE hoặc NONE (cộng gộp bằng logsumexp, Viterbi lấy max)
Mọi chuỗi ngoài ràng buộc (15) có xác suất 0 theo cấu tạo. Cặp ψ_t(i,j) cộng vào khi i bị DESTROY và j được CREATE
ở cùng câu t (công thức 14); tồn tại của i trước câu và không tồn tại của j trước câu đã được ràng buộc đảm bảo,
nên bộ lọc PP2 không loại thêm cặp nào trong giải mã chính xác (PP2 chỉ còn vai trò tăng tốc/khi giải mã xấp xỉ).
"""
from __future__ import annotations

import math
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import propara_core as pc

PP4_VERSION = "pp4_v1.3"
LAB_IDX = {pc.CREATE: 0, pc.DESTROY: 1, pc.MOVE: 2, pc.NONE: 3}
IDX_LAB = {v: k for k, v in LAB_IDX.items()}
_STATE_CACHE: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}


# ============================================================================ CRF
def _state_tables(n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """bits (S,n) và loại chuyển K (S,S,n) ∈ {0:(0,0),1:(0,1)=C,2:(1,0)=D,3:(1,1)}."""
    key = (n, str(device))
    if key not in _STATE_CACHE:
        S = 1 << n
        idx = torch.arange(S, device=device)
        bits = ((idx[:, None] >> torch.arange(n, device=device)[None, :]) & 1).long()
        K = 2 * bits[:, None, :] + bits[None, :, :]
        _STATE_CACHE[key] = (bits, K)
    return _STATE_CACHE[key]


def _transition_scores(phi_t: torch.Tensor, psi_t: Optional[torch.Tensor], K: torch.Tensor, use_max: bool):
    """phi_t (n,4) theo thứ tự [C,D,M,N]; psi_t (n,n) hoặc None. Trả về (S,S) và (nếu use_max) lựa chọn M/N (n,)."""
    stay = torch.stack([phi_t[:, 2], phi_t[:, 3]], -1)
    stay_val = stay.max(-1).values if use_max else torch.logsumexp(stay, -1)
    table = torch.stack([phi_t[:, 3], phi_t[:, 0], phi_t[:, 1], stay_val], -1)       # (n,4) theo loại 0..3
    n = phi_t.shape[0]
    sc = table[torch.arange(n, device=phi_t.device)[None, None, :].expand_as(K), K].sum(-1)   # (S,S)
    if psi_t is not None:
        Dm = (K == 2).to(phi_t.dtype)
        Cm = (K == 1).to(phi_t.dtype)
        psi0 = psi_t * (1 - torch.eye(n, device=phi_t.device, dtype=phi_t.dtype))
        sc = sc + torch.einsum("abi,ij,abj->ab", Dm, psi0, Cm)
    return sc, (stay[:, 0] >= stay[:, 1])


def crf_log_partition(phi: torch.Tensor, psi: Optional[torch.Tensor], init: torch.Tensor) -> torch.Tensor:
    """phi (T,n,4), psi (T,n,n) hoặc None, init (n,) điểm 'tồn tại ban đầu'. Trả về log Z."""
    T, n, _ = phi.shape
    bits, K = _state_tables(n, phi.device)
    alpha = (bits.to(phi.dtype) * init[None, :]).sum(-1)                               # (S,)
    for t in range(T):
        tr, _ = _transition_scores(phi[t], None if psi is None else psi[t], K, use_max=False)
        alpha = torch.logsumexp(alpha[:, None] + tr, dim=0)
    return torch.logsumexp(alpha, 0)


def crf_existence_marginals(phi: torch.Tensor, psi: Optional[torch.Tensor], init: torch.Tensor) -> torch.Tensor:
    """p(e_{t,i} = 1) cho t = 0..T (t = 0: trước câu 1) bằng forward-backward trên 2^n trạng thái tồn tại.
    Dùng cho PP2 (công thức 11): p(e_{X,t-1}=1) > τ1 và p(e_{Y,t-1}=1) < τ2."""
    with torch.no_grad():
        T, n, _ = phi.shape
        bits, K = _state_tables(n, phi.device)
        trs = [_transition_scores(phi[t], None if psi is None else psi[t], K, use_max=False)[0] for t in range(T)]
        alphas = [(bits.to(phi.dtype) * init[None, :]).sum(-1)]
        for t in range(T):
            alphas.append(torch.logsumexp(alphas[-1][:, None] + trs[t], dim=0))
        logZ = torch.logsumexp(alphas[-1], 0)
        betas = [torch.zeros_like(alphas[0])]
        for t in range(T - 1, -1, -1):
            betas.append(torch.logsumexp(trs[t] + betas[-1][None, :], dim=1))
        betas.reverse()
        out = []
        for t in range(T + 1):
            pst = torch.exp(alphas[t] + betas[t] - logZ)                      # (S,)
            out.append((pst[:, None] * bits.to(phi.dtype)).sum(0))
        return torch.stack(out)                                               # (T+1, n)


def crf_gold_score(phi, psi, init, gold_lab: torch.Tensor, gold_init: torch.Tensor) -> torch.Tensor:
    """gold_lab (T,n) chỉ số nhãn [C,D,M,N]; gold_init (n,) 0/1."""
    T, n, _ = phi.shape
    s = (gold_init.to(phi.dtype) * init).sum()
    s = s + phi.gather(-1, gold_lab[..., None]).squeeze(-1).sum()
    if psi is not None:
        D = (gold_lab == LAB_IDX[pc.DESTROY]).to(phi.dtype)
        C = (gold_lab == LAB_IDX[pc.CREATE]).to(phi.dtype)
        eye = torch.eye(n, device=phi.device, dtype=phi.dtype)
        s = s + torch.einsum("ti,tij,tj->", D, psi * (1 - eye), C)
    return s


def crf_nll(phi, psi, init, gold_lab, gold_init) -> torch.Tensor:
    return crf_log_partition(phi, psi, init) - crf_gold_score(phi, psi, init, gold_lab, gold_init)


def crf_viterbi(phi: torch.Tensor, psi: Optional[torch.Tensor], init: torch.Tensor) -> Tuple[List[List[str]], List[bool], float]:
    """Trả về (nhãn [n][T], tồn tại ban đầu [n], điểm tốt nhất)."""
    with torch.no_grad():
        T, n, _ = phi.shape
        bits, K = _state_tables(n, phi.device)
        delta = (bits.to(phi.dtype) * init[None, :]).sum(-1)
        back, mn_choice = [], []
        for t in range(T):
            tr, mchoice = _transition_scores(phi[t], None if psi is None else psi[t], K, use_max=True)
            cand = delta[:, None] + tr
            delta, arg = cand.max(0)
            back.append(arg)
            mn_choice.append(mchoice)
        e = int(delta.argmax())
        best = float(delta[e])
        path = [e]
        for t in range(T - 1, -1, -1):
            e = int(back[t][e])
            path.append(e)
        path.reverse()                                                                  # e_0 .. e_T
        labels = [[pc.NONE] * T for _ in range(n)]
        for t in range(T):
            a, b = path[t], path[t + 1]
            for i in range(n):
                ea, eb = (a >> i) & 1, (b >> i) & 1
                if ea == 0 and eb == 1:
                    labels[i][t] = pc.CREATE
                elif ea == 1 and eb == 0:
                    labels[i][t] = pc.DESTROY
                elif ea == 1 and eb == 1:
                    labels[i][t] = pc.MOVE if bool(mn_choice[t][i]) else pc.NONE
        init_exists = [bool((path[0] >> i) & 1) for i in range(n)]
        return labels, init_exists, best


def brute_force(phi, psi, init):
    """Liệt kê vét cạn mọi cấu hình hợp lệ (chỉ dùng trong kiểm thử, n·T rất nhỏ)."""
    import itertools
    T, n, _ = phi.shape
    best, best_cfg, scores = -float("inf"), None, []
    for e0 in itertools.product([0, 1], repeat=n):
        per_entity_opts = []
        for i in range(n):
            seqs = []
            def rec(t, e, acc):
                if t == T:
                    seqs.append(tuple(acc)); return
                for y in ((1, 2, 3) if e else (0, 3)):
                    rec(t + 1, (0 if y == 1 else 1 if y == 0 else e), acc + [y])
            rec(0, e0[i], [])
            per_entity_opts.append(seqs)
        for cfg in itertools.product(*per_entity_opts):
            lab = torch.tensor([[cfg[i][t] for i in range(n)] for t in range(T)])
            s = float(crf_gold_score(phi, psi, init, lab, torch.tensor(e0)))
            scores.append(s)
            if s > best:
                best, best_cfg = s, (lab, e0)
    return torch.logsumexp(torch.tensor(scores, dtype=torch.float64), 0), best, best_cfg


# ============================================================================ đặc trưng văn bản
def word_tokens(sentence: str) -> List[str]:
    return sentence.split()


def mention_word_mask(entity: str, words: List[str], mode: str = "report") -> List[bool]:
    """Từ nào trong câu khớp từ nội dung của tên thực thể (cùng quy tắc cờ nhắc của core)."""
    out = []
    for w in words:
        out.append(pc.is_mentioned(entity, w, mode))
    return out


def build_entity_input(tok, rec: dict, i: int, max_len: int = 512):
    """<s> tên thực thể </s></s> câu1 câu2 ... </s>; trả về ids, span mỗi câu, chỉ số token được nhắc mỗi câu."""
    ent = rec["entities"][i]
    ids = [tok.cls_token_id] + tok.encode(" " + ent.replace(";", " /"), add_special_tokens=False)[:30] + [tok.sep_token_id, tok.sep_token_id]
    name_span = (1, len(ids) - 2)
    spans, ment, word_toks = [], [], []
    for t, s in enumerate(rec["sentences"]):
        start = len(ids)
        mt, wt = [], []
        for k, w in enumerate(word_tokens(s)):
            wid = tok.encode(" " + w, add_special_tokens=False)
            if pc.is_mentioned(ent, w):
                mt.extend(range(len(ids), len(ids) + len(wid)))
            wt.append((len(ids), len(ids) + len(wid)))
            ids.extend(wid)
        spans.append((start, len(ids)))
        ment.append(mt)
        word_toks.append(wt)
    ids.append(tok.sep_token_id)
    if len(ids) > max_len:
        raise ValueError(f"đoạn {rec['paragraph_id']} dài {len(ids)} > {max_len} token")
    return ids, spans, ment, name_span, word_toks


_NP_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VBN", "VBG", "CD", "DT", "PRP$"}
_NOUN = {"NN", "NNS", "NNP", "NNPS"}
_LOC_STOP = {"the", "a", "an", "its", "their", "his", "her", "this", "that", "these", "those", "some", "any"}


def norm_loc(s: str) -> str:
    """Chuẩn hoá chuỗi vị trí để so khớp: chữ thường, bỏ mạo từ/sở hữu, đưa về số ít đơn giản."""
    ws = [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _LOC_STOP]
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in ws)


def location_candidates(rec: dict, tagger=None) -> List[dict]:
    """Ứng viên vị trí (mục 4.3 báo cáo: 'đầu riêng chấm điểm trên các cụm danh từ').
    Mỗi cụm danh từ cho 3 ứng viên: cả cụm (bỏ mạo từ), 2 từ cuối, từ cuối. Gộp theo chuỗi chuẩn hoá.
    Không có tagger: lùi về cửa sổ 1–2 từ không phải từ rỗng."""
    cands: Dict[str, dict] = {}

    def add(norm, surface, occ):
        if not norm:
            return
        c = cands.setdefault(norm, {"norm": norm, "surface": surface, "occ": []})
        c["occ"].append(occ)

    for t, s in enumerate(rec["sentences"]):
        words = word_tokens(s)
        clean = [re.sub(r"[^A-Za-z0-9\-']", "", w) for w in words]
        if tagger is not None:
            idx = [k for k, w in enumerate(clean) if w]
            tags = dict(zip(idx, [tg for _, tg in tagger([clean[k] for k in idx])])) if idx else {}
            k = 0
            while k < len(words):
                if tags.get(k) in _NP_TAGS:
                    j = k + 1
                    while j < len(words) and tags.get(j) in _NP_TAGS and tags.get(j) not in ("DT", "PRP$"):
                        j += 1
                    last_noun = max((q for q in range(k, j) if tags.get(q) in _NOUN), default=None)
                    if last_noun is not None:
                        first = k
                        while first <= last_noun and clean[first].lower() in _LOC_STOP:
                            first += 1
                        for a in {first, max(first, last_noun - 1), last_noun}:
                            phrase = " ".join(clean[a:last_noun + 1]).lower()
                            add(norm_loc(phrase), phrase, (t, a, last_noun + 1))
                    k = j
                else:
                    k += 1
        else:
            for k in range(len(words)):
                for L in (1, 2):
                    seg = clean[k:k + L]
                    if len(seg) == L and all(w and w.lower() not in pc.STOPWORDS for w in seg):
                        phrase = " ".join(seg).lower()
                        add(norm_loc(phrase), phrase, (t, k, k + L))
    return list(cands.values())


def last_mention_distance(rec: dict, i: int, t: int) -> float:
    """d_it: số câu kể từ lần nhắc gần nhất của thực thể i trước câu t (chưa từng nhắc → T)."""
    T = len(rec["sentences"])
    for k in range(t - 1, 0, -1):
        if rec["mentions"][i][k - 1]:
            return float(t - k)
    return float(T)


# ============================================================================ mô hình
class PP4Model(nn.Module):
    """variant ∈ {"M0", "CRF_NOPAIR", "CRF_NOE", "CRF_FULL"}; M0 = phân loại từng ô độc lập (cross-entropy)."""

    def __init__(self, encoder, hidden: int, variant: str, lstm_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.variant = variant
        self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True, bidirectional=True)
        H = 2 * lstm_hidden
        self.att_q = nn.Linear(H, H)
        self.drop = nn.Dropout(dropout)
        self.phi = nn.Sequential(nn.Linear(H + 1, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 4))
        self.init_head = nn.Linear(H, 1)
        self.pair_w = nn.Linear(6, 1)        # [log E, m_i, m_j, d_i/T, cos, 1]
        nn.init.zeros_(self.pair_w.weight)
        nn.init.zeros_(self.pair_w.bias)
        self.loc_q = nn.Linear(H + 1, H)      # đầu vị trí: chấm điểm ứng viên cụm danh từ
        self.loc_unk = nn.Linear(H + 1, 1)    # lựa chọn '?' (tồn tại nhưng không rõ vị trí)
        self.lambda_loc = 1.0

    def entity_states(self, batch_ids, attn, spans, ment, name_spans):
        out = self.encoder(input_ids=batch_ids, attention_mask=attn).last_hidden_state
        out, _ = self.lstm(out.float())
        out = self.drop(out)
        n = out.shape[0]
        T = len(spans[0])
        H = out.shape[-1]
        h = out.new_zeros(T, n, H)
        m = out.new_zeros(T, n)
        pooled = out.new_zeros(n, H)
        for i in range(n):
            a, b = name_spans[i]
            q = self.att_q(out[i, a:b].mean(0))
            pooled[i] = out[i, spans[i][0][0]:spans[i][-1][1]].mean(0)
            for t, (s, e) in enumerate(spans[i]):
                if ment[i][t]:
                    h[t, i] = out[i, ment[i][t]].mean(0)
                    m[t, i] = 1.0
                else:
                    seg = out[i, s:e]
                    w = torch.softmax(seg @ q / math.sqrt(H), 0)
                    h[t, i] = (w[:, None] * seg).sum(0)
        return h, m, pooled, out

    def potentials(self, feats: dict, device):
        h, m, pooled, out = self.entity_states(feats["ids"].to(device), feats["attn"].to(device), feats["spans"],
                                               feats["ment"], feats["name_spans"])
        T, n, _ = h.shape
        phi = self.phi(torch.cat([h, m[..., None]], -1))                                 # (T,n,4)
        init = self.init_head(pooled).squeeze(-1)                                         # (n,)
        psi = None
        if self.variant in ("CRF_NOE", "CRF_FULL"):
            hn = F.normalize(h, dim=-1)
            cos = torch.einsum("tid,tjd->tij", hn, hn)
            logE = feats["logE"].to(device) if self.variant == "CRF_FULL" else torch.zeros(n, n, device=device)
            mi = m[:, :, None].expand(T, n, n)
            mj = m[:, None, :].expand(T, n, n)
            d = feats["dist"].to(device)[:, :, None].expand(T, n, n)
            f = torch.stack([logE[None].expand(T, n, n), mi, mj, d, cos, torch.ones_like(cos)], -1)
            psi = self.pair_w(f).squeeze(-1)
        loc = None
        if feats.get("cand_tok") and len(feats["cand_tok"][0]) > 0:
            hm = torch.cat([h, m[..., None]], -1)                                        # (T,n,H+1)
            q = self.loc_q(hm)                                                            # (T,n,H)
            K = len(feats["cand_tok"][0])
            loc = h.new_zeros(T, n, K + 1)
            for i in range(n):
                C = torch.stack([out[i, toks].mean(0) for toks in feats["cand_tok"][i]])  # (K,H)
                loc[:, i, :K] = q[:, i] @ C.T / math.sqrt(C.shape[-1])
            loc[:, :, K] = self.loc_unk(hm).squeeze(-1)
        self._loc = loc
        return phi, psi, init

    def loss(self, feats: dict, device):
        phi, psi, init = self.potentials(feats, device)
        gl, gi = feats["gold_lab"].to(device), feats["gold_init"].to(device)
        if self.variant == "M0":
            ce = F.cross_entropy(phi.reshape(-1, 4), gl.reshape(-1), reduction="sum")
            bce = F.binary_cross_entropy_with_logits(init, gi.float(), reduction="sum")
            main = (ce + bce) / gl.numel()
        else:
            main = crf_nll(phi.float(), None if psi is None else psi.float(), init.float(), gl, gi) / gl.numel()
        # L_total = L_PP4 + λ_loc · L_vị trí  (chỉ trên ô vàng tồn tại sau câu và có vị trí khớp ứng viên)
        if self._loc is not None:
            tgt = feats["loc_target"].to(device)
            ok = tgt >= 0
            if ok.any():
                main = main + self.lambda_loc * F.cross_entropy(self._loc.float()[ok], tgt[ok], reduction="sum") / gl.numel()
        return main

    @torch.no_grad()
    def predict(self, feats: dict, device, constrained: Optional[bool] = None):
        """M0: argmax từng ô (constrained=False) hoặc Viterbi với log-softmax (M0c, constrained=True).
        CRF: luôn Viterbi."""
        phi, psi, init = self.potentials(feats, device)
        if self.variant == "M0" and not constrained:
            lab = phi.argmax(-1).T.tolist()
            labels = [[IDX_LAB[y] for y in row] for row in lab]
            init_e = (init > 0).tolist()
        else:
            if self.variant == "M0":
                phi = F.log_softmax(phi.float(), -1)
                init = F.logsigmoid(init.float()) - F.logsigmoid(-init.float())
            labels, init_e, _ = crf_viterbi(phi.float(), None if psi is None else psi.float(), init.float())
        la = self._decode_locations(labels, init_e, feats)
        return labels, init_e, (la, self._init_loc)

    def _decode_locations(self, labels, init_e, feats):
        """Vị trí sau mỗi câu, nhất quán với nhãn: CREATE/MOVE → ứng viên điểm cao nhất; NONE → giữ; DESTROY → '-'."""
        n, T = len(labels), len(labels[0])
        surf = [c for c in feats.get("cand_surface", [])]
        best = self._loc.argmax(-1).tolist() if self._loc is not None else None
        out, self._init_loc = [], []
        for i in range(n):
            cur = pc.NO_LOC
            if init_e[i]:                     # vị trí ban đầu = ứng viên điểm cao nhất ở câu 1
                k0 = best[0][i] if best is not None else len(surf)
                cur = surf[k0] if k0 < len(surf) else "?"
            self._init_loc.append(cur)
            row = []
            for t in range(T):
                y = labels[i][t]
                if y == pc.DESTROY:
                    cur = pc.NO_LOC
                elif y in (pc.CREATE, pc.MOVE):
                    k = best[t][i] if best is not None else len(surf)
                    cur = surf[k] if k < len(surf) else "?"
                row.append(cur)
            out.append(row)
        return out

    @torch.no_grad()
    def existence_probs(self, feats: dict, device) -> torch.Tensor:
        """p(tồn tại) trước/sau mỗi câu (T+1, n) từ biên CRF; M0 dùng log-softmax trong cùng khung CRF (PP2, công thức 11)."""
        phi, psi, init = self.potentials(feats, device)
        if self.variant == "M0":
            phi = F.log_softmax(phi.float(), -1)
            init = F.logsigmoid(init.float()) - F.logsigmoid(-init.float())
        return crf_existence_marginals(phi.float(), None if psi is None else psi.float(), init.float())


def featurize(tok, rec: dict, logE: Optional[List[List[float]]] = None, max_len: int = 512, tagger=None) -> dict:
    n, T = len(rec["entities"]), len(rec["sentences"])
    seqs, spans, ment, names, wtoks = [], [], [], [], []
    for i in range(n):
        ids, sp, mt, ns, wt = build_entity_input(tok, rec, i, max_len)
        seqs.append(ids); spans.append(sp); ment.append(mt); names.append(ns); wtoks.append(wt)
    cands = location_candidates(rec, tagger)
    cand_tok = [[sorted({x for (t, a, b) in c["occ"] for w in range(a, b) for x in range(*wtoks[i][t][w])}) for c in cands]
                for i in range(n)]
    norm2k = {c["norm"]: k for k, c in enumerate(cands)}
    K = len(cands)
    loc_target = torch.full((T, n), -1, dtype=torch.long)
    n_loc, n_hit = 0, 0
    for i in range(n):
        for t in range(1, T + 1):
            st = rec["states"][i][t]
            if st == pc.NO_LOC:
                continue
            n_loc += 1
            if st == pc.UNK_LOC:
                loc_target[t - 1, i] = K; n_hit += 1
            elif norm_loc(st) in norm2k:
                loc_target[t - 1, i] = norm2k[norm_loc(st)]; n_hit += 1
    L = max(len(s) for s in seqs)
    ids = torch.full((n, L), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
    gold_lab = torch.tensor([[LAB_IDX[rec["labels"][i][t]] for i in range(n)] for t in range(T)])
    gold_init = torch.tensor([int(rec["existence"][i][0]) for i in range(n)])
    dist = torch.tensor([[last_mention_distance(rec, i, t) / T for i in range(n)] for t in range(1, T + 1)])
    E = torch.tensor(logE, dtype=torch.float) if logE is not None else torch.zeros(n, n)
    return {"ids": ids, "attn": attn, "spans": spans, "ment": ment, "name_spans": names,
            "gold_lab": gold_lab, "gold_init": gold_init, "dist": dist, "logE": E, "pid": rec["paragraph_id"],
            "cand_tok": cand_tok if K else [], "cand_surface": [c["surface"] for c in cands], "loc_target": loc_target,
            "loc_cover": (n_hit, n_loc)}


def to_prediction(rec: dict, labels, init_exists, locs=None) -> dict:
    p = {"paragraph_id": rec["paragraph_id"], "labels": labels, "exists_initial": [bool(x) for x in init_exists]}
    if locs is not None:
        p["locations_after"], p["location_initial"] = locs
    return p


# ============================================================================ huấn luyện
def train_variant(make_encoder, tok, variant: str, train_feats: List[dict], dev_recs: List[dict], dev_feats: List[dict],
                  evaluate_fn, seed: int, epochs: int = 8, lr_enc: float = 2e-5, lr_head: float = 1e-3,
                  accum: int = 4, device: str = "cuda", log=print, use_amp: bool = True,
                  ckpt_path: Optional[str] = None, patience: Optional[int] = None, stop_after_epochs: Optional[int] = None):
    """Huấn luyện một biến thể; chọn epoch theo PAIR_F1 trên DEV (quy tắc khoá trước; hoà → dev loss thấp hơn).
    ckpt_path: sau MỖI epoch lưu đủ trạng thái (mô hình, optimizer, scheduler, scaler, RNG, lịch sử, bản tốt nhất)
               → chạy lại sẽ TIẾP TỤC từ epoch kế tiếp. Người gọi xoá file khi đã lưu kết quả.
    patience:  dừng sớm khi PAIR_F1 dev (tốt nhất trong các tên theo dõi) không tăng sau `patience` epoch liên tiếp.
    stop_after_epochs: chỉ dùng để KIỂM THỬ việc bị ngắt (giả lập phiên chết sau N epoch của lần gọi này).
    Với M0: theo dõi ĐỒNG THỜI M0 (argmax) và M0c (Viterbi ràng buộc trên cùng mô hình), mỗi cái chọn epoch riêng.
    Trả về (model, lịch sử, {tên: (state_dict tốt nhất, epoch tốt nhất)})."""
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc, hidden = make_encoder()
    model = PP4Model(enc, hidden, variant).to(device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": lr_enc}, {"params": head_params, "lr": lr_head}],
                            weight_decay=0.01)
    steps = max(1, epochs * math.ceil(len(train_feats) / accum))
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / warm) * max(0.0, (steps - st) / steps))
    amp = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    names = ["M0", "M0c"] if variant == "M0" else [variant]
    best = {nm: ((-1.0, -float("inf")), None, None) for nm in names}
    hist, order = [], list(range(len(train_feats)))
    start_ep, bad, best_main = 1, 0, -1.0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); hist, order = ck["hist"], ck["order"]
        for nm in names:                      # bản tốt nhất được lưu riêng (chỉ ghi khi cải thiện)
            key_, ep_ = ck["best_meta"][nm]
            bp = f"{ckpt_path}.best_{nm}.pt"
            best[nm] = (tuple(key_), torch.load(bp, map_location="cpu", weights_only=True) if os.path.exists(bp) else None, ep_)
        bad, best_main, start_ep = ck["bad"], ck["best_main"], ck["epoch"] + 1
        random.setstate(ck["py_rng"]); torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        log(f"  [{variant} seed={seed}] TIẾP TỤC từ checkpoint epoch {ck['epoch']}" + (" (đã dừng sớm)" if ck.get("stopped") else ""))
        if ck.get("stopped"):
            start_ep = epochs + 1
    n_run = 0
    for ep in range(start_ep, epochs + 1):
        model.train()
        random.shuffle(order)
        t0, tot = time.time(), 0.0
        opt.zero_grad()
        for k, j in enumerate(order, 1):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = model.loss(train_feats[j], device)
            scaler.scale(loss / accum).backward()
            tot += float(loss.detach())
            if k % accum == 0 or k == len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        model.eval()
        dev_loss, preds = 0.0, {nm: {} for nm in names}
        with torch.no_grad():
            for rec, f in zip(dev_recs, dev_feats):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    dev_loss += float(model.loss(f, device))
                for nm in names:
                    lab, ie, la = model.predict(f, device, constrained=(nm == "M0c"))
                    preds[nm][rec["paragraph_id"]] = to_prediction(rec, lab, ie, la)
        row = {"epoch": ep, "train_loss": tot / len(order), "dev_loss": dev_loss / len(dev_recs), "sec": round(time.time() - t0, 1)}
        for nm in names:
            res = evaluate_fn(preds[nm], dev_recs)
            row[f"{nm}:dev_PAIR_F1"] = res["PAIR@all"]["F1"]
            row[f"{nm}:dev_PAIR_F1@any_imp"] = res["PAIR@any_implicit"]["F1"]
            row[f"{nm}:dev_IMP_ACC_DESTROY"] = res["IMPLICIT_ACC"]["DESTROY"]["acc"]
            row[f"{nm}:dev_IMP_ACC_CREATE"] = res["IMPLICIT_ACC"]["CREATE"]["acc"]
            row[f"{nm}:dev_PAIR_LOC_F1"] = res["PAIR_LOC"]["F1"]
            key = (res["PAIR@all"]["F1"], -row["dev_loss"])
            if key > best[nm][0]:
                best[nm] = (key, {kk: v.detach().to("cpu", copy=True) for kk, v in model.state_dict().items()}, ep)
                if ckpt_path:                    # ghi bản tốt nhất CHỈ khi cải thiện (tránh ghi 2,5 GB mỗi epoch)
                    torch.save(best[nm][1], f"{ckpt_path}.best_{nm}.pt.tmp")
                    os.replace(f"{ckpt_path}.best_{nm}.pt.tmp", f"{ckpt_path}.best_{nm}.pt")
        hist.append(row)
        cur = max(row[f"{nm}:dev_PAIR_F1"] for nm in names)
        if cur > best_main + 1e-9:
            best_main, bad = cur, 0
        else:
            bad += 1
        stopped = patience is not None and bad >= patience and ep < epochs
        log(f"  [{variant} seed={seed}] ep{ep} train={row['train_loss']:.4f} dev={row['dev_loss']:.4f} " +
            " ".join(f"{nm}:F1={row[nm + ':dev_PAIR_F1']:.3f}/imp={row[nm + ':dev_PAIR_F1@any_imp']:.3f}" for nm in names) +
            f" ({row['sec']}s)" + (f" → DỪNG SỚM (không tăng {bad} epoch)" if stopped else ""))
        if ckpt_path:
            tmp = ckpt_path + ".tmp"
            torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "hist": hist, "order": order,
                        "best_meta": {nm: (list(b[0]), b[2]) for nm, b in best.items()}, "bad": bad,
                        "best_main": best_main, "stopped": stopped, "py_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
            os.replace(tmp, ckpt_path)
        n_run += 1
        if stopped:
            break
        if stop_after_epochs is not None and n_run >= stop_after_epochs and ep < epochs:
            raise KeyboardInterrupt(f"giả lập phiên bị ngắt sau epoch {ep}")
    return model, hist, {nm: (b[1], b[2]) for nm, b in best.items()}


def cleanup_ckpt(ckpt_path: str) -> None:
    """Xoá checkpoint đang-chạy và các file bản-tốt-nhất đi kèm (gọi sau khi đã lưu kết quả)."""
    import glob
    for f in [ckpt_path] + glob.glob(ckpt_path + ".best_*.pt*"):
        if os.path.exists(f):
            os.remove(f)


@torch.no_grad()
def predict_all(model, recs: List[dict], feats: List[dict], device: str, constrained: Optional[bool] = None) -> Dict[str, dict]:
    model.eval()
    out = {}
    for rec, f in zip(recs, feats):
        lab, ie, la = model.predict(f, device, constrained=constrained)
        out[rec["paragraph_id"]] = to_prediction(rec, lab, ie, la)
    return out
