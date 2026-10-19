"""Kiểm thử CRF của PP4 bằng liệt kê vét cạn (cần torch, không cần GPU/transformers)."""
import random

import torch

import propara_core as pc
import propara_pp4 as pp


def _rand(n, T, seed, with_psi=True):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(T, n, 4, generator=g, dtype=torch.float64)
    psi = torch.randn(T, n, n, generator=g, dtype=torch.float64) if with_psi else None
    init = torch.randn(n, generator=g, dtype=torch.float64)
    return phi, psi, init


def test_log_partition_matches_brute_force():
    for (n, T, seed) in [(1, 3, 0), (2, 2, 1), (2, 3, 2), (3, 2, 3)]:
        for with_psi in (False, True):
            phi, psi, init = _rand(n, T, seed, with_psi)
            z = pp.crf_log_partition(phi, psi, init)
            zb, _, _ = pp.brute_force(phi, psi, init)
            assert abs(float(z) - float(zb)) < 1e-8, (n, T, with_psi, float(z), float(zb))


def test_viterbi_matches_brute_force():
    for (n, T, seed) in [(2, 2, 4), (2, 3, 5), (3, 2, 6)]:
        phi, psi, init = _rand(n, T, seed)
        labels, init_e, best = pp.crf_viterbi(phi, psi, init)
        _, bb, (lab, e0) = pp.brute_force(phi, psi, init)
        assert abs(best - bb) < 1e-8
        assert [[pp.IDX_LAB[int(lab[t, i])] for t in range(T)] for i in range(n)] == labels
        assert init_e == [bool(x) for x in e0]


def test_viterbi_output_always_satisfies_constraints():
    for seed in range(20):
        n, T = random.Random(seed).randint(1, 4), random.Random(seed + 1).randint(1, 5)
        phi, psi, init = _rand(n, T, 100 + seed)
        labels, init_e, _ = pp.crf_viterbi(phi * 5, psi, init)
        assert pc.count_constraint_violations(labels, init_e)[0] == 0


def test_nll_nonnegative_and_gradient_flows():
    phi, psi, init = _rand(3, 3, 7)
    phi.requires_grad_(True); psi.requires_grad_(True)
    gl = torch.tensor([[3, 3, 0], [1, 3, 3], [3, 3, 3]])          # e0=(1,1,0): t1 C cho thực thể 2, t2 D cho thực thể 0
    gi = torch.tensor([1, 1, 0])
    nll = pp.crf_nll(phi, psi, init, gl, gi)
    assert float(nll.detach()) >= 0
    nll.backward()
    assert phi.grad is not None and float(psi.grad.abs().sum()) > 0


def test_pair_score_rewards_simultaneous_destroy_create():
    # φ trung tính; ψ(0→1) lớn ở câu 1 → Viterbi chọn 0 bị DESTROY và 1 được CREATE cùng lúc
    phi = torch.zeros(1, 2, 4, dtype=torch.float64)
    psi = torch.zeros(1, 2, 2, dtype=torch.float64); psi[0, 0, 1] = 5.0
    init = torch.tensor([3.0, -3.0], dtype=torch.float64)
    labels, init_e, _ = pp.crf_viterbi(phi, psi, init)
    assert init_e == [True, False] and labels[0][0] == pc.DESTROY and labels[1][0] == pc.CREATE


def test_existence_marginals_match_brute_force():
    import itertools
    for (n, T, seed) in [(2, 2, 11), (3, 2, 12), (2, 3, 13)]:
        phi, psi, init = _rand(n, T, seed)
        m = pp.crf_existence_marginals(phi, psi, init)
        # vét cạn: liệt kê mọi cấu hình hợp lệ, cộng trọng số theo tồn tại
        acc = torch.zeros(T + 1, n, dtype=torch.float64); Z = 0.0
        for e0 in itertools.product([0, 1], repeat=n):
            opts = []
            for i in range(n):
                seqs = []
                def rec(tt, e, a):
                    if tt == T:
                        seqs.append(tuple(a)); return
                    for y in ((1, 2, 3) if e else (0, 3)):
                        rec(tt + 1, (0 if y == 1 else 1 if y == 0 else e), a + [y])
                rec(0, e0[i], [])
                opts.append(seqs)
            for cfg in itertools.product(*opts):
                lab = torch.tensor([[cfg[i][tt] for i in range(n)] for tt in range(T)])
                w = float(torch.exp(pp.crf_gold_score(phi, psi, init, lab, torch.tensor(e0))))
                Z += w
                for i in range(n):
                    e = e0[i]; acc[0, i] += w * e
                    for tt in range(T):
                        y = cfg[i][tt]; e = 0 if y == 1 else 1 if y == 0 else e
                        acc[tt + 1, i] += w * e
        assert torch.allclose(m, acc / Z, atol=1e-8), (m, acc / Z)


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
