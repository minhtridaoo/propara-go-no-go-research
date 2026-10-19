"""final_analysis.py — D7/D8: chấm MỘT LẦN các dự đoán test đã lưu kín, thống kê, áp quy tắc quyết định đã khoá.

Chạy trên CPU (trợ lý), sau khi có output K1 (qwen7b_dev_v1), K2 (pp34_v1), K3 (qwen7b_test_v1) đã giải nén.
  python tools/final_analysis.py --data <thư mục dữ liệu raw ProPara> --k1 <qwen7b_dev_v1> --k2 <pp34_v1> --k3 <qwen7b_test_v1> --out <results/final>

Quy tắc đã đăng ký trước (protocol v1.1 + §7):
  * Hệ thống Qwen chính = chế độ đầu-cuối có PAIR_F1 DEV cao nhất trong {DIRECT, FEWSHOT, STRUCTURED, CONSTRAINED} (đọc từ K1).
  * M3 / M0c: báo cáo mọi seed; Δ = trung bình qua seed; CI lấy từ seed có PAIR_F1 dev trung vị.
  * Ngưỡng thực tiễn: 5 điểm PAIR_F1 [cần GV xác nhận].
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import propara_core as pc            # noqa: E402
import propara_metrics as pm         # noqa: E402
import propara_llm as pl             # noqa: E402
import propara_robust as pr          # noqa: E402

E2E_QWEN = ["QWEN_DIRECT", "QWEN_FEWSHOT", "QWEN_STRUCTURED", "QWEN_CONSTRAINED"]
THRESH = 0.05


def load_jsonl(p: Path):
    return {json.loads(l)["paragraph_id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def clean(pred):
    return pl.as_eval_prediction({k: v for k, v in pred.items() if k != "diag"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--k1"); ap.add_argument("--k2"); ap.add_argument("--k3")
    ap.add_argument("--out", required=True); ap.add_argument("--n_boot", type=int, default=10000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = Path(a.data)
    raws = {sp: pc.load_jsonl(str(D / f"grids.v1.{sp}.json")) for sp in ("train", "dev", "test")}
    topics = pc.load_topics_from_naacl_grid(str(D / "gold-full-grids.v3.tsv"))
    recs = {sp: [pc.build_record(r, sp, topics.get(str(r["para_id"]))) for r in raws[sp]] for sp in raws}
    nonce = pr.nonce_records({"dev": raws["dev"], "test": raws["test"]}, topics)
    R = {"rows": [], "notes": []}

    def score(name, preds, rs, split, src):
        res = pm.evaluate(preds, rs)
        ci = pm.bootstrap_ci_pair_f1(res, "all", n_boot=a.n_boot)
        row = pm.summary_row(name, res, {"split": split, "source": src, "PAIR_F1_CI95": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"})
        R["rows"].append(row)
        return res

    # --- K1: Qwen dev → chọn hệ thống Qwen chính
    qwen_dev, best_qwen = {}, None
    if a.k1:
        K1 = Path(a.k1)
        for m in E2E_QWEN + ["ORACLE_PREV_STATE"]:
            p = K1 / "predictions" / f"dev_{m}.jsonl"
            if p.exists():
                qwen_dev[m] = score(m, {k: clean(v) for k, v in load_jsonl(p).items()}, recs["dev"], "dev", "K1")
        cand = {m: r["PAIR@all"]["F1"] for m, r in qwen_dev.items() if m in E2E_QWEN}
        best_qwen = max(cand, key=cand.get) if cand else None
        R["best_qwen_by_dev"] = best_qwen
    # --- K2: mô hình nhóm dev + test kín
    team = {}
    if a.k2:
        K2 = Path(a.k2)
        summ = json.loads((K2 / "summary_05.json").read_text(encoding="utf-8"))
        R["k2_cv"] = summ.get("cv"); R["k2_cv_bootstrap"] = summ.get("cv_bootstrap"); R["k2_pp3"] = summ.get("pp3")
        R["k2_ablation_dev"] = summ.get("ablation_dev"); R["k2_nonce_dev"] = summ.get("nonce_dev"); R["k2_pp2"] = summ.get("pp2_predicted")
        for p in sorted((K2 / "sealed_test_predictions").glob("test_*_seed*.jsonl")):
            stem = p.stem[len("test_"):]
            is_nonce = stem.startswith("nonce_")
            stem = stem[len("nonce_"):] if is_nonce else stem
            name, seed = stem.rsplit("_seed", 1)
            rs = nonce["test"] if is_nonce else recs["test"]
            res = score(f"{name}|seed{seed}", load_jsonl(p), rs, "test_nonce" if is_nonce else "test", "K2 (kín → chấm 1 lần)")
            team[(name, int(seed), is_nonce)] = res
    # --- K3: Qwen test kín
    qwen_test = {}
    if a.k3:
        K3 = Path(a.k3)
        for p in sorted((K3 / "sealed_predictions").glob("*.jsonl")):
            is_nonce = p.stem.startswith("nonce_test_")
            m = p.stem.replace("nonce_test_", "").replace("test_", "")
            rs = nonce["test"] if is_nonce else recs["test"]
            qwen_test[(m, is_nonce)] = score(m, {k: clean(v) for k, v in load_jsonl(p).items()}, rs,
                                             "test_nonce" if is_nonce else "test", "K3 (kín → chấm 1 lần)")
    # --- So sánh chính trên test: Qwen chính vs M3, M3 vs M0c (paired bootstrap)
    comp = {}
    def seeds_of(name, nonce_flag=False):
        return sorted(s for (n, s, nf) in team if n == name and nf == nonce_flag)
    if best_qwen and (best_qwen, False) in qwen_test and seeds_of("CRF_FULL"):
        for sl in ("all", "any_implicit", "src_implicit", "tgt_implicit"):
            ds = [pm.paired_bootstrap_pair_f1(qwen_test[(best_qwen, False)], team[("CRF_FULL", s, False)], sl, n_boot=a.n_boot)
                  for s in seeds_of("CRF_FULL")]
            comp[f"QWEN[{best_qwen}]-M3@{sl}"] = {"delta_mean": statistics.mean(d["delta"] for d in ds), "per_seed": ds}
    if seeds_of("CRF_FULL") and seeds_of("M0c"):
        for sl in ("all", "any_implicit"):
            ds = [pm.paired_bootstrap_pair_f1(team[("CRF_FULL", s, False)], team[("M0c", s, False)], sl, n_boot=a.n_boot)
                  for s in seeds_of("CRF_FULL") if ("M0c", s, False) in team]
            comp[f"M3-M0c@{sl}"] = {"delta_mean": statistics.mean(d["delta"] for d in ds), "per_seed": ds}
    # CGLI (B2): đối chứng học máy mạnh, dự đoán test công bố của tác giả
    cg_path = HERE.parent / "vendor" / "cgli_propara_test_outputs.tsv"
    if cg_path.exists():
        rows = [tuple(x.strip() if isinstance(x, str) else x for x in r) for r in pc.load_action_file(str(cg_path))]
        cgli = score("CGLI_2022", {r["paragraph_id"]: pm.action_rows_to_prediction(rows, r) for r in recs["test"]}, recs["test"],
                     "test", "công bố (Mayer123/CGLI)")
        for sl in ("all", "any_implicit", "src_implicit", "tgt_implicit"):
            if seeds_of("CRF_FULL"):
                ds = [pm.paired_bootstrap_pair_f1(team[("CRF_FULL", s, False)], cgli, sl, n_boot=a.n_boot) for s in seeds_of("CRF_FULL")]
                comp[f"M3-CGLI@{sl}"] = {"delta_mean": statistics.mean(d["delta"] for d in ds), "per_seed": ds}
            if best_qwen and (best_qwen, False) in qwen_test:
                d = pm.paired_bootstrap_pair_f1(qwen_test[(best_qwen, False)], cgli, sl, n_boot=a.n_boot)
                comp[f"QWEN[{best_qwen}]-CGLI@{sl}"] = {"delta_mean": d["delta"], "per_seed": [d]}
    R["comparisons_test"] = comp
    # --- Áp quy tắc §7 (mô tả từng điều kiện; kết luận cuối do người viết báo cáo xác nhận)
    cv_b = R.get("k2_cv_bootstrap") or {}
    cv_imp = next((v for k, v in cv_b.items() if k.endswith("@any_implicit")), None)
    q_all = comp.get(f"QWEN[{best_qwen}]-M3@all"); q_imp = comp.get(f"QWEN[{best_qwen}]-M3@any_implicit")
    def sig_pos(d):  # mọi seed CI > 0
        return d is not None and all(x["ci_low"] > 0 for x in d["per_seed"])
    cond = {
        "KEEP_a: M3 > M0c có ý nghĩa trên lát ngầm (CV)": bool(cv_imp and cv_imp["ci_low"] > 0),
        "STOP_1: Qwen ≥ M3 + 5đ trên PAIR_F1 test, CI>0 mọi seed": bool(q_all and q_all["delta_mean"] >= THRESH and sig_pos(q_all)),
        "STOP_2: Qwen ≥ M3 + 5đ trên lát ngầm test, CI>0 mọi seed": bool(q_imp and q_imp["delta_mean"] >= THRESH and sig_pos(q_imp)),
    }
    R["decision_conditions"] = cond
    if cond["KEEP_a: M3 > M0c có ý nghĩa trên lát ngầm (CV)"]:
        verdict = "KEEP (điều kiện KEEP_a đạt)"
    elif cond["STOP_1: Qwen ≥ M3 + 5đ trên PAIR_F1 test, CI>0 mọi seed"] and cond["STOP_2: Qwen ≥ M3 + 5đ trên lát ngầm test, CI>0 mọi seed"]:
        verdict = "CONSIDER_STOP hoặc PIVOT — cần xét thêm ablation, khái quát (nonce), chi phí trước khi chốt"
    elif q_all and q_all["delta_mean"] > 0:
        verdict = "PIVOT/INCONCLUSIVE — Qwen hơn nhưng chưa đủ điều kiện STOP; xét lỗi hệ thống và chi phí"
    else:
        verdict = "INCONCLUSIVE/KEEP — không có bằng chứng Qwen vượt trội rõ"
    R["verdict_auto"] = verdict
    (out / "final_results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    cols = ["split", "system", "source", "PAIR_F1", "PAIR_F1_CI95", "PAIR_F1@src_imp", "PAIR_F1@tgt_imp", "PAIR_F1@any_imp",
            "IMP_ACC_DESTROY", "IMP_ACC_CREATE", "VIOL_per_entity"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in R["rows"]:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (out / "final_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print(json.dumps({"best_qwen": best_qwen, "conditions": cond, "verdict_auto": verdict}, ensure_ascii=False, indent=1))
    for k, v in comp.items():
        print(k, round(v["delta_mean"], 3), [(round(x["ci_low"], 3), round(x["ci_high"], 3)) for x in v["per_seed"]])


if __name__ == "__main__":
    main()
