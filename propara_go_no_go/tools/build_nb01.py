"""Sinh notebooks/01_metrics_heuristics.ipynb.  Cách dùng: python tools/build_nb01.py <thư_mục_dự_án>"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common as C  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODS = {n: (SRC / n).read_text(encoding="utf-8") for n in
        ["propara_core.py", "propara_core_tests.py", "propara_metrics.py", "propara_metrics_tests.py", "propara_heuristics.py"]}
VENDOR = (ROOT / "vendor" / "aristo_propara_evaluator.zip").read_bytes()
CGLI = (ROOT / "vendor" / "cgli_propara_test_outputs.tsv").read_bytes()   # dự đoán test công bố của CGLI (Mayer123/CGLI@9c910de, GPL-3.0; chỉ dùng làm dữ liệu)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 01: Bộ chấm chung + heuristic mốc · `propara_metrics_v1`

**Vị trí trong pipeline:** Giai đoạn 2 (D2), **Cửa nghiệm thu 1**. Notebook Qwen chỉ được chạy sau khi notebook này đạt PASS.

**Notebook tự chứa:** tự tải ProPara theo commit ghim và tự dựng lại dữ liệu chuẩn như notebook 00 (cùng `propara_core`). **Không cần** Add Input output của notebook 00.

**Nội dung:**
1. Nhúng **mã evaluator chính thức** `allenai/aristo-leaderboard/propara/evaluator` (nguyên văn, Apache-2.0) + bộ chấm của đề tài (`propara_metrics`).
2. Kiểm thử: 16 (core) + 17 (bộ chấm) kiểm thử tự viết + 17 kiểm thử gốc của evaluator + hồi quy trên dự đoán **ProStruct (0,545)** và **ProGlobal (0,519)** đã công bố.
3. Kiểm tra vàng → điểm tuyệt đối; bản sao conversion khớp mã gốc.
4. Heuristic điểm quyết định (Bảng 3 báo cáo): ngẫu nhiên / ngẫu nhiên-không-nhắc / nhắc-gần-nhất.
5. Hệ thống tham chiếu đầu-cuối: ALL-NONE, **heuristic đầu-cuối B1-E2E** (không dùng vàng), ProStruct 2018, ProGlobal 2018, **CGLI 2022** (dự đoán công bố, chỉ có trên test) + 2 phép **chẩn đoán dùng vàng**.
6. Bootstrap theo đoạn cho khoảng tin cậy.

| Mục | Giá trị |
|---|---|
| Accelerator | **None (CPU)** |
| Internet | On (hoặc Add Input dataset `propara-raw-v1`) |
| Output | `/kaggle/working/propara_metrics_v1/` + `.zip` |

**Thước đo** (xem `docs/experiment_protocol.md`): `PAIR_F1` (chính, micro, khớp chính xác (câu, nguồn, đích)); lát `@src_imp/@tgt_imp/@any_imp` theo cờ nhắc tính từ văn bản; `Q3_OFFICIAL` = điểm “Conversions” của evaluator gốc.

**Lưu ý test:** ProStruct/ProGlobal là **hệ thống tham chiếu cố định đã công bố**; xem điểm test của chúng không điều chỉnh bất cứ thứ gì của đề tài. Mọi heuristic ở đây **không có tham số học**.
""")

md("## 1. Cấu hình")
code(C.cfg_header("propara_metrics_v1", extra='N_BOOT = 10000\n'))
md("## 2. Môi trường")
code(C.ENV_SRC)
md("## 3. Nạp mã (nhúng): `propara_core`, `propara_metrics`, `propara_heuristics`, evaluator chính thức")
code(C.embed_modules_cell(MODS, {"aristo_propara_evaluator.zip": VENDOR, "cgli_propara_test_outputs.tsv": CGLI}) + r'''
import propara_core as pc, propara_core_tests as pct, propara_metrics as pm, propara_metrics_tests as pmt, propara_heuristics as ph
EVAL_DIR = str(MOD_DIR / "aristo_propara_evaluator")
print(pc.CORE_VERSION, pm.METRICS_VERSION, ph.HEUR_VERSION, "| evaluator:", EVAL_DIR)
''')

md("## 4. Kiểm thử đơn vị (tự viết + mã gốc của evaluator)")
code(r'''
import unittest
print("— propara_core"); n1, f1 = pct.run_all(verbose=False)
print("— propara_metrics"); n2, f2 = pmt.run_all(verbose=True)
# Kiểm thử gốc của evaluator (dùng assertEquals đã bị bỏ ở Python 3.12 → thêm bí danh, không sửa mã gốc)
unittest.TestCase.assertEquals = unittest.TestCase.assertEqual
_cwd = os.getcwd(); os.chdir(EVAL_DIR); sys.path.insert(0, EVAL_DIR)
try:
    _r = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w")).run(
        unittest.defaultTestLoader.discover(".", pattern="test_*.py", top_level_dir="."))
finally:
    os.chdir(_cwd)
n3, f3 = _r.testsRun, [str(t) for t, _ in _r.failures + _r.errors]
print(f"core {n1 - len(f1)}/{n1} | metrics {n2 - len(f2)}/{n2} | evaluator gốc {n3 - len(f3)}/{n3}")
if f1 or f2 or f3:
    raise RuntimeError(f"Kiểm thử thất bại: {f1} {f2} {f3}")
''')

md("## 5. Hồi quy evaluator gốc trên các tệp kiểm thử chính thức (điểm đã công bố trong `test.sh`)")
code(r'''
REG_EXPECT = {1: {"precision": 0.743, "recall": 0.43, "f1": 0.545},   # ProStruct, test
              2: {"precision": 1.0, "recall": 1.0, "f1": 1.0},
              3: {"precision": 0.833, "recall": 0.583, "f1": 0.686},
              7: {"precision": 0.617, "recall": 0.448, "f1": 0.519}}  # ProGlobal, test
REG = {}
for i, exp in REG_EXPECT.items():
    got = pm.official_q3(EVAL_DIR, f"{EVAL_DIR}/testfiles-{i}/answers.tsv", f"{EVAL_DIR}/testfiles-{i}/predictions.tsv")["overall_rounded"]
    REG[i] = (got == exp, got)
    print(f"testfiles-{i}: {got}  kỳ vọng {exp}  {'✓' if got == exp else '✗'}")
for i in (5, 6):   # phải bị từ chối
    try:
        pm.official_q3(EVAL_DIR, f"{EVAL_DIR}/testfiles-{i}/answers.tsv", f"{EVAL_DIR}/testfiles-{i}/predictions.tsv")
        REG[i] = (False, "không bị từ chối")
    except pm.PredictionError as e:
        REG[i] = (True, "bị từ chối đúng")
    print(f"testfiles-{i}: {REG[i][1]}")
''')

md("## 6. Dữ liệu (tự tải + dựng lại, tự kiểm tra 488/3302/1915 và 533 cặp)")
code(C.LOAD_SRC)

md("## 7. Vàng → điểm tuyệt đối; bản sao conversion khớp mã gốc")
code(r'''
TMP = OUT / "action_files"; TMP.mkdir(exist_ok=True)
ANS = {sp: str(RAW[f"leaderboard_answers_{sp}.tsv"]) for sp in SPLITS}

def score_system(preds, sp, name):
    """Chấm một hệ thống trên một tập: PAIR (bộ chấm đề tài) + Q3_OFFICIAL (evaluator gốc)."""
    recs = BY_SPLIT[sp]
    res = pm.evaluate(preds, recs)
    path = TMP / f"{name}_{sp}.tsv"
    pm.write_action_file(str(path), [row for r in recs for row in pm.to_official_action_rows(preds[r["paragraph_id"]], r)])
    res["Q3_OFFICIAL"] = pm.official_q3(EVAL_DIR, ANS[sp], str(path))
    return res

GOLD_CHECK = {}
for sp in SPLITS:
    res = score_system({r["paragraph_id"]: pm.gold_as_prediction(r) for r in BY_SPLIT[sp]}, sp, "GOLD")
    GOLD_CHECK[sp] = {"PAIR_F1": res["PAIR@all"]["F1"], "PAIR_LOC_F1": res["PAIR_LOC"]["F1"],
                      "Q3_conv_F1": res["Q3_OFFICIAL"]["conversions"]["F1"], "Q3_overall": res["Q3_OFFICIAL"]["overall_rounded"],
                      "VIOL": res["VIOLATION_RATE_per_entity"]}
    print(sp, GOLD_CHECK[sp])

# Bản sao official_q3_conversions (core) so với ProcessSummary của mã gốc
sys.path.insert(0, EVAL_DIR)
import importlib
_AF = importlib.import_module("process").ActionFile
REPLICA_MISMATCH = {}
for sp in SPLITS:
    summ = _AF.from_file(ANS[sp]).summarize()
    bad = []
    for pid, s in summ.items():
        off = sorted((int(c.step_id), frozenset(c.destroyed.split(" AND ")), frozenset(c.created.split(" AND ")),
                      frozenset(c.locations.split(" AND "))) for c in s.conversions)
        mine = sorted((c["step_id"], frozenset(c["destroyed"].split(" AND ")), frozenset(c["created"].split(" AND ")),
                       frozenset(c["locations"])) for c in pc.official_q3_conversions(BY_PID[str(pid)]))
        if off != mine:
            bad.append(str(pid))
    REPLICA_MISMATCH[sp] = bad
    print(sp, "đoạn lệch giữa bản sao và mã gốc:", bad)
DUP_PIDS = sorted({r["paragraph_id"] for r in RECS if any("trùng tên" in m for m in pc.validate_record(r))})
OFFICIAL_GOLD_COUNT = {sp: sum(len(s.conversions) for s in _AF.from_file(ANS[sp]).summarize().values()) for sp in SPLITS}
print("conversion vàng theo evaluator gốc:", OFFICIAL_GOLD_COUNT, "| đoạn trùng tên:", DUP_PIDS)
''')

md("## 8. Heuristic trên điểm quyết định ngầm (tái hiện Bảng 3 báo cáo)")
code(r'''
DPS = [d for r in RECS for d in pc.decision_points(r)]
DPT = ph.decision_point_table(DPS, BY_PID, seeds=20, seed0=0)
REPORT_T3 = {"random": {"DESTROY": 51, "CREATE": 40}, "random_unmentioned": {"DESTROY": 59, "CREATE": 42},
             "nearest_mention": {"DESTROY": 74, "CREATE": 33}}
print(f"{'phương pháp':20s} {'nhãn':8s} {'n':>3s} {'kỳ vọng':>8s} {'TB 20 seed':>10s} {'báo cáo':>8s}")
for m, v in DPT.items():
    for lab, x in v.items():
        print(f"{m:20s} {lab:8s} {x['n']:3d} {100*x['expected']:7.1f}% {100*x['sim_mean_20seeds']:9.1f}% {REPORT_T3[m][lab]:7d}%")
# Theo tập (cho biết test có bao nhiêu điểm → độ bất định)
DPT_SPLIT = {}
for sp in SPLITS:
    sub = [d for d in DPS if d["split"] == sp]
    DPT_SPLIT[sp] = {m: {lab: round(100 * statistics.mean([ph.expected_acc(d, BY_PID[d["paragraph_id"]], m) for d in sub if d["label"] == lab]), 1)
                         if any(d["label"] == lab for d in sub) else None for lab in ("DESTROY", "CREATE")}
                     for m in REPORT_T3}
    print(sp, {lab: sum(d["label"] == lab for d in sub) for lab in ("DESTROY", "CREATE")}, DPT_SPLIT[sp])
''')

md("## 9. Hệ thống tham chiếu đầu-cuối (dev + test)")
code(r'''
SYSTEMS = {}   # (name, split) -> kết quả
def add(name, sp, preds, kind):
    res = score_system(preds, sp, name)
    res["_kind"] = kind
    SYSTEMS[(name, sp)] = res

rng = random.Random(SEED)
for sp in ("dev", "test"):
    add("ALL_NONE", sp, {r["paragraph_id"]: ph.all_none(r) for r in BY_SPLIT[sp]}, "baseline")
    add("DIAG_ORACLE_MENTIONED_ONLY", sp, {r["paragraph_id"]: ph.oracle_mentioned_only(r) for r in BY_SPLIT[sp]}, "diagnostic(gold)")
    add("DIAG_ORACLE_MENTIONED+NEAREST", sp,
        {r["paragraph_id"]: ph.oracle_mentioned_plus_nearest(r, [d for d in DPS if d["paragraph_id"] == r["paragraph_id"]], rng)
         for r in BY_SPLIT[sp]}, "diagnostic(gold)")

    add("B1_E2E_RULE+NEAREST", sp, {r["paragraph_id"]: ph.rule_nearest_e2e(r) for r in BY_SPLIT[sp]}, "baseline (heuristic đầu-cuối, không dùng vàng)")
    add("B1_E2E_RULE_ONLY", sp, {r["paragraph_id"]: ph.rule_only_e2e(r) for r in BY_SPLIT[sp]}, "baseline (heuristic, bỏ suy ngầm)")

# Dự đoán công bố: ProStruct (EMNLP18), ProGlobal (NAACL18) — từ bộ kiểm thử evaluator; CGLI (COLING22) — từ repo tác giả. Chỉ có trên test
_cg = [tuple(x.strip() if isinstance(x, str) else x for x in r) for r in pc.load_action_file(str(MOD_DIR / "cgli_propara_test_outputs.tsv"))]   # bỏ khoảng trắng thừa ở vị trí
pm.write_action_file(str(MOD_DIR / "cgli_clean.tsv"), _cg)
PUBLISHED = {"ProStruct_2018": f"{EVAL_DIR}/testfiles-1/predictions.tsv", "ProGlobal_2018": f"{EVAL_DIR}/testfiles-7/predictions.tsv",
             "CGLI_2022": str(MOD_DIR / "cgli_clean.tsv")}
PUBLISHED_Q3_ORIGINAL = {}
for name, path in PUBLISHED.items():
    rows = pc.load_action_file(path)
    preds = {r["paragraph_id"]: pm.action_rows_to_prediction(rows, r) for r in BY_SPLIT["test"]}
    add(name, "test", preds, "published")
    # Q3_OFFICIAL chấm trên FILE GỐC (không qua chuyển đổi) để đúng điểm công bố
    PUBLISHED_Q3_ORIGINAL[name] = pm.official_q3(EVAL_DIR, ANS["test"], path)
    SYSTEMS[(name, "test")]["Q3_OFFICIAL_roundtrip"] = SYSTEMS[(name, "test")]["Q3_OFFICIAL"]["overall_rounded"]
    SYSTEMS[(name, "test")]["Q3_OFFICIAL"] = PUBLISHED_Q3_ORIGINAL[name]

ROWS = []
for (name, sp), res in SYSTEMS.items():
    ci = pm.bootstrap_ci_pair_f1(res, "all", n_boot=N_BOOT, seed=SEED)
    q = res["Q3_OFFICIAL"]
    ROWS.append(pm.summary_row(name, res, {"split": sp, "kind": res["_kind"],
                "PAIR_F1_CI95": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                "Q3_conv_P": q["conversions"]["P"], "Q3_conv_R": q["conversions"]["R"], "Q3_conv_F1": q["conversions"]["F1"],
                "Q3_overall_F1": q["overall"]["F1"]}))
cols = ["split", "system", "kind", "PAIR_F1", "PAIR_F1_CI95", "PAIR_F1@src_imp", "PAIR_F1@tgt_imp", "PAIR_F1@any_imp",
        "F1_CREATE", "F1_DESTROY", "IMP_ACC_CREATE", "IMP_ACC_DESTROY", "VIOL_per_entity", "Q3_conv_F1", "Q3_overall_F1"]
print("  ".join(f"{c[:14]:>14s}" for c in cols))
for r in sorted(ROWS, key=lambda r: (r["split"], r["kind"], r["system"])):
    print("  ".join(f"{(f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c]))[:14]:>14s}" for c in cols))
''')

md("## 10. Ví dụ so sánh có kiểm định: ProStruct − ProGlobal trên test (paired bootstrap theo đoạn)")
code(r'''
BOOT = {}
for sl in ("all", "any_implicit"):
    BOOT[sl] = pm.paired_bootstrap_pair_f1(SYSTEMS[("ProStruct_2018", "test")], SYSTEMS[("ProGlobal_2018", "test")],
                                          slice_name=sl, n_boot=N_BOOT, seed=SEED)
    b = BOOT[sl]
    print(f"Δ PAIR_F1@{sl} = {b['delta']:+.3f}  CI95 [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]  (n={b['n_paragraphs']} đoạn)")
''')

md("## 11. Cổng nghiệm thu 1")
code(r'''
CHECKS = []
def check(cid, desc, expected, observed, ok, severity="FAIL", note=""):
    CHECKS.append({"id": cid, "check": desc, "expected": str(expected), "observed": str(observed),
                   "status": "PASS" if ok else severity, "note": note})

check("M01", "Kiểm thử core / metrics / evaluator gốc", f"{n1}/{n2}/{n3}", f"{n1-len(f1)}/{n2-len(f2)}/{n3-len(f3)}", not (f1 or f2 or f3))
check("M02", "Hồi quy evaluator gốc (testfiles 1,2,3,7 đúng điểm; 5,6 bị từ chối)", "6/6", f"{sum(v[0] for v in REG.values())}/6",
      all(v[0] for v in REG.values()))
check("M03", "Dữ liệu tải về khớp SHA-256 ghim", "8/8", f"{sum(v['match'] for v in CHECKSUM.values())}/8",
      all(v["match"] for v in CHECKSUM.values()), "WARN")
check("M04", "Vàng → PAIR_F1 = 1 (train/dev/test)", "1/1/1", "/".join(f"{GOLD_CHECK[sp]['PAIR_F1']:.3f}" for sp in SPLITS),
      all(GOLD_CHECK[sp]["PAIR_F1"] == 1.0 for sp in SPLITS))
check("M05", "Vàng → Q3_OFFICIAL = 1 trên dev/test", "1.0/1.0",
      "/".join(str(GOLD_CHECK[sp]["Q3_overall"]["f1"]) for sp in ("dev", "test")),
      all(GOLD_CHECK[sp]["Q3_overall"]["f1"] == 1.0 for sp in ("dev", "test")))
check("M06", "Vàng → Q3_OFFICIAL train < 1 chỉ do thực thể trùng tên", "~0.998", GOLD_CHECK["train"]["Q3_overall"]["f1"],
      GOLD_CHECK["train"]["Q3_overall"]["f1"] >= 0.99, "WARN", f"đoạn trùng tên: {DUP_PIDS}")
check("M07", "Bản sao conversion khớp mã gốc (dev/test)", "0/0", f"{len(REPLICA_MISMATCH['dev'])}/{len(REPLICA_MISMATCH['test'])}",
      not REPLICA_MISMATCH["dev"] and not REPLICA_MISMATCH["test"])
check("M08", "Bản sao lệch mã gốc trên train CHỈ ở đoạn trùng tên", f"⊆ {DUP_PIDS}", REPLICA_MISMATCH["train"],
      set(REPLICA_MISMATCH["train"]) <= set(DUP_PIDS))
check("M09", "Vàng không vi phạm ràng buộc tồn tại", 0, max(GOLD_CHECK[sp]["VIOL"] for sp in SPLITS),
      all(GOLD_CHECK[sp]["VIOL"] == 0 for sp in SPLITS))
for m, v in REPORT_T3.items():
    for lab, rep in v.items():
        obs = 100 * DPT[m][lab]["sim_mean_20seeds"]
        check(f"H-{m[:7]}-{lab[0]}", f"Bảng 3: {m} / {lab}", f"{rep}%", f"{obs:.1f}% (kỳ vọng {100*DPT[m][lab]['expected']:.1f}%)",
              abs(obs - rep) <= 2.0, "WARN", "dung sai ±2 điểm (cách xử lý hoà khác nhau)")
for name in ("ProStruct_2018", "ProGlobal_2018"):
    got = PUBLISHED_Q3_ORIGINAL[name]["overall_rounded"]["f1"]
    exp = REG_EXPECT[1 if "ProStruct" in name else 7]["f1"]
    check(f"P-{name[:9]}", f"{name}: Q3_OFFICIAL tổng trên file gốc", exp, got, got == exp)
check("P-CGLI", "CGLI_2022: Q3_OFFICIAL tổng (file dự đoán công bố của tác giả)", "ghi nhận", PUBLISHED_Q3_ORIGINAL["CGLI_2022"]["overall_rounded"]["f1"],
      True, "INFO", "chấm bằng evaluator gốc; chưa đối chiếu với số trong bài báo")
rt = SYSTEMS[("ProGlobal_2018", "test")]["Q3_OFFICIAL_roundtrip"]["f1"]
check("P-roundtrip", "ProGlobal: chuyển đổi định dạng không làm đổi điểm chính thức", 0.519, rt, rt == 0.519)
an = SYSTEMS[("ALL_NONE", "test")]
check("I01", "Q3_OFFICIAL Conversions của ALL_NONE (hệ thống rỗng) — mức sàn do trường hợp biên macro", "ghi nhận",
      f"{an['Q3_OFFICIAL']['conversions']['F1']:.3f} (test)", True, "INFO",
      "đoạn không có conversion vàng & không dự đoán được P=R=1 → lý do chọn PAIR_F1 làm thước đo chính")

n_fail = sum(c["status"] == "FAIL" for c in CHECKS)
n_warn = sum(c["status"] == "WARN" for c in CHECKS)
GATE_STATUS = "PASS" if n_fail == 0 else "FAIL"
w = max(len(c["check"]) for c in CHECKS)
for c in CHECKS:
    print(f"{c['id']:16s} {c['status']:5s} {c['check']:{w}s}  {c['expected']} → {c['observed']}")
print(f"\nFAIL={n_fail} WARN={n_warn} → GATE_STATUS (Cửa nghiệm thu 1) = {GATE_STATUS}")
''')

md("## 12. Lưu đầu ra")
code(r'''
def jdump(obj, path):
    def conv(o):
        if isinstance(o, Counter): return {f"{k}": v for k, v in o.items()}
        if isinstance(o, tuple): return list(o)
        if isinstance(o, float) and o != o: return None
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=conv), encoding="utf-8")

R = OUT / "results"; R.mkdir(exist_ok=True)
with open(R / "reference_systems.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
    w.writeheader(); w.writerows(ROWS)
for (name, sp), res in SYSTEMS.items():
    jdump({k: v for k, v in res.items() if k != "_per_paragraph"}, R / f"metrics_{name}_{sp}.json")
jdump({m: {lab: {k: v for k, v in x.items() if k != "expected_per_dp"} for lab, x in d.items()} for m, d in DPT.items()},
      R / "decision_point_heuristics.json")
SUMMARY = {"run_tag": RUN_TAG, "gate_status": GATE_STATUS, "n_fail": n_fail, "n_warn": n_warn, "env": ENV,
           "versions": {"core": pc.CORE_VERSION, "metrics": pm.METRICS_VERSION, "heuristics": ph.HEUR_VERSION},
           "raw_sources": RAW_SOURCE, "checks": CHECKS, "gold_check": GOLD_CHECK, "official_gold_conversions": OFFICIAL_GOLD_COUNT,
           "decision_point_table": {m: {lab: {"n": x["n"], "expected": x["expected"], "sim20": x["sim_mean_20seeds"]} for lab, x in d.items()} for m, d in DPT.items()},
           "decision_point_by_split": DPT_SPLIT, "reference_systems": ROWS, "bootstrap_prostruct_vs_proglobal": BOOT,
           "runtime_sec": round(time.time() - T0, 1)}
jdump(SUMMARY, OUT / "summary_01.json")
manifest = {p.relative_to(OUT).as_posix(): pc.sha256_file(str(p)) for p in sorted(OUT.rglob("*")) if p.is_file() and "raw" not in p.parts}
jdump(manifest, OUT / "manifest.json")
ZIP = WORK_ROOT / f"{RUN_TAG}.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "raw" not in p.parts:
            z.write(p, Path(RUN_TAG) / p.relative_to(OUT))
print("Đã lưu", OUT, "| zip", ZIP, f"{ZIP.stat().st_size/1e6:.2f} MB | {SUMMARY['runtime_sec']} s")
''')

md("## 13. Gửi lại")
code(r'''
print("=== GỬI LẠI ===")
print(f"notebook=01_metrics_heuristics GATE_STATUS={GATE_STATUS} FAIL={n_fail} WARN={n_warn} python={ENV['python']} on_kaggle={ENV['on_kaggle']}")
for c in CHECKS:
    if c["status"] != "PASS":
        print(f"  {c['id']} {c['status']}: {c['check']} | {c['expected']} → {c['observed']}")
for r in sorted(ROWS, key=lambda r: (r["split"], r["system"])):
    print(f"  {r['split']:4s} {r['system']:30s} PAIR_F1={r['PAIR_F1']:.3f} {r['PAIR_F1_CI95']} @any_imp={r['PAIR_F1@any_imp']:.3f} Q3conv={r['Q3_conv_F1']:.3f}")
print("=== HẾT ===")
if GATE_STATUS == "FAIL":
    raise RuntimeError("Cửa nghiệm thu 1 FAIL — xem mục 11.")
''')

out = C.save_notebook(cells, ROOT / "notebooks" / "01_metrics_heuristics.ipynb")
print("Đã ghi", out, "| số ô:", len(cells))
