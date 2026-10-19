"""Sinh notebooks/04_qwen7b_test_locked.ipynb (lượt K3): Qwen trên TEST + biến thể NONCE, cấu hình đã khoá ở K1.
Tái dùng NGUYÊN VĂN các ô môi trường / nạp mã / nạp mô hình / hàm sinh của notebook 03 (đã kiểm thử).
python tools/build_nb04.py <dự_án>   (phải chạy build_nb03.py trước)"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common as C  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
nb03 = nbf.read(ROOT / "notebooks" / "03_qwen7b_dev.ipynb", as_version=4)


def cell_after(header: str) -> str:
    for k, c in enumerate(nb03.cells):
        if c.cell_type == "markdown" and c.source.startswith(header):
            return nb03.cells[k + 1].source
    raise KeyError(header)


ENV_CELL = cell_after("## 2.")
MOD_CELL = cell_after("## 3.").replace("import propara_core_tests as pct, propara_metrics_tests as pmt, propara_llm_tests as plt_",
                                       "import propara_core_tests as pct, propara_metrics_tests as pmt, propara_llm_tests as plt_\nimport propara_robust as pr")
SRC = ROOT / "src"
MOD_CELL = MOD_CELL.replace("MODULES = {}", "MODULES = {}\nMODULES['propara_robust.py'] = r'''\n" + (SRC / "propara_robust.py").read_text(encoding="utf-8") + "'''", 1)
LOAD_MODEL_CELL = cell_after("## 5.")
GEN_CELL = cell_after("## 6.")

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 04: Qwen2.5-7B-Instruct trên TEST (cấu hình đã khoá) + biến thể đổi tên NONCE · lượt Kaggle **K3**

**Cấu hình ĐÃ KHOÁ sau K1 (19/10):** prompt `prompt_v1` giữ nguyên; **Qwen chính = `QWEN_FEWSHOT`** (PAIR_F1 dev 0,377, cao nhất trong các chế độ đầu-cuối). Thêm: Qwen chính chạy cả **train** (trừ 2 đoạn few-shot) để so với M3 trong CV 488 đoạn; hậu xử lý `QWEN_CONSTRAINED_V2` (sửa tồn tại ban đầu trước).
Thời gian **ước tính** từ số đo K1: test 4 chế độ ≈ 45–55 phút, NONCE ≈ 25 phút, train ≈ 60 phút → tổng khoảng 2–2,5 giờ.

- Test: chạy mọi chế độ đã khoá; đầu ra được **lưu kín, KHÔNG chấm trong notebook**. Trợ lý chấm một lần trên CPU sau khi khoá.
- NONCE (đổi tên thực thể bằng từ vô nghĩa, nhãn giữ nguyên): dev được chấm công khai; test lưu kín.
- Mã môi trường / nạp mô hình / sinh **giống hệt notebook 03**.

Cài đặt: **GPU T4 x2**, **Internet On**. Chạy bằng **Save & Run All (Commit)**. Gửi lại: khối `=== GỬI LẠI ===` + file zip `qwen7b_test_v1.zip`.
""")
md("## 1. Cấu hình (khoá)")
code(C.cfg_header("qwen7b_test_v1", extra=r'''
MODEL_ID       = os.environ.get("PROPARA_LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MODEL_REVISION = os.environ.get("PROPARA_LLM_REVISION", "a09a35458c702b33eeacc393d103063234e8bc28")
EXPECT_7B      = os.environ.get("PROPARA_EXPECT_7B", "1") == "1"
# CẤU HÌNH ĐÃ KHOÁ sau K1 (19/10): prompt_v1 giữ nguyên; Qwen chính = QWEN_FEWSHOT (PAIR_F1 dev cao nhất, quy tắc protocol v1.2 (e))
PRIMARY_MODE   = "QWEN_FEWSHOT"
TEST_MODES     = ["QWEN_DIRECT", "QWEN_FEWSHOT", "QWEN_STRUCTURED", "ORACLE_PREV_STATE"]   # khoá theo protocol §6
NONCE_MODES    = ["QWEN_DIRECT", "QWEN_FEWSHOT"]
RUN_TRAIN_FOR_CV = os.environ.get("PROPARA_RUN_TRAIN", "1") == "1"   # Qwen chính chạy cả train (trừ 2 đoạn few-shot) để so với M3 trong CV 488 đoạn
LIMIT_PARAS    = int(os.environ.get("PROPARA_LIMIT_PARAS", "0"))
N_BOOT         = 10000
'''))
md("## 2. Môi trường + GPU")
code(ENV_CELL)
md("## 3. Nạp mã nhúng + kiểm thử")
code(MOD_CELL)
md("## 4. Dữ liệu: test + NONCE (dev, test)")
code(C.LOAD_SRC + r'''
FEWSHOT = pl.select_fewshot(BY_SPLIT["train"], k=2)
_fs_ids = {r["paragraph_id"] for r in FEWSHOT}
_raws = {sp: pc.load_jsonl(str(RAW[f"grids.v1.{sp}.json"])) for sp in ("dev", "test")}
NONCE = pr.nonce_records(_raws, TOPICS)
TARGETS = {"test": BY_SPLIT["test"], "nonce_dev": NONCE["dev"], "nonce_test": NONCE["test"]}
if RUN_TRAIN_FOR_CV:
    TARGETS["train"] = [r for r in BY_SPLIT["train"] if r["paragraph_id"] not in _fs_ids]
MODES_BY_TARGET = {"test": TEST_MODES, "nonce_dev": NONCE_MODES, "nonce_test": NONCE_MODES, "train": [PRIMARY_MODE]}
if LIMIT_PARAS:
    TARGETS = {k: v[:LIMIT_PARAS] for k, v in TARGETS.items()}
print({k: len(v) for k, v in TARGETS.items()}, "| few-shot:", sorted(_fs_ids))
SMOKE_RECS = [r for r in BY_SPLIT["train"] if r["paragraph_id"] not in _fs_ids][:1]
''')
md("## 5. Nạp mô hình (giống notebook 03)")
code(LOAD_MODEL_CELL)
md("## 6. Hàm sinh (giống notebook 03)")
code(GEN_CELL)
md("## 7. Smoke 1 đoạn train → chạy test + NONCE")
code(r'''
raw = generate_all(items_for(SMOKE_RECS, "QWEN_DIRECT"), "smoke_QWEN_DIRECT")
_p, _d = predictions_for(SMOKE_RECS, "QWEN_DIRECT", raw)
SMOKE_OK = all(d["json_status"] in ("valid", "repaired") for d in _d.values())
print("smoke:", {k: v["json_status"] for k, v in _d.items()})
if not SMOKE_OK and os.environ.get("PROPARA_RUN_FULL", "auto") != "force":
    raise RuntimeError("Smoke không đạt — dừng để không tốn GPU. Gửi output cho trợ lý.")
PREDS, DIAGS, SEC = {}, {}, {}
for tgt, recs in TARGETS.items():
    for mode in MODES_BY_TARGET[tgt]:
        t0 = time.time()
        raw = generate_all(items_for(recs, mode), f"{tgt}_{mode}")
        PREDS[(tgt, mode)], DIAGS[(tgt, mode)] = predictions_for(recs, mode, raw)
        SEC[f"{tgt}|{mode}"] = round(time.time() - t0, 1)
        print(f"{tgt} {mode}: {SEC[f'{tgt}|{mode}']}s")
    # hậu xử lý (không gọi lại mô hình, không dùng vàng): v1 = luật gốc; v2 = sửa tồn tại ban đầu trước (đăng ký sau K1)
    for base in ("QWEN_DIRECT", PRIMARY_MODE):
        if (tgt, base) in PREDS:
            suf = "" if base == "QWEN_DIRECT" else "_FS"
            PREDS[(tgt, "QWEN_CONSTRAINED" + suf)] = {pid: pl.apply_existence_constraints(p)[0] for pid, p in PREDS[(tgt, base)].items()}
            PREDS[(tgt, "QWEN_CONSTRAINED_V2" + suf)] = {pid: pl.apply_existence_constraints_v2(p)[0] for pid, p in PREDS[(tgt, base)].items()}
''')
md("## 8. Chấm NONCE-dev (công khai); test và NONCE-test chỉ lưu kín")
code(r'''
NONCE_DEV_SCORES = {}
for (tgt, mode), preds in PREDS.items():
    if tgt == "nonce_dev":
        res = pm.evaluate({k: pl.as_eval_prediction(v) for k, v in preds.items()}, TARGETS["nonce_dev"])
        NONCE_DEV_SCORES[mode] = pm.summary_row(mode, res)
        r = NONCE_DEV_SCORES[mode]
        print(f"NONCE-dev {mode:18s} PAIR_F1={r['PAIR_F1']:.3f} @any_imp={r['PAIR_F1@any_imp']:.3f} IMP_D={r['IMP_ACC_DESTROY']:.3f} IMP_C={r['IMP_ACC_CREATE']:.3f}")
JSON_RATE = {f"{t}|{m}": sum(d.get("json_status") in ("valid", "repaired") for d in DIAGS[(t, m)].values()) / max(len(DIAGS[(t, m)]), 1)
             for (t, m) in DIAGS if m != "ORACLE_PREV_STATE"}
print("Tỉ lệ JSON hợp lệ:", {k: round(v, 3) for k, v in JSON_RATE.items()})
''')
md("## 9. Lưu + gửi lại")
code(r'''
def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
S = OUT / "sealed_predictions"; S.mkdir(exist_ok=True)
for (tgt, mode), preds in PREDS.items():
    d = S if tgt in ("test", "nonce_test") else OUT
    with open(d / f"{tgt}_{mode}.jsonl", "w", encoding="utf-8") as f:
        for pid, p in preds.items():
            f.write(json.dumps(dict(p, diag=DIAGS.get((tgt, mode), {}).get(pid)), ensure_ascii=False) + "\n")
jdump({"nonce_dev": {r["paragraph_id"]: {"entities": r["entities"], "sentences": r["sentences"]} for r in TARGETS["nonce_dev"]},
       "nonce_test": {r["paragraph_id"]: {"entities": r["entities"], "sentences": r["sentences"]} for r in TARGETS["nonce_test"]}},
      OUT / "nonce_texts.json")
SUMMARY = {"run_tag": RUN_TAG, "model": MODEL_ORIGIN, "quant": QUANT, "env": ENV, "sec": SEC, "json_rate": JSON_RATE,
           "nonce_dev": NONCE_DEV_SCORES, "prompt_version": pl.PROMPT_VERSION, "fewshot_ids": sorted(_fs_ids),
           "runtime_sec": round(time.time() - T0, 1), "gen_log_tail": GEN_LOG[-10:]}
jdump(SUMMARY, OUT / "summary_04.json")
ZIP = WORK_ROOT / f"{RUN_TAG}.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "code" not in p.parts and p.parent.name != "raw":
            z.write(p, Path(RUN_TAG) / p.relative_to(OUT))
print("=== GỬI LẠI ===")
print(f"notebook=04_qwen7b_test model={MODEL_ORIGIN} quant={QUANT} runtime={SUMMARY['runtime_sec']}s")
print("json_rate:", {k: round(v, 3) for k, v in JSON_RATE.items()})
for m, r in NONCE_DEV_SCORES.items():
    print(f"  NONCE-dev {m}: PAIR_F1={r['PAIR_F1']:.3f} @any_imp={r['PAIR_F1@any_imp']:.3f}")
print("Đã lưu kín:", sorted(p.name for p in S.glob("*.jsonl")))
print("=== HẾT ===  (gửi file", ZIP.name, "cho trợ lý)")
''')

out = C.save_notebook(cells, ROOT / "notebooks" / "04_qwen7b_test_locked.ipynb", gpu=True)
print("Đã ghi", out, "| số ô:", len(cells))
