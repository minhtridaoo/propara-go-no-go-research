"""Sinh notebooks/03_qwen7b_dev.ipynb (lượt Kaggle K1: smoke + dev, GPU).  python tools/build_nb03.py <dự_án>"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common as C  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODS = {n: (SRC / n).read_text(encoding="utf-8") for n in
        ["propara_core.py", "propara_core_tests.py", "propara_metrics.py", "propara_metrics_tests.py",
         "propara_heuristics.py", "propara_llm.py", "propara_llm_tests.py"]}
VENDOR = (ROOT / "vendor" / "aristo_propara_evaluator.zip").read_bytes()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 03: Qwen2.5-7B-Instruct trên DEV (smoke tự động → chạy đầy đủ) · lượt Kaggle **K1**

**Mục tiêu:** đo Qwen 7B (không huấn luyện thêm) trên tập **dev** theo 4 chế độ đã khoá trong `experiment_protocol.md`, chấm bằng **cùng bộ chấm** đã qua Cửa nghiệm thu 1 (notebook 01):

| Chế độ | Nội dung | Loại |
|---|---|---|
| `QWEN_DIRECT` (Q0) | zero-shot, xuất thay đổi + conversions | đầu-cuối |
| `QWEN_FEWSHOT` (Q1) | + 2 ví dụ cố định từ **train** | đầu-cuối |
| `QWEN_STRUCTURED` (Q2) | bảng đầy đủ tồn tại trước → nhãn → tồn tại sau → vị trí | đầu-cuối, đo trung gian |
| `QWEN_CONSTRAINED` (Q3) | Q0 + hậu xử lý luật tồn tại (không gọi lại mô hình) | đầu-cuối |
| `ORACLE_PREV_STATE` (Q4) | mỗi câu một lời gọi, cho trạng thái **vàng trước câu t** | **ORACLE, bảng riêng** |

**Không chạy test** trong notebook này. Test chạy một lần ở lượt K3, sau khi cấu hình được khoá trên dev.

### Cài đặt trên Kaggle
| Mục | Giá trị |
|---|---|
| Accelerator | **GPU T4 x2** (khuyến nghị: fp16 chia trên 2 GPU, không cần nén). Nếu chỉ có 1 GPU (T4/P100), notebook tự chuyển sang **4-bit** (bitsandbytes) |
| Internet | **On** (tải mô hình ~15 GB từ Hugging Face vào bộ nhớ tạm, **không** vào Output). Hoặc Add Input mô hình Qwen2.5-7B-Instruct từ *Kaggle Models* nếu có, notebook tự dò |
| Input | Không bắt buộc. Dữ liệu ProPara và bộ chấm được tải/nhúng sẵn |
| Output | `/kaggle/working/qwen7b_dev_v1/` + `.zip` (dự đoán thô, dự đoán chuẩn, điểm, cấu hình, nhật ký). **Không** chứa trọng số mô hình |

### Chạy
**Run All**. Notebook tự chạy *smoke test* trên 3 đoạn **train**:
- Nếu smoke đạt (mô hình nạp được, JSON hợp lệ, không hết bộ nhớ), notebook **tự chạy tiếp toàn bộ dev**.
- Nếu smoke không đạt, notebook **dừng** trước khi tốn thêm GPU và in lý do.

Mọi đầu ra thô được ghi ngay sau mỗi lô nên phiên bị ngắt thì chạy lại sẽ **tiếp tục** từ chỗ dừng (cùng phiên). Xong thì **Save Version → Save & Run All (Commit)**. Có thể chọn chạy nền (commit) ngay từ đầu để không phải giữ trình duyệt mở.

**Gửi lại:** khối `=== GỬI LẠI ===` ở ô cuối + file `qwen7b_dev_v1/summary_03.json`.
""")

md("## 1. Cấu hình")
code(C.cfg_header("qwen7b_dev_v1", extra=r'''
MODEL_ID       = os.environ.get("PROPARA_LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MODEL_REVISION = os.environ.get("PROPARA_LLM_REVISION", "a09a35458c702b33eeacc393d103063234e8bc28")  # ghim 2025-01-12
EXPECT_7B      = os.environ.get("PROPARA_EXPECT_7B", "1") == "1"   # chỉ tắt khi kiểm thử cục bộ bằng mô hình nhỏ
SPLIT          = "dev"                                          # KHÔNG đổi thành test ở notebook này
MODES_TO_RUN   = ["QWEN_DIRECT", "QWEN_FEWSHOT", "QWEN_STRUCTURED", "ORACLE_PREV_STATE"]
RUN_FULL       = os.environ.get("PROPARA_RUN_FULL", "auto")     # "auto": chạy dev khi smoke đạt; "no": chỉ smoke
SMOKE_N        = 3                                              # số đoạn TRAIN cho smoke
SMOKE_MIN_VALID = 2                                             # tối thiểu số đầu ra JSON hợp lệ / SMOKE_N (chế độ Q0)
LIMIT_PARAS    = int(os.environ.get("PROPARA_LIMIT_PARAS", "0"))  # 0 = toàn bộ dev; >0 chỉ để kiểm thử cục bộ
N_BOOT         = 10000
assert SPLIT == "dev", "Notebook 03 chỉ chạy dev (protocol §2)"
'''))
md("## 2. Môi trường + GPU")
code(C.ENV_SRC + r'''
import importlib.util as _u
try:
    import torch
    ENV["torch"] = torch.__version__
    ENV["cuda"] = torch.version.cuda
    ENV["gpus"] = [(torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1))
                   for i in range(torch.cuda.device_count())]
except ImportError:
    raise RuntimeError("Không có PyTorch — hãy dùng môi trường Kaggle mặc định.")
import transformers
ENV["transformers"] = transformers.__version__
import importlib
for _m in ("accelerate", "bitsandbytes"):
    ENV[_m] = importlib.import_module(_m).__version__ if _u.find_spec(_m) else None
ENV["hf_cache_free_gb"] = round(shutil.disk_usage(Path.home()).free / 1e9, 1)
print("GPU:", ENV["gpus"], "| torch", ENV["torch"], "| transformers", ENV["transformers"],
      "| accelerate", ENV["accelerate"], "| bitsandbytes", ENV["bitsandbytes"], "| trống ~", ENV["hf_cache_free_gb"], "GB")
if not ENV["gpus"]:
    raise RuntimeError("Không thấy GPU. Settings → Accelerator → GPU T4 x2 rồi chạy lại.")
''')

md("## 3. Nạp mã nhúng + kiểm thử đơn vị (core, bộ chấm, LLM)")
code(C.embed_modules_cell(MODS, {"aristo_propara_evaluator.zip": VENDOR}) + r'''
import propara_core as pc, propara_metrics as pm, propara_heuristics as ph, propara_llm as pl
import propara_core_tests as pct, propara_metrics_tests as pmt, propara_llm_tests as plt_
EVAL_DIR = str(MOD_DIR / "aristo_propara_evaluator")
_fails = []
for _name, _mod in (("core", pct), ("metrics", pmt), ("llm", plt_)):
    _n, _f = _mod.run_all(verbose=False)
    print(f"{_name}: {_n - len(_f)}/{_n}")
    _fails += _f
if _fails:
    raise RuntimeError(f"Kiểm thử thất bại: {_fails}")
print(pc.CORE_VERSION, pm.METRICS_VERSION, pl.LLM_VERSION, pl.PROMPT_VERSION)
''')

md("## 4. Dữ liệu (tự tải + tự kiểm tra)")
code(C.LOAD_SRC + r'''
FEWSHOT = pl.select_fewshot(BY_SPLIT["train"], k=2)
_fs_ids = {r["paragraph_id"] for r in FEWSHOT}
SMOKE_RECS = [r for r in sorted(BY_SPLIT["train"], key=lambda r: pc.stable_hash("smoke:" + r["paragraph_id"]))
              if r["paragraph_id"] not in _fs_ids][:SMOKE_N]
DEV_RECS = BY_SPLIT[SPLIT][:LIMIT_PARAS] if LIMIT_PARAS else BY_SPLIT[SPLIT]
print("few-shot (train):", sorted(_fs_ids), "| smoke (train):", [r["paragraph_id"] for r in SMOKE_RECS], "| dev:", len(DEV_RECS), "đoạn")
''')

md("## 5. Nạp mô hình (tự chọn fp16 nhiều GPU / fp16 một GPU / 4-bit)")
code(r'''
from transformers import AutoTokenizer, AutoModelForCausalLM

def find_local_model():
    """Dò /kaggle/input cho thư mục mô hình Qwen2 7B (config.json + tokenizer)."""
    if not INPUT_ROOT.exists():
        return None
    for cfgp in sorted(INPUT_ROOT.rglob("config.json")):
        try:
            cfg = json.loads(cfgp.read_text())
        except Exception:
            continue
        if cfg.get("model_type") == "qwen2" and (cfg.get("hidden_size") == 3584 or not EXPECT_7B)                 and (cfgp.parent / "tokenizer_config.json").exists():
            return cfgp.parent
    return None

_local = find_local_model()
if _local is not None:
    MODEL_SRC, MODEL_ORIGIN = str(_local), f"kaggle_input:{_local}"
else:
    if ENV["internet"] is not True:
        raise RuntimeError("Không thấy mô hình trong /kaggle/input và không có Internet. Bật Internet hoặc Add Input → Models → Qwen2.5 7B Instruct (transformers).")
    if ENV["hf_cache_free_gb"] < 20 and EXPECT_7B:
        print("[CẢNH BÁO] Dung lượng trống < 20 GB, có thể không đủ để tải 7B.")
    from huggingface_hub import snapshot_download
    MODEL_SRC = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json"])
    MODEL_ORIGIN = f"hf:{MODEL_ID}@{MODEL_REVISION}"

n_gpu = torch.cuda.device_count()
vram = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(n_gpu)]
_cfg = json.loads((Path(MODEL_SRC) / "config.json").read_text())
_weights_gb = sum(f.stat().st_size for f in Path(MODEL_SRC).glob("*.safetensors")) / 2**30   # trọng số bf16/fp16 trên đĩa
_need_fp16 = _weights_gb * 1.1 + 1.5                  # GiB trọng số fp16 + dư cho KV cache/hoạt hoá
if EXPECT_7B and _cfg.get("hidden_size") != 3584:
    raise RuntimeError(f"Mô hình nạp được không phải 7B (hidden_size={_cfg.get('hidden_size')}).")
if n_gpu >= 2 and sum(vram) - 2 * n_gpu >= _need_fp16:
    QUANT = "fp16_multi_gpu"
elif vram[0] - 2 >= _need_fp16:
    QUANT = "fp16_single_gpu"
else:
    QUANT = "nf4_4bit"

tok = AutoTokenizer.from_pretrained(MODEL_SRC)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
t_load = time.time()
if QUANT == "fp16_multi_gpu":
    model = AutoModelForCausalLM.from_pretrained(MODEL_SRC, torch_dtype=torch.float16, device_map="auto",
                                                 max_memory={i: f"{int(vram[i] - 1.5)}GiB" for i in range(n_gpu)})
elif QUANT == "fp16_single_gpu":
    model = AutoModelForCausalLM.from_pretrained(MODEL_SRC, torch_dtype=torch.float16).to("cuda:0")
else:
    if ENV["bitsandbytes"] is None:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.43"], check=True)
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SRC, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
model.eval()
LOAD_SEC = round(time.time() - t_load, 1)
BATCH = {"fp16_multi_gpu": 8, "fp16_single_gpu": 8, "nf4_4bit": 4}[QUANT]
FIRST_DEVICE = next(model.parameters()).device
print(f"Mô hình: {MODEL_ORIGIN} | lượng tử hoá: {QUANT} | nạp {LOAD_SEC}s | batch {BATCH} | thiết bị đầu: {FIRST_DEVICE}")
for i in range(n_gpu):
    print(f"  GPU{i}: đã dùng {torch.cuda.memory_allocated(i)/2**30:.1f} / {vram[i]:.1f} GiB")
''')

md("## 6. Hàm sinh có lô, lùi lô khi hết bộ nhớ, ghi tạm để tiếp tục khi bị ngắt")
code(r'''
RAWDIR = OUT / "raw_outputs"; RAWDIR.mkdir(exist_ok=True)
GEN_LOG = []
# Tiếp tục sang phiên mới: nếu đã Add Input output phiên bản trước của notebook này, chép đầu ra thô đã sinh
if INPUT_ROOT.exists():
    for _prev in sorted(INPUT_ROOT.rglob(f"{RUN_TAG}/raw_outputs")):
        _n = 0
        for _f in _prev.glob("*.jsonl"):
            if not (RAWDIR / _f.name).exists():
                shutil.copy2(_f, RAWDIR / _f.name); _n += 1
        print(f"Khôi phục {_n} file đầu ra thô từ phiên trước: {_prev}")
        break

def _done_keys(path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["key"]] = d
    return out

def generate_all(items, tag):
    """items: [(key, messages, max_new_tokens)] → {key: record}. Ghi JSONL sau mỗi lô; bỏ qua key đã có."""
    path = RAWDIR / f"{tag}.jsonl"
    done = _done_keys(path)
    todo = [it for it in items if it[0] not in done]
    prompts = {k: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for k, m, _ in todo}
    todo.sort(key=lambda it: len(prompts[it[0]]))
    i, bs = 0, BATCH
    with open(path, "a", encoding="utf-8") as f:
        while i < len(todo):
            batch = todo[i:i + bs]
            texts = [prompts[k] for k, _, _ in batch]
            mnt = max(m for _, _, m in batch)
            try:
                enc = tok(texts, return_tensors="pt", padding=True).to(FIRST_DEVICE)
                for d in range(n_gpu):
                    torch.cuda.reset_peak_memory_stats(d)
                t0 = time.time()
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                         pad_token_id=tok.pad_token_id)
                dt = time.time() - t0
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                GEN_LOG.append({"tag": tag, "event": "oom_backoff", "new_batch": bs})
                print(f"  [OOM] giảm lô còn {bs}")
                continue
            peak = max(torch.cuda.max_memory_allocated(d) for d in range(n_gpu)) / 2**30
            new = gen[:, enc["input_ids"].shape[1]:]
            for (k, m, cap), row, inp in zip(batch, new, enc["attention_mask"]):
                ids = row.tolist()
                n_out = sum(1 for x in ids if x != tok.pad_token_id)
                txt = tok.decode(row, skip_special_tokens=True)
                rec = {"key": k, "text": txt, "prompt_tokens": int(inp.sum()), "output_tokens": n_out,
                       "hit_max_new_tokens": n_out >= cap, "batch_sec": round(dt, 2), "batch_size": len(batch),
                       "peak_vram_gib": round(peak, 2)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[k] = rec
            f.flush()
            GEN_LOG.append({"tag": tag, "batch": len(batch), "sec": round(dt, 2), "peak_vram_gib": round(peak, 2), "max_new": mnt})
            i += len(batch)
            print(f"  {tag}: {min(i, len(todo))}/{len(todo)} | lô {len(batch)} | {dt:.1f}s | đỉnh VRAM {peak:.1f} GiB")
    return done

def items_for(recs, mode):
    if mode == "ORACLE_PREV_STATE":
        return [(f"{r['paragraph_id']}|{t}", pl.build_messages(r, mode, t=t), pl.max_new_tokens(r, mode))
                for r in recs for t in range(1, len(r["sentences"]) + 1)]
    return [(r["paragraph_id"], pl.build_messages(r, mode, fewshot=FEWSHOT), pl.max_new_tokens(r, mode)) for r in recs]

def predictions_for(recs, mode, raw):
    preds, diags = {}, {}
    for r in recs:
        if mode == "ORACLE_PREV_STATE":
            steps, dsum = {}, Counter()
            for t in range(1, len(r["sentences"]) + 1):
                ch, prs, d = pl.parse_oracle_step(raw[f"{r['paragraph_id']}|{t}"]["text"], r, t)
                steps[t] = (ch, prs)
                dsum[d["json_status"]] += 1
                dsum["hallucinated"] += d["hallucinated"]
            preds[r["paragraph_id"]] = pl.assemble_oracle(r, steps)
            diags[r["paragraph_id"]] = dict(dsum)
        else:
            p, d = pl.parse_to_prediction(raw[r["paragraph_id"]]["text"], r, mode)
            d["hit_max_new_tokens"] = raw[r["paragraph_id"]]["hit_max_new_tokens"]
            preds[r["paragraph_id"]], diags[r["paragraph_id"]] = p, d
    return preds, diags
''')

md("## 7. SMOKE TEST (3 đoạn train): tự quyết định có chạy dev hay không")
code(r'''
SMOKE = {}
t_smoke = time.time()
for mode in MODES_TO_RUN:
    recs = SMOKE_RECS[:1] if mode == "ORACLE_PREV_STATE" else SMOKE_RECS
    raw = generate_all(items_for(recs, mode), f"smoke_{mode}")
    preds, diags = predictions_for(recs, mode, raw)
    if mode == "ORACLE_PREV_STATE":
        n_valid = sum(d.get("valid", 0) + d.get("repaired", 0) for d in diags.values())
        n_tot = sum(len(r["sentences"]) for r in recs)
    else:
        n_valid = sum(d["json_status"] in ("valid", "repaired") for d in diags.values())
        n_tot = len(recs)
    res = pm.evaluate({k: pl.as_eval_prediction(v) for k, v in preds.items()}, recs)
    SMOKE[mode] = {"valid_json": f"{n_valid}/{n_tot}", "pair_f1_train_smoke": round(res["PAIR@all"]["F1"], 3),
                   "hallucinated": sum(d.get("hallucinated", 0) for d in diags.values())}
    print(mode, SMOKE[mode])
first = SMOKE_RECS[0]["paragraph_id"]
print("\n--- Đầu ra thô mẫu (QWEN_DIRECT, đoạn train", first, ") ---\n", _done_keys(RAWDIR / "smoke_QWEN_DIRECT.jsonl")[first]["text"][:1500])
SMOKE_SEC = round(time.time() - t_smoke, 1)

_q0_valid = int(SMOKE["QWEN_DIRECT"]["valid_json"].split("/")[0])
SMOKE_OK = _q0_valid >= SMOKE_MIN_VALID
SMOKE_REASON = "OK" if SMOKE_OK else f"QWEN_DIRECT chỉ có {_q0_valid}/{SMOKE_N} JSON hợp lệ (< {SMOKE_MIN_VALID})"
print(f"\nSMOKE {'ĐẠT' if SMOKE_OK else 'KHÔNG ĐẠT'} ({SMOKE_REASON}) | {SMOKE_SEC}s")
DO_FULL = (SMOKE_OK and RUN_FULL == "auto") or RUN_FULL == "force"   # "force" chỉ dùng khi kiểm thử cục bộ
if not DO_FULL:
    print("→ KHÔNG chạy dev. Lưu kết quả smoke rồi dừng ở ô cuối. Gửi output cho trợ lý.")
''')

md("## 8. Chạy toàn bộ DEV (chỉ khi smoke đạt)")
code(r'''
RAW_ALL, PREDS, DIAGS, MODE_SEC = {}, {}, {}, {}
if DO_FULL:
    for mode in MODES_TO_RUN:
        t0 = time.time()
        RAW_ALL[mode] = generate_all(items_for(DEV_RECS, mode), f"{SPLIT}_{mode}")
        PREDS[mode], DIAGS[mode] = predictions_for(DEV_RECS, mode, RAW_ALL[mode])
        MODE_SEC[mode] = round(time.time() - t0, 1)
        print(f"{mode}: xong trong {MODE_SEC[mode]}s")
    # Q3 = hậu xử lý luật tồn tại trên Q0 (không gọi lại mô hình)
    PREDS["QWEN_CONSTRAINED"], DIAGS["QWEN_CONSTRAINED"] = {}, {}
    for pid, p in PREDS["QWEN_DIRECT"].items():
        q, fixed = pl.apply_existence_constraints(p)
        PREDS["QWEN_CONSTRAINED"][pid] = q
        DIAGS["QWEN_CONSTRAINED"][pid] = {"cells_fixed": fixed}
''')

md("## 9. Chấm điểm DEV (cùng bộ chấm Cửa nghiệm thu 1) + so với mốc tham chiếu")
code(r'''
TMP = OUT / "action_files"; TMP.mkdir(exist_ok=True)
ANS = str(RAW[f"leaderboard_answers_{SPLIT}.tsv"])
SCORES, ROWS = {}, []

def score(name, preds, kind, pair_source="labels"):
    ev = {k: pl.as_eval_prediction(v, pair_source) if "pairs_llm" in v or pair_source == "llm" else v for k, v in preds.items()}
    res = pm.evaluate(ev, DEV_RECS)
    path = TMP / f"{name}.tsv"
    pm.write_action_file(str(path), [row for r in DEV_RECS for row in pm.to_official_action_rows(ev[r["paragraph_id"]], r)])
    res["Q3_OFFICIAL"] = pm.official_q3(EVAL_DIR, ANS, str(path)) if not LIMIT_PARAS else None
    SCORES[name] = res
    ci = pm.bootstrap_ci_pair_f1(res, "all", n_boot=N_BOOT, seed=SEED)
    q = res["Q3_OFFICIAL"]
    ROWS.append(pm.summary_row(name, res, {"kind": kind, "pairs_from": pair_source,
                "PAIR_F1_CI95": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                "Q3_conv_F1": q["conversions"]["F1"] if q else None, "Q3_overall_F1": q["overall"]["F1"] if q else None}))

score("ALL_NONE", {r["paragraph_id"]: ph.all_none(r) for r in DEV_RECS}, "reference")
score("DIAG_ORACLE_MENTIONED_ONLY", {r["paragraph_id"]: ph.oracle_mentioned_only(r) for r in DEV_RECS}, "diagnostic(gold)")
if DO_FULL:
    for mode in ["QWEN_DIRECT", "QWEN_FEWSHOT", "QWEN_STRUCTURED", "QWEN_CONSTRAINED"]:
        if mode in PREDS:
            score(mode, PREDS[mode], "end-to-end")
            score(mode + "[llm_pairs]", PREDS[mode], "end-to-end (cặp LLM tự khai, thước đo phụ)", pair_source="llm")
    if "ORACLE_PREV_STATE" in PREDS:
        score("ORACLE_PREV_STATE", PREDS["ORACLE_PREV_STATE"], "ORACLE (không so đầu-cuối)")

cols = ["system", "kind", "PAIR_F1", "PAIR_F1_CI95", "PAIR_P", "PAIR_R", "PAIR_F1@src_imp", "PAIR_F1@tgt_imp", "PAIR_F1@any_imp",
        "IMP_ACC_CREATE", "IMP_ACC_DESTROY", "F1_MOVE", "EXIST_ACC", "VIOL_per_entity", "Q3_conv_F1"]
print("  ".join(f"{c[:15]:>15s}" for c in cols))
for r in ROWS:
    print("  ".join(f"{(f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c]))[:15]:>15s}" for c in cols))
''')

md("## 10. Chẩn đoán kỹ thuật + điểm quyết định ngầm + so sánh có kiểm định")
code(r'''
TECH = {}
if DO_FULL:
    for mode in MODES_TO_RUN:
        raw = RAW_ALL[mode]
        lat = sorted(v["batch_sec"] / v["batch_size"] for v in raw.values())
        if mode == "ORACLE_PREV_STATE":
            st = Counter(k for d in DIAGS[mode].values() for k, v in d.items() if k in ("valid", "repaired", "truncated", "invalid") for _ in range(v))
            n_calls = sum(st.values())
        else:
            st = Counter(d["json_status"] for d in DIAGS[mode].values())
            n_calls = len(DIAGS[mode])
        TECH[mode] = {"calls": n_calls, "json_valid_rate": (st["valid"] + st["repaired"]) / max(n_calls, 1),
                      "json_status": dict(st), "hit_max_new_tokens": sum(v["hit_max_new_tokens"] for v in raw.values()),
                      "hallucinated_total": sum(d.get("hallucinated", 0) for d in DIAGS[mode].values()),
                      "latency_per_call_p50_s": lat[len(lat) // 2], "latency_per_call_p95_s": lat[int(0.95 * (len(lat) - 1))],
                      "prompt_tokens_mean": statistics.mean(v["prompt_tokens"] for v in raw.values()),
                      "output_tokens_mean": statistics.mean(v["output_tokens"] for v in raw.values()),
                      "peak_vram_gib": max(v["peak_vram_gib"] for v in raw.values()), "mode_wall_sec": MODE_SEC[mode],
                      "gpu_hours_per_1000_paragraphs_est": MODE_SEC[mode] / len(DEV_RECS) * 1000 / 3600}
        print(mode, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in TECH[mode].items()})
    TECH["QWEN_CONSTRAINED"] = {"cells_fixed_total": sum(d["cells_fixed"] for d in DIAGS["QWEN_CONSTRAINED"].values())}
    print("QWEN_CONSTRAINED", TECH["QWEN_CONSTRAINED"])

# Điểm quyết định ngầm trên dev: hệ thống 'chọn' tập S = ứng viên được gán nhãn L; trúng nếu S ∩ G ≠ ∅, đúng hẳn nếu S = G
DPS_DEV = [d for r in DEV_RECS for d in pc.decision_points(r)]
DP_RES = {}
for name in [n for n in SCORES if not n.endswith("[llm_pairs]")]:
    src = PREDS.get(name) or ({r["paragraph_id"]: ph.all_none(r) for r in DEV_RECS} if name == "ALL_NONE" else
                              {r["paragraph_id"]: ph.oracle_mentioned_only(r) for r in DEV_RECS})
    hit = exact = 0
    for d in DPS_DEV:
        S = {i for i in d["candidates"] if src[d["paragraph_id"]]["labels"][i][d["sentence_id"] - 1] == d["label"]}
        hit += bool(S & set(d["gold"])); exact += S == set(d["gold"])
    DP_RES[name] = {"n": len(DPS_DEV), "hit": hit, "exact": exact}
print("Điểm quyết định ngầm dev (n =", len(DPS_DEV), "):", DP_RES)
print("Mốc heuristic nhắc-gần-nhất trên dev (kỳ vọng):",
      {lab: round(statistics.mean(ph.expected_acc(d, BY_PID[d["paragraph_id"]], "nearest_mention") for d in DPS_DEV if d["label"] == lab), 3)
       for lab in ("DESTROY", "CREATE")})

BOOT = {}
if DO_FULL:
    for a, b in [("QWEN_FEWSHOT", "QWEN_DIRECT"), ("QWEN_STRUCTURED", "QWEN_DIRECT"), ("QWEN_CONSTRAINED", "QWEN_DIRECT")]:
        for sl in ("all", "any_implicit"):
            BOOT[f"{a}-{b}@{sl}"] = pm.paired_bootstrap_pair_f1(SCORES[a], SCORES[b], slice_name=sl, n_boot=N_BOOT, seed=SEED)
            x = BOOT[f"{a}-{b}@{sl}"]
            print(f"Δ {a} − {b} @{sl}: {x['delta']:+.3f}  CI95 [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]")
''')

md("## 11. Lưu đầu ra")
code(r'''
def jdump(obj, path):
    def conv(o):
        if isinstance(o, Counter): return dict(o)
        if isinstance(o, tuple): return list(o)
        if isinstance(o, Path): return str(o)
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=conv), encoding="utf-8")

P = OUT / "predictions"; P.mkdir(exist_ok=True)
for mode, preds in PREDS.items():
    with open(P / f"{SPLIT}_{mode}.jsonl", "w", encoding="utf-8") as f:
        for pid, p in preds.items():
            f.write(json.dumps(dict(p, diag=DIAGS[mode][pid]), ensure_ascii=False) + "\n")
with open(OUT / f"scores_{SPLIT}.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys())); w.writeheader(); w.writerows(ROWS)
for name, res in SCORES.items():
    jdump({k: v for k, v in res.items() if k != "_per_paragraph"}, OUT / f"metrics_{SPLIT}_{name.replace('[', '_').replace(']', '')}.json")
(OUT / "prompts").mkdir(exist_ok=True)
for mode in MODES_TO_RUN:
    ex = DEV_RECS[0]
    jdump(pl.build_messages(ex, mode, fewshot=FEWSHOT, t=1 if mode == "ORACLE_PREV_STATE" else None), OUT / "prompts" / f"example_{mode}.json")
CONFIG = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_origin": MODEL_ORIGIN, "quantization": QUANT,
          "decoding": "greedy (do_sample=False, repetition_penalty=1.0)", "batch_size_initial": BATCH, "max_new_tokens_rule": "propara_llm.max_new_tokens",
          "prompt_version": pl.PROMPT_VERSION, "llm_version": pl.LLM_VERSION, "core": pc.CORE_VERSION, "metrics": pm.METRICS_VERSION,
          "fewshot_train_ids": sorted(_fs_ids), "smoke_train_ids": [r["paragraph_id"] for r in SMOKE_RECS], "split": SPLIT,
          "modes": MODES_TO_RUN, "seed": SEED, "limit_paras": LIMIT_PARAS}
SUMMARY = {"run_tag": RUN_TAG, "smoke_ok": SMOKE_OK, "smoke_reason": SMOKE_REASON, "smoke": SMOKE, "did_full_dev": DO_FULL,
           "config": CONFIG, "env": ENV, "load_sec": LOAD_SEC, "smoke_sec": SMOKE_SEC, "mode_sec": MODE_SEC, "tech": TECH,
           "scores": ROWS, "decision_points_dev": DP_RES, "bootstrap": BOOT, "gen_log_tail": GEN_LOG[-20:],
           "runtime_sec": round(time.time() - T0, 1)}
jdump(SUMMARY, OUT / "summary_03.json")
jdump(CONFIG, OUT / "config_locked_candidate.json")
ZIP = WORK_ROOT / f"{RUN_TAG}.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "raw" != p.parent.name and "code" not in p.parts:
            z.write(p, Path(RUN_TAG) / p.relative_to(OUT))
print("Đã lưu", OUT, "| zip", ZIP, f"{ZIP.stat().st_size/1e6:.1f} MB | tổng {SUMMARY['runtime_sec']} s")
''')

md("## 12. Gửi lại")
code(r'''
print("=== GỬI LẠI ===")
print(f"notebook=03_qwen7b_dev model={MODEL_ORIGIN} quant={QUANT} gpus={ENV['gpus']} torch={ENV['torch']} transformers={ENV['transformers']}")
print(f"smoke_ok={SMOKE_OK} ({SMOKE_REASON}) smoke={SMOKE} | full_dev={DO_FULL} | load={LOAD_SEC}s smoke={SMOKE_SEC}s modes={MODE_SEC} total={SUMMARY['runtime_sec']}s")
for m, t in TECH.items():
    print(f"  tech {m}: " + ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in t.items()
                                         if k in ("json_valid_rate", "hit_max_new_tokens", "hallucinated_total", "peak_vram_gib", "latency_per_call_p50_s", "cells_fixed_total")))
for r in ROWS:
    print(f"  {r['system']:32s} PAIR_F1={r['PAIR_F1']:.3f} {r['PAIR_F1_CI95']} @src_imp={r['PAIR_F1@src_imp']:.3f} @tgt_imp={r['PAIR_F1@tgt_imp']:.3f} "
          f"IMP_D={r['IMP_ACC_DESTROY']:.2f} IMP_C={r['IMP_ACC_CREATE']:.2f} VIOL={r['VIOL_per_entity']:.3f} Q3conv={r['Q3_conv_F1'] if r['Q3_conv_F1'] is None else round(r['Q3_conv_F1'], 3)}")
print("  DP dev:", DP_RES)
for k, x in BOOT.items():
    print(f"  Δ {k}: {x['delta']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]")
print("=== HẾT ===")
''')

out = C.save_notebook(cells, ROOT / "notebooks" / "03_qwen7b_dev.ipynb", gpu=True)
print("Đã ghi", out, "| số ô:", len(cells))
