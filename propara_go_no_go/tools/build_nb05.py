"""Sinh notebooks/05_pp3_pp4_dev_cv.ipynb (lượt Kaggle K2: PP3 + PP4 + ablation + CV).  python tools/build_nb05.py <dự_án>"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common as C  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODS = {n: (SRC / n).read_text(encoding="utf-8") for n in
        ["propara_core.py", "propara_core_tests.py", "propara_metrics.py", "propara_metrics_tests.py", "propara_heuristics.py",
         "propara_pp3.py", "propara_pp4.py", "propara_pp4_tests.py", "propara_robust.py"]}
VENDOR = (ROOT / "vendor" / "aristo_propara_evaluator.zip").read_bytes()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 05: Phương pháp nhóm: PP3 (tri thức X→Y từ WikiHow) + PP4 (RoBERTa+BiLSTM+CRF) + ablation + CV · lượt Kaggle **K2**

**Mục tiêu (H1–H3):** huấn luyện các phiên bản của mô hình nhóm, chọn epoch trên **dev**, đo đóng góp từng thành phần; rồi so **M3 với M0c** bằng CV 5 phần theo chủ đề trên lát **ngầm**.

| Mã | Biến thể (`variant`) | Ý nghĩa |
|---|---|---|
| M0 | `M0` | phân loại từng ô độc lập (cross-entropy), không ràng buộc |
| M0c | `M0` + Viterbi ràng buộc | **cùng mô hình M0**, chỉ thêm luật tồn tại khi giải mã → tách giá trị của *luật* |
| M3-no-pair | `CRF_NOPAIR` | CRF giải mã chung, huấn luyện −log p(y_vàng), không điểm cặp |
| M3-no-E | `CRF_NOE` | CRF + điểm cặp ψ **không** có log E(X,Y) |
| M2r | `CRF_RANDE` | CRF + ψ với log E **xáo trộn ngẫu nhiên** (kiểm tra “thêm đặc trưng” ≠ “tri thức hữu ích”) |
| **M3** | `CRF_FULL` | CRF + ψ + log E(X,Y) từ PP3 (**phương pháp nhóm đầy đủ**) |

**PP2:** trong giải mã chính xác, ràng buộc tồn tại đã bao hàm bộ lọc PP2. PP2 được đo riêng như **độ phủ cặp vàng khi dùng tồn tại dự đoán** (H1).

**Quy tắc khoá trước:**
- chọn epoch theo PAIR_F1 dev (hoà thì lấy dev loss thấp hơn);
- 3 seed;
- CV dùng số epoch = trung vị epoch tốt nhất trên dev của biến thể đó (không chỉnh trên fold kiểm tra);
- dự đoán **test được lưu kín, KHÔNG chấm** trong notebook này (chấm ở bước khoá K3).

### Cài đặt Kaggle
| Mục | Giá trị |
|---|---|
| Accelerator | **GPU** (T4 x2 hoặc P100; notebook dùng 1 GPU) |
| Internet | **On** (tải `roberta-base` ~500 MB và WikiHow `wikihowAll.csv` ~620 MB vào bộ nhớ tạm, không vào Output) |
| Output | `/kaggle/working/pp34_v2/`: dự đoán dev (+ test kín), lịch sử loss/F1, biểu đồ, E(X,Y), checkpoint tốt nhất của 6 hệ thống (fp16) |
| Thời gian | **ước tính** 3–5 giờ GPU (chưa đo; 25 lượt huấn luyện ≤ 8 epoch + PP3 + CV). Dừng sớm (patience 3); `MAX_HOURS`=9 để kết thúc êm trước giới hạn phiên Kaggle |
| Checkpoint | Mỗi epoch lưu `state/inprogress_*.pt` (mô hình + optimizer + scheduler + RNG). Bị ngắt thì **Run All lại trong cùng phiên** sẽ tiếp tục đúng epoch đó. **Sang phiên mới:** Add Input → *Notebooks* → output phiên bản trước của chính notebook này → Run All (phần đã xong được bỏ qua). Checkpoint TỐT NHẤT của mọi hệ thống (seed đầu): `checkpoints/*_best.pt` |

**Chạy:** Save Version → **Save & Run All (Commit)**. **Gửi lại:** khối `=== GỬI LẠI ===` + `pp34_v2/summary_05.json`.
""")

md("## 1. Cấu hình")
code(C.cfg_header("pp34_v2", extra=r'''
ENCODER_ID   = os.environ.get("PROPARA_ENCODER_ID", "FacebookAI/roberta-base")
ENCODER_REV  = os.environ.get("PROPARA_ENCODER_REV", "e2da8e2f811d1448a5b465c236feacd80ffbac7b")
WIKIHOW_URL  = "https://huggingface.co/datasets/Lajvi/wikihowAll/resolve/5e44fed0e204c27ad985e6292e55e311504d4fe5/wikihowAll.csv"
WIKIHOW_SHA  = "8ebf5bd485275af895aa83ea3e55c416fe76aa4fefc58e0b9e6d344852bde5aa"
WIKIHOW_LOCAL = os.environ.get("PROPARA_WIKIHOW_LOCAL", "")        # chỉ dùng khi kiểm thử cục bộ
VARIANTS     = ["M0", "CRF_NOPAIR", "CRF_NOE", "CRF_RANDE", "CRF_FULL"]
SEEDS        = [int(x) for x in os.environ.get("PROPARA_SEEDS", "13,42,2026").split(",")]
EPOCHS       = int(os.environ.get("PROPARA_EPOCHS", "8"))
RUN_CV       = os.environ.get("PROPARA_RUN_CV", "1") == "1"
CV_VARIANTS  = ["M0", "CRF_FULL"]                                    # M0 cho cả M0 và M0c
LIMIT_TRAIN  = int(os.environ.get("PROPARA_LIMIT_TRAIN", "0"))       # >0 chỉ để kiểm thử cục bộ
MAX_HOURS    = float(os.environ.get("PROPARA_MAX_HOURS", "9"))   # không bắt đầu lượt huấn luyện mới sau mốc này → kết thúc êm, kịp lưu Output
PATIENCE     = int(os.environ.get("PROPARA_PATIENCE", "3"))      # dừng sớm: PAIR_F1 dev không tăng 3 epoch liên tiếp
STOP_AFTER   = int(os.environ["PROPARA_STOP_AFTER_EPOCHS"]) if os.environ.get("PROPARA_STOP_AFTER_EPOCHS") else None  # chỉ để kiểm thử
N_BOOT       = 10000
'''))
md("## 2. Môi trường + GPU")
code(C.ENV_SRC + r'''
import importlib
import torch, transformers
ENV.update({"torch": torch.__version__, "transformers": transformers.__version__,
            "gpus": [(torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)) for i in range(torch.cuda.device_count())]})
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU:", ENV["gpus"], "| torch", ENV["torch"], "| transformers", ENV["transformers"], "| device", DEVICE)
if DEVICE == "cpu" and not LIMIT_TRAIN:
    raise RuntimeError("Không thấy GPU. Settings → Accelerator → GPU.")
''')
md("## 3. Nạp mã nhúng + kiểm thử (core, bộ chấm, CRF so với liệt kê vét cạn)")
code(C.embed_modules_cell(MODS, {"aristo_propara_evaluator.zip": VENDOR}) + r'''
import propara_core as pc, propara_metrics as pm, propara_heuristics as ph, propara_pp3 as p3, propara_pp4 as pp, propara_robust as pr
import propara_core_tests as pct, propara_metrics_tests as pmt, propara_pp4_tests as ppt
EVAL_DIR = str(MOD_DIR / "aristo_propara_evaluator")
_f = []
for _nm, _m in (("core", pct), ("metrics", pmt), ("pp4/CRF", ppt)):
    _n, _ff = _m.run_all(verbose=False); print(f"{_nm}: {_n - len(_ff)}/{_n}"); _f += _ff
if _f:
    raise RuntimeError(f"Kiểm thử thất bại: {_f}")
''')
md("## 4. Dữ liệu")
code(C.LOAD_SRC + r'''
TRAIN = BY_SPLIT["train"][:LIMIT_TRAIN] if LIMIT_TRAIN else BY_SPLIT["train"]
DEV = BY_SPLIT["dev"][:max(5, LIMIT_TRAIN // 2)] if LIMIT_TRAIN else BY_SPLIT["dev"]
TEST = BY_SPLIT["test"][:max(5, LIMIT_TRAIN // 2)] if LIMIT_TRAIN else BY_SPLIT["test"]
FOLDS = pc.cv_folds_by_topic(RECS, k=5, seed=SEED)
print("train/dev/test:", len(TRAIN), len(DEV), len(TEST))
STATE = OUT / "state"; STATE.mkdir(exist_ok=True)
def tdone(): return (time.time() - T0) / 3600

# TIẾP TỤC SANG PHIÊN MỚI: nếu đã Add Input output của một phiên bản trước của notebook này,
# chép các phần đã xong (kết quả, dự đoán, test kín, checkpoint) vào OUT rồi bỏ qua chúng khi chạy.
RESTORED = []
if INPUT_ROOT.exists():
    for prev in sorted(INPUT_ROOT.rglob(f"{RUN_TAG}/state/results.json")):
        prev_out = prev.parent.parent
        for sub in ("state", "predictions", "sealed_test_predictions", "checkpoints"):
            if (prev_out / sub).exists():
                for f in (prev_out / sub).rglob("*"):
                    if f.is_file():
                        dst = OUT / sub / f.relative_to(prev_out / sub)
                        if not dst.exists():
                            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst); RESTORED.append(str(dst.relative_to(OUT)))
        print(f"Khôi phục từ phiên trước: {prev_out} → {len(RESTORED)} file")
        break
''')

md("## 5. Encoder `roberta-base` (ghim revision) + kiểm tra độ dài đầu vào")
code(r'''
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
_local_enc = os.environ.get("PROPARA_ENCODER_LOCAL", "")
ENC_PATH = _local_enc or snapshot_download(ENCODER_ID, revision=ENCODER_REV, allow_patterns=["*.json", "*.txt", "*.safetensors"])
tok = AutoTokenizer.from_pretrained(ENC_PATH)
def make_encoder():
    enc = AutoModel.from_pretrained(ENC_PATH)
    return enc, enc.config.hidden_size
_lens = [len(pp.build_entity_input(tok, r, i)[0]) for r in RECS for i in range(len(r["entities"]))]
print(f"độ dài đầu vào (thực thể + đoạn): TB {statistics.mean(_lens):.0f}, max {max(_lens)} token")
_emb_model = AutoModel.from_pretrained(ENC_PATH)
EMB_MATRIX = _emb_model.get_input_embeddings().weight.detach().float().cpu()
del _emb_model
def embed_words(words):
    """Embedding TĨNH (ma trận đầu vào roberta, trung bình sub-token) cho danh từ đầu, dùng cho E(X,Y)."""
    out = []
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False) or [tok.unk_token_id]
        out.append(EMB_MATRIX[ids].mean(0))
    return torch.stack(out)
''')

md("## 6. PP3: khai thác WikiHow → K, độ phủ trên 533 cặp vàng, huấn luyện E(X,Y)")
code(r'''
PP3 = {"status": "not_run"}
LOGE = {}
TAGGER = None
try:
    if WIKIHOW_LOCAL:
        wh_path = Path(WIKIHOW_LOCAL); PP3["source"] = f"local:{wh_path}"
    else:
        wh_path = Path.home() / "wikihowAll.csv"
        if not wh_path.exists():
            print("Đang tải WikiHow (~620 MB)…"); urllib.request.urlretrieve(WIKIHOW_URL, wh_path)
        _sha = pc.sha256_file(str(wh_path))
        PP3["source"] = WIKIHOW_URL; PP3["sha256_match"] = _sha == WIKIHOW_SHA
        print("SHA-256 khớp:", PP3["sha256_match"])
    # bộ gán nhãn từ loại (pp3_v2, cũng dùng để tách cụm danh từ cho đầu vị trí); thiếu thì lùi về regex và ghi cảnh báo
    try:
        try:
            import nltk
        except ImportError:                      # Kaggle thường có sẵn; nếu thiếu thì cài (cần Internet)
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nltk"], check=True)
            import nltk
        from nltk.tag import PerceptronTagger
        for _res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
            try:
                TAGGER = PerceptronTagger().tag; break
            except LookupError:
                nltk.download(_res, quiet=True)
                try:
                    TAGGER = PerceptronTagger().tag; break
                except LookupError:
                    continue
    except Exception as _e:
        print("[CẢNH BÁO] không có NLTK tagger:", _e)
    PP3["extractor"] = "pos_nltk_perceptron (pp3_v2)" if TAGGER else "regex_only (pp3_v1, chất lượng thấp)"
    print("Bộ trích:", PP3["extractor"])
    csv.field_size_limit(10**9)
    def _rows():
        with open(wh_path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                yield row.get("title") or "", row.get("text") or ""
    t0 = time.time()
    PAIR_CNT, PAIR_ART, PAT_CNT, CORPUS, ART_NOUNS = p3.mine_corpus(_rows(), tagger=TAGGER)
    PP3.update({"mine_sec": round(time.time() - t0, 1), "corpus": CORPUS, "n_pairs_distinct": len(PAIR_CNT),
                "n_pair_mentions": sum(PAIR_CNT.values()), "pattern_counts": dict(PAT_CNT),
                "top_pairs": [f"{x}->{y}:{c}" for (x, y), c in PAIR_CNT.most_common(40)]})
    print({k: v for k, v in PP3.items() if k != "top_pairs"}); print("top:", PP3["top_pairs"][:20])
    GOLD_ALL = [g for r in RECS for g in pc.gold_pairs_report(r)]
    PP3["coverage_all_min1"] = p3.coverage(GOLD_ALL, set(PAIR_CNT))
    PP3["coverage_all_min2"] = p3.coverage(GOLD_ALL, {p for p, c in PAIR_CNT.items() if c >= 2})
    print("Độ phủ (24)–(26) trên 533 cặp vàng:", PP3["coverage_all_min1"], "| K với tần suất ≥2:", PP3["coverage_all_min2"])
    DATA, K = p3.build_training_pairs(PAIR_CNT, PAIR_ART, ART_NOUNS, seed=SEED)
    PP3["train_val_sizes"] = {k: len(v) for k, v in DATA.items()}
    print("Tập huấn luyện E:", PP3["train_val_sizes"])
    E_MODEL, E_HIST = p3.train_pair_scorer(DATA, embed_words, device=DEVICE, seed=SEED)
    E_MODEL = E_MODEL.to(DEVICE).eval()
    PP3["history"] = E_HIST
    PP3["best_val"] = min(E_HIST, key=lambda r: r["val_loss"])
    for r in RECS:
        LOGE[r["paragraph_id"]] = p3.logE_matrix(E_MODEL, embed_words, r, device=DEVICE)
    # E có phân biệt cặp vàng ProPara với cặp có thứ tự khác trong cùng đoạn không? (chẩn đoán, theo tập)
    for sp in SPLITS:
        posv, negv = [], []
        for r in BY_SPLIT[sp]:
            G = {(g["source_idx"], g["target_idx"]) for g in pc.gold_pairs_report(r)}
            n = len(r["entities"])
            for i in range(n):
                for j in range(n):
                    if i != j:
                        (posv if (i, j) in G else negv).append(LOGE[r["paragraph_id"]][i][j])
        PP3[f"propara_auc_{sp}"] = p3._auc(posv, negv)
    print("AUC của log E trên cặp ProPara (vàng vs cặp khác cùng đoạn):", {sp: round(PP3[f"propara_auc_{sp}"], 3) for sp in SPLITS})
    PP3["status"] = "ok"
except Exception as e:
    import traceback; traceback.print_exc()
    PP3["status"] = f"FAILED: {type(e).__name__}: {e}"
    print("[CẢNH BÁO] PP3 lỗi → bỏ CRF_FULL/CRF_RANDE, vẫn chạy các biến thể còn lại.")
if PP3["status"] != "ok":
    VARIANTS = [v for v in VARIANTS if v not in ("CRF_FULL", "CRF_RANDE")]
    CV_VARIANTS = [v if v != "CRF_FULL" else "CRF_NOE" for v in CV_VARIANTS]
    LOGE = {r["paragraph_id"]: [[0.0] * len(r["entities"]) for _ in r["entities"]] for r in RECS}
# M2r: xáo trộn log E giữa mọi cặp (giữ phân phối, phá tri thức), seed cố định
_vals = [v for r in RECS for i, row in enumerate(LOGE[r["paragraph_id"]]) for j, v in enumerate(row) if i != j]
random.Random(SEED).shuffle(_vals)
LOGE_RAND, _k = {}, 0
for r in RECS:
    n = len(r["entities"]); M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i][j] = _vals[_k]; _k += 1
    LOGE_RAND[r["paragraph_id"]] = M
''')

md("## 7. Đặc trưng cho PP4")
code(r'''
t0 = time.time()
FEATS = {r["paragraph_id"]: pp.featurize(tok, r, LOGE[r["paragraph_id"]], tagger=TAGGER) for r in RECS}
LOC_COVER = {sp: (sum(FEATS[r["paragraph_id"]]["loc_cover"][0] for r in BY_SPLIT[sp]), sum(FEATS[r["paragraph_id"]]["loc_cover"][1] for r in BY_SPLIT[sp]))
             for sp in SPLITS}
print("Đầu vị trí: tỉ lệ vị trí vàng có trong ứng viên cụm danh từ (gồm '?') =",
      {sp: f"{h}/{n}={h/max(n,1):.1%}" for sp, (h, n) in LOC_COVER.items()}, "| tách cụm bằng", "NLTK" if TAGGER else "cửa sổ từ (lùi)")
FEATS_RAND = {pid: dict(f, logE=torch.tensor(LOGE_RAND[pid], dtype=torch.float)) for pid, f in FEATS.items()}
def feats_for(variant, recs):
    src = FEATS_RAND if variant == "CRF_RANDE" else FEATS
    return [src[r["paragraph_id"]] for r in recs]
def model_variant(v):          # CRF_RANDE dùng kiến trúc CRF_FULL với log E xáo trộn
    return "CRF_FULL" if v == "CRF_RANDE" else v
print(f"featurize {len(FEATS)} đoạn trong {time.time()-t0:.1f}s")
def evaluate_fn(preds, recs):
    return pm.evaluate(preds, recs)

# Biến thể NONCE (đổi tên thực thể bằng từ vô nghĩa, nhãn giữ nguyên): dev chấm công khai, test lưu kín
_raws = {sp: pc.load_jsonl(str(RAW[f"grids.v1.{sp}.json"])) for sp in ("dev", "test")}
_keep = {"dev": {r["paragraph_id"] for r in DEV}, "test": {r["paragraph_id"] for r in TEST}}
NONCE = pr.nonce_records({sp: [x for x in _raws[sp] if str(x["para_id"]) in _keep[sp]] for sp in _raws}, TOPICS)
_e_ok = PP3.get("status") == "ok"
FEATS_NONCE = {sp: [pp.featurize(tok, r, p3.logE_matrix(E_MODEL, embed_words, r, device=DEVICE) if _e_ok else None, tagger=TAGGER) for r in NONCE[sp]]
               for sp in NONCE}
print("NONCE:", {sp: len(v) for sp, v in NONCE.items()}, "| ví dụ:", NONCE["dev"][0]["entities"], "|", NONCE["dev"][0]["sentences"][0])
''')

md("## 8. Huấn luyện + chọn epoch trên DEV (5 biến thể × 3 seed); test dự đoán KÍN")
code(r'''
RESULTS = json.loads((STATE / "results.json").read_text()) if (STATE / "results.json").exists() else {}
PRED_DIR = OUT / "predictions"; PRED_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUT / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)
SEALED_DIR = OUT / "sealed_test_predictions"; SEALED_DIR.mkdir(exist_ok=True)

def dump_preds(path, preds):
    with open(path, "w", encoding="utf-8") as f:
        for pid, p in preds.items():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

for v in VARIANTS:
    for seed in SEEDS:
        key = f"DONE|{v}|{seed}"
        if key in RESULTS:
            print("bỏ qua (đã có):", key); continue
        if tdone() > MAX_HOURS:
            print("[DỪNG] vượt MAX_HOURS"); break
        CK = STATE / f"inprogress_{v}_seed{seed}.pt"          # checkpoint theo epoch → bị ngắt thì chạy lại sẽ tiếp tục
        model, hist, bests = pp.train_variant(make_encoder, tok, model_variant(v), feats_for(v, TRAIN), DEV, feats_for(v, DEV),
                                              evaluate_fn, seed=seed, epochs=EPOCHS, device=DEVICE, ckpt_path=str(CK),
                                              patience=PATIENCE, stop_after_epochs=STOP_AFTER)
        for nm, (state, best_ep) in bests.items():
            name = nm if v == "M0" else v
            model.load_state_dict(state)
            dev_preds = pp.predict_all(model, DEV, feats_for(v, DEV), DEVICE, constrained=(nm == "M0c"))
            test_preds = pp.predict_all(model, TEST, feats_for(v, TEST), DEVICE, constrained=(nm == "M0c"))
            dump_preds(PRED_DIR / f"dev_{name}_seed{seed}.jsonl", dev_preds)
            dump_preds(SEALED_DIR / f"test_{name}_seed{seed}.jsonl", test_preds)      # KHÔNG chấm ở đây
            res = pm.evaluate(dev_preds, DEV)
            RESULTS[f"{name}|{seed}"] = {"variant": name, "seed": seed, "best_epoch": best_ep, "history": hist,
                                         "dev": pm.summary_row(name, res), "_per_paragraph": res["_per_paragraph"]}
            if v != "CRF_RANDE":
                c = (nm == "M0c")
                dn = pp.predict_all(model, NONCE["dev"], FEATS_NONCE["dev"], DEVICE, constrained=c)
                RESULTS[f"{name}|{seed}"]["dev_nonce"] = pm.summary_row(name, pm.evaluate(dn, NONCE["dev"]))
                dump_preds(SEALED_DIR / f"test_nonce_{name}_seed{seed}.jsonl",
                           pp.predict_all(model, NONCE["test"], FEATS_NONCE["test"], DEVICE, constrained=c))
            print(f"==> {name} seed={seed}: best_ep={best_ep} dev PAIR_F1={res['PAIR@all']['F1']:.3f} @any_imp={res['PAIR@any_implicit']['F1']:.3f}")
            if seed == SEEDS[0]:          # checkpoint TỐT NHẤT (theo dev) của mọi hệ thống, seed đầu, fp16 ~250 MB/file
                torch.save({k: t.half() if t.is_floating_point() else t for k, t in state.items()}, CKPT_DIR / f"{name}_seed{seed}_best.pt")
        RESULTS[key] = {"done": True, "epochs_run": len(hist), "early_stopped": len(hist) < EPOCHS}
        (STATE / "results.json").write_text(json.dumps(RESULTS, default=str))
        pp.cleanup_ckpt(str(CK))        # kết quả đã lưu → xoá checkpoint đang-chạy + bản tốt nhất tạm (tránh phình Output)
        del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
''')

md("## 9. Bảng ablation trên DEV (trung bình ± độ lệch chuẩn qua seed) + kiểm định cặp")
code(r'''
SYS = ["M0", "M0c", "CRF_NOPAIR", "CRF_NOE", "CRF_RANDE", "CRF_FULL"]
LABEL = {"M0": "M0 độc lập", "M0c": "M0c = M0 + luật", "CRF_NOPAIR": "M3-no-pair (CRF)", "CRF_NOE": "M3-no-E",
         "CRF_RANDE": "M2r (E xáo trộn)", "CRF_FULL": "M3 (đầy đủ)"}
METRICS = ["PAIR_F1", "PAIR_F1@src_imp", "PAIR_F1@tgt_imp", "PAIR_F1@any_imp", "IMP_ACC_DESTROY", "IMP_ACC_CREATE", "F1_MOVE", "EXIST_ACC",
           "VIOL_per_entity", "PAIR_LOC_F1"]
ABL = []
for s in SYS:
    rows = [RESULTS[f"{s}|{seed}"] for seed in SEEDS if f"{s}|{seed}" in RESULTS]
    if not rows:
        continue
    r = {"system": LABEL[s], "code": s, "n_seeds": len(rows), "best_epochs": [x["best_epoch"] for x in rows]}
    for m in METRICS:
        vals = [x["dev"][m] for x in rows if x["dev"][m] == x["dev"][m]]
        r[m] = statistics.mean(vals) if vals else float("nan")
        r[m + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    ABL.append(r)
print(f"{'hệ thống':22s} " + " ".join(f"{m[:15]:>15s}" for m in METRICS))
for r in ABL:
    print(f"{r['system']:22s} " + " ".join(f"{r[m]:.3f}±{r[m+'_sd']:.3f}"[:15].rjust(15) for m in METRICS))

print("Khái quát (NONCE, dev): PAIR_F1 gốc → đổi tên (TB qua seed)")
NONCE_TAB = {}
for s_ in SYS:
    rows = [RESULTS[f"{s_}|{seed}"] for seed in SEEDS if f"{s_}|{seed}" in RESULTS and "dev_nonce" in RESULTS[f"{s_}|{seed}"]]
    if rows:
        a_ = statistics.mean(x["dev"]["PAIR_F1"] for x in rows); b_ = statistics.mean(x["dev_nonce"]["PAIR_F1"] for x in rows)
        ai = statistics.mean(x["dev"]["PAIR_F1@any_imp"] for x in rows); bi = statistics.mean(x["dev_nonce"]["PAIR_F1@any_imp"] for x in rows)
        NONCE_TAB[s_] = {"orig": a_, "nonce": b_, "orig_imp": ai, "nonce_imp": bi}
        print(f"  {LABEL[s_]:22s} {a_:.3f} → {b_:.3f}  | @any_imp {ai:.3f} → {bi:.3f}")

# Kiểm định cặp trên dev, gộp seed bằng cách so seed tương ứng; báo cáo seed đầu + trung bình Δ
PAIRS_TO_TEST = [("CRF_FULL", "M0c"), ("CRF_FULL", "CRF_NOE"), ("CRF_FULL", "CRF_RANDE"), ("CRF_NOE", "CRF_NOPAIR"),
                 ("CRF_NOPAIR", "M0c"), ("M0c", "M0")]
BOOT_DEV = {}
for a, b in PAIRS_TO_TEST:
    for sl in ("all", "any_implicit"):
        ds = []
        for seed in SEEDS:
            ka, kb = f"{a}|{seed}", f"{b}|{seed}"
            if ka in RESULTS and kb in RESULTS:
                ds.append(pm.paired_bootstrap_pair_f1({"_per_paragraph": RESULTS[ka]["_per_paragraph"]},
                                                      {"_per_paragraph": RESULTS[kb]["_per_paragraph"]}, sl, n_boot=2000, seed=SEED))
        if ds:
            BOOT_DEV[f"{a}-{b}@{sl}"] = {"delta_mean": statistics.mean(d["delta"] for d in ds),
                                         "seed0_ci": [ds[0]["ci_low"], ds[0]["ci_high"]], "n_seeds": len(ds)}
            x = BOOT_DEV[f"{a}-{b}@{sl}"]
            print(f"Δ {a} − {b} @{sl}: TB {x['delta_mean']:+.3f} (seed0 CI95 [{x['seed0_ci'][0]:+.3f}, {x['seed0_ci'][1]:+.3f}])")
''')

md("## 10. PP2 theo đúng công thức (11): lọc cặp bằng XÁC SUẤT tồn tại dự đoán, khảo sát ngưỡng τ1, τ2 trên DEV (H1)")
code(r"""
# C_t = {(X,Y): p(e_{X,t-1}=1) > τ1, p(e_{Y,t-1}=1) < τ2}; p lấy từ biên CRF (forward-backward) của mô hình đã huấn luyện.
TAU1 = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
TAU2 = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
PP2_RES, PP2_GRID = {}, {}
for s_ in ["M0c", "CRF_FULL" if (CKPT_DIR / f"CRF_FULL_seed{SEEDS[0]}_best.pt").exists() else "CRF_NOE"]:
    ck = CKPT_DIR / f"{s_}_seed{SEEDS[0]}_best.pt"
    if not ck.exists():
        continue
    enc, hid = make_encoder()
    mdl = pp.PP4Model(enc, hid, "M0" if s_ == "M0c" else model_variant(s_)).to(DEVICE)
    mdl.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in torch.load(ck, map_location="cpu").items()})
    mdl.eval()
    probs = {r["paragraph_id"]: mdl.existence_probs(f, DEVICE).cpu() for r, f in zip(DEV, feats_for(s_, DEV))}
    grid = {}
    for t1 in TAU1:
        for t2 in TAU2:
            kept = total = gold = gold_kept = 0
            for r in DEV:
                P_ = probs[r["paragraph_id"]]
                n, T = len(r["entities"]), len(r["sentences"])
                G = {(g["sentence_id"], g["source_idx"], g["target_idx"]) for g in pc.gold_pairs_report(r)}
                for t in range(1, T + 1):
                    for x in range(n):
                        for y in range(n):
                            if x == y: continue
                            total += 1
                            ok = bool(P_[t - 1, x] > t1 and P_[t - 1, y] < t2)
                            kept += ok
                            if (t, x, y) in G:
                                gold += 1; gold_kept += ok
            grid[(t1, t2)] = {"candidate_ratio": kept / max(total, 1), "gold_recall": gold_kept / max(gold, 1)}
    PP2_GRID[s_] = {f"{a}|{b}": v for (a, b), v in grid.items()}
    # điểm vận hành khoá trước: τ1 = 0.1, τ2 = 0.9 ('đặt lỏng để hạn chế bỏ sót'); H1 đạt nếu phủ ≥ 95% và giữ ≤ 50%
    op = grid[(0.1, 0.9)]
    PP2_RES[s_] = dict(op, H1_pass=op["gold_recall"] >= 0.95 and op["candidate_ratio"] <= 0.5, n_gold=gold)
    print(f"PP2 [{s_}] τ1=0.1, τ2=0.9: giữ {op['candidate_ratio']:.1%} ứng viên, phủ {op['gold_recall']:.1%} cặp vàng → H1 {'ĐẠT' if PP2_RES[s_]['H1_pass'] else 'CHƯA ĐẠT'}")
    del mdl; torch.cuda.empty_cache() if DEVICE == "cuda" else None
if PP2_GRID:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    for s_, col in zip(PP2_GRID, ["#2a78d6", "#6250d6"]):
        pts = sorted((v["candidate_ratio"], v["gold_recall"]) for v in PP2_GRID[s_].values())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=col, ms=5, label=s_, alpha=0.8)
    ax.axhline(0.95, color="#52514e", lw=1, ls="--"); ax.axvline(0.5, color="#52514e", lw=1, ls="--")
    ax.set_xlabel("tỉ lệ ứng viên còn lại"); ax.set_ylabel("độ phủ cặp vàng (dev)")
    ax.set_title("PP2: đánh đổi số cặp phải xét ↔ độ phủ (lưới τ1, τ2)", loc="left", fontsize=10); ax.legend(frameon=False)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    (OUT / "figures").mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(OUT / "figures" / "pp2_tradeoff.png"); plt.show(); plt.close(fig)
""")

md("## 11. CV 5 phần theo chủ đề: M3 vs M0c trên toàn bộ 488 đoạn (kiểm định chính cho lát ngầm)")
code(r'''
CV = json.loads((STATE / "cv.json").read_text()) if (STATE / "cv.json").exists() else {}
CV_EPOCHS = {}
for v in CV_VARIANTS:
    names = ["M0c"] if v == "M0" else [v]
    eps = [RESULTS[f"{names[0]}|{s}"]["best_epoch"] for s in SEEDS if f"{names[0]}|{s}" in RESULTS]
    CV_EPOCHS[v] = int(statistics.median(eps)) if eps else EPOCHS
print("Số epoch cố định cho CV (trung vị epoch tốt nhất trên dev):", CV_EPOCHS)
CV_RECS = [r for r in RECS if not LIMIT_TRAIN or r in TRAIN + DEV + TEST]
if RUN_CV:
    for v in CV_VARIANTS:
        for fold in range(5):
            key = f"DONE|{v}|fold{fold}"
            if key in CV or tdone() > MAX_HOURS:
                continue
            tr = [r for r in CV_RECS if FOLDS[r["paragraph_id"]] != fold]
            te = [r for r in CV_RECS if FOLDS[r["paragraph_id"]] == fold]
            if LIMIT_TRAIN:
                tr = tr[:LIMIT_TRAIN]
            # không chọn epoch trên fold kiểm tra: huấn luyện đúng CV_EPOCHS[v] epoch, lấy mô hình cuối
            CK = STATE / f"inprogress_cv_{v}_fold{fold}.pt"
            model, hist, _ = pp.train_variant(make_encoder, tok, model_variant(v), feats_for(v, tr), te[:1], feats_for(v, te[:1]),
                                              evaluate_fn, seed=SEEDS[0], epochs=CV_EPOCHS[v], device=DEVICE, ckpt_path=str(CK),
                                              stop_after_epochs=STOP_AFTER)
            for nm in (["M0", "M0c"] if v == "M0" else [v]):
                preds = pp.predict_all(model, te, feats_for(v, te), DEVICE, constrained=(nm == "M0c"))
                CV[f"{nm}|fold{fold}"] = preds
            CV[key] = {"done": True}
            (STATE / "cv.json").write_text(json.dumps(CV))
            pp.cleanup_ckpt(str(CK))
            print(f"CV {v} fold {fold}: xong ({len(te)} đoạn)")
            del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
CV_RES, CV_BOOT = {}, {}
for nm in ["M0", "M0c", "CRF_FULL" if "CRF_FULL" in CV_VARIANTS else "CRF_NOE"]:
    allp = {}
    for fold in range(5):
        allp.update(CV.get(f"{nm}|fold{fold}", {}))
    recs_cov = [r for r in CV_RECS if r["paragraph_id"] in allp]
    if len(recs_cov) == len(CV_RECS):
        CV_RES[nm] = pm.evaluate(allp, CV_RECS)
if len(CV_RES) >= 2:
    m3 = "CRF_FULL" if "CRF_FULL" in CV_RES else "CRF_NOE"
    for sl in ("all", "src_implicit", "tgt_implicit", "any_implicit"):
        CV_BOOT[f"{m3}-M0c@{sl}"] = pm.paired_bootstrap_pair_f1(CV_RES[m3], CV_RES["M0c"], sl, n_boot=N_BOOT, seed=SEED)
    for nm, res in CV_RES.items():
        print(f"CV {nm:10s} PAIR_F1={res['PAIR@all']['F1']:.3f} @src_imp={res['PAIR@src_implicit']['F1']:.3f} "
              f"@tgt_imp={res['PAIR@tgt_implicit']['F1']:.3f} @any_imp={res['PAIR@any_implicit']['F1']:.3f} "
              f"IMP_D={res['IMPLICIT_ACC']['DESTROY']['acc']:.3f} IMP_C={res['IMPLICIT_ACC']['CREATE']['acc']:.3f} VIOL={res['VIOLATION_RATE_per_entity']:.3f}")
    for k, b in CV_BOOT.items():
        print(f"Δ CV {k}: {b['delta']:+.3f} CI95 [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] (n={b['n_paragraphs']} đoạn)")
else:
    print("CV chưa đủ (RUN_CV tắt hoặc vượt MAX_HOURS).")
''')

md("## 12. Biểu đồ: loss train/dev, F1 dev theo epoch, ablation")
code(r'''
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
COL = {"M0": "#2a78d6", "CRF_NOPAIR": "#eb6834", "CRF_NOE": "#1baf7a", "CRF_RANDE": "#eda100", "CRF_FULL": "#6250d6"}
def _style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(True, color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
for v in VARIANTS:
    k = f"{'M0' if v == 'M0' else v}|{SEEDS[0]}"
    if k not in RESULTS: continue
    h = RESULTS[k]["history"]; ep = [x["epoch"] for x in h]
    axes[0].plot(ep, [x["train_loss"] for x in h], color=COL[v], lw=2, label=v)
    axes[1].plot(ep, [x["dev_loss"] for x in h], color=COL[v], lw=2, label=v)
    f1k = "M0c:dev_PAIR_F1" if v == "M0" else f"{model_variant(v)}:dev_PAIR_F1"
    axes[2].plot(ep, [x[f1k] for x in h], color=COL[v], lw=2, label=("M0c" if v == "M0" else v))
for ax, t in zip(axes, ["Loss train (theo ô)", "Loss dev (theo ô)", "PAIR_F1 dev"]):
    ax.set_title(t, loc="left", fontsize=10); ax.set_xlabel("epoch"); _style(ax)
axes[2].legend(frameon=False, fontsize=8)
fig.suptitle(f"Huấn luyện PP4, seed {SEEDS[0]} (M0: CE; CRF: −log p(y_vàng)/số ô, nên không so trực tiếp độ lớn loss giữa M0 và CRF)", fontsize=9, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "pp4_training_curves.png"); plt.show(); plt.close(fig)

if ABL:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ms = ["PAIR_F1", "PAIR_F1@src_imp", "PAIR_F1@tgt_imp", "PAIR_F1@any_imp"]
    w = 0.8 / len(ABL)
    for k, r in enumerate(ABL):
        xs = [j + k * w for j in range(len(ms))]
        ax.bar(xs, [r[m] for m in ms], width=w * 0.92, yerr=[r[m + "_sd"] for m in ms], capsize=2,
               color=["#2a78d6", "#8fb8ea", "#eb6834", "#1baf7a", "#eda100", "#6250d6"][k], label=r["system"])
    ax.set_xticks([j + 0.4 - w / 2 for j in range(len(ms))], ["PAIR_F1", "@nguồn ngầm", "@đích ngầm", "@có phía ngầm"])
    ax.set_title("Ablation trên dev (TB ± SD qua seed)", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, ncol=3); _style(ax)
    fig.tight_layout(); fig.savefig(FIG / "pp4_ablation_dev.png"); plt.show(); plt.close(fig)
if PP3.get("history"):
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    h = PP3["history"]
    ax.plot([x["epoch"] for x in h], [x["train_loss"] for x in h], color="#2a78d6", lw=2, label="train")
    ax.plot([x["epoch"] for x in h], [x["val_loss"] for x in h], color="#eb6834", lw=2, label="validation")
    ax.set_title("PP3: BCE của E(X,Y)", loc="left", fontsize=10); ax.legend(frameon=False); _style(ax)
    fig.tight_layout(); fig.savefig(FIG / "pp3_loss.png"); plt.show(); plt.close(fig)
''')

md("## 13. Lưu đầu ra + gửi lại")
code(r'''
def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
CV_SUMMARY = {nm: pm.summary_row(nm, r) for nm, r in CV_RES.items()}
SUMMARY = {"run_tag": RUN_TAG, "env": ENV, "encoder": f"{ENCODER_ID}@{ENCODER_REV}", "variants": VARIANTS, "seeds": SEEDS, "epochs": EPOCHS,
           "pp3": {k: v for k, v in PP3.items() if k != "history"}, "pp3_history": PP3.get("history"),
           "ablation_dev": ABL, "nonce_dev": NONCE_TAB, "bootstrap_dev": BOOT_DEV, "pp2_predicted": PP2_RES, "pp2_grid": PP2_GRID, "location_candidate_cover": LOC_COVER, "cv_epochs": CV_EPOCHS, "cv": CV_SUMMARY,
           "cv_bootstrap": CV_BOOT, "histories": {k: v["history"] for k, v in RESULTS.items() if "history" in v and k.endswith(f"|{SEEDS[0]}")},
           "sealed_test_files": sorted(p.name for p in SEALED_DIR.glob("*.jsonl")), "runtime_h": round(tdone(), 2),
           "restored_files": len(RESTORED), "early_stopping": {k: v for k, v in RESULTS.items() if k.startswith("DONE|")},
           "checkpoints": sorted(p.name for p in CKPT_DIR.glob("*.pt")),
           "versions": {"core": pc.CORE_VERSION, "metrics": pm.METRICS_VERSION, "pp3": p3.PP3_VERSION, "pp4": pp.PP4_VERSION}}
jdump(SUMMARY, OUT / "summary_05.json")
if PP3.get("status") == "ok":
    jdump({"top_pairs": PAIR_CNT.most_common(5000), "coverage": {k: v for k, v in PP3.items() if k.startswith("coverage")}}, OUT / "pp3_pairs_top5000.json")
    jdump(LOGE, OUT / "pp3_logE_propara.json")
ZIP = WORK_ROOT / f"{RUN_TAG}.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "code" not in p.parts and "raw" not in p.parts and p.suffix not in (".pt", ".tmp"):
            z.write(p, Path(RUN_TAG) / p.relative_to(OUT))
print("=== GỬI LẠI ===")
print(f"notebook=05_pp3_pp4 gpus={ENV['gpus']} runtime={SUMMARY['runtime_h']}h seeds={SEEDS} epochs<={EPOCHS} patience={PATIENCE} khoi_phuc={len(RESTORED)} file")
print("đã xong:", sorted(k[5:] for k in RESULTS if k.startswith("DONE|")), "| CV:", sorted(k[5:] for k in CV if k.startswith("DONE|")))
print("checkpoint tốt nhất:", SUMMARY["checkpoints"])
print("PP3:", PP3.get("status"), {k: PP3.get(k) for k in ("n_pairs_distinct", "n_pair_mentions", "coverage_all_min1", "propara_auc_dev", "best_val")})
for r in ABL:
    print(f"  DEV {r['system']:20s} PAIR_F1={r['PAIR_F1']:.3f}±{r['PAIR_F1_sd']:.3f} @any_imp={r['PAIR_F1@any_imp']:.3f}±{r['PAIR_F1@any_imp_sd']:.3f} LOC={r['PAIR_LOC_F1']:.3f} "
          f"IMP_D={r['IMP_ACC_DESTROY']:.3f} IMP_C={r['IMP_ACC_CREATE']:.3f} VIOL={r['VIOL_per_entity']:.3f} ep={r['best_epochs']}")
for k, x in BOOT_DEV.items():
    print(f"  Δdev {k}: {x['delta_mean']:+.3f} seed0CI {x['seed0_ci'][0]:+.3f}..{x['seed0_ci'][1]:+.3f}")
print("  NONCE dev:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in NONCE_TAB.items()})
print("  PP2 (τ1=0.1, τ2=0.9):", PP2_RES, "| phủ vị trí ứng viên:", LOC_COVER)
for nm, r in CV_SUMMARY.items():
    print(f"  CV {nm:10s} PAIR_F1={r['PAIR_F1']:.3f} @src={r['PAIR_F1@src_imp']:.3f} @tgt={r['PAIR_F1@tgt_imp']:.3f} @any={r['PAIR_F1@any_imp']:.3f}")
for k, b in CV_BOOT.items():
    print(f"  ΔCV {k}: {b['delta']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]")
print("=== HẾT ===")
''')

out = C.save_notebook(cells, ROOT / "notebooks" / "05_pp3_pp4_dev_cv.ipynb", gpu=True)
print("Đã ghi", out, "| số ô:", len(cells))
