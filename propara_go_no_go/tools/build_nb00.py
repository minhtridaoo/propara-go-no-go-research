"""Sinh notebooks/00_propara_data_audit.ipynb bằng nbformat.

Mã lõi (src/propara_core.py, src/propara_core_tests.py) được NHÚNG nguyên văn vào notebook
để notebook tự chạy độc lập trên Kaggle, rồi ghi lại thành file .py trong output cho các notebook sau.

Cách dùng:  python tools/build_nb00.py <thư_mục_dự_án>
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
CORE = (ROOT / "src" / "propara_core.py").read_text(encoding="utf-8")
TESTS = (ROOT / "src" / "propara_core_tests.py").read_text(encoding="utf-8")
assert "'''" not in CORE and "'''" not in TESTS, "mã lõi không được chứa ''' (dùng để nhúng)"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =============================================================================
md(r"""
# 00 — Kiểm tra dữ liệu ProPara (data audit) · `propara_audit_v1`

**Vị trí trong pipeline:** Giai đoạn 1 (D1) của kế hoạch Go/No-Go. Là notebook đầu tiên; **01 (bộ chấm + heuristic) và mọi notebook Qwen đều phụ thuộc vào output của notebook này.**

**Mục tiêu:** tải ProPara gốc (bản EMNLP'18 `grids.v1` mà báo cáo dùng), chuẩn hoá, suy nhãn CREATE/DESTROY/MOVE/NONE, cờ nhắc, thay đổi ngầm, 533 cặp vàng, điểm quyết định ngầm. Sau đó **đối chiếu từng con số trong báo cáo 16/10/2026**, đối chiếu với đáp án của bộ chấm chính thức (`aristo-leaderboard`) và xuất dữ liệu chuẩn có checksum.

| Mục | Giá trị |
|---|---|
| Accelerator | **None (CPU)**. Không cần GPU, nên **tắt GPU** để giữ quota |
| Internet | **On** (để tải dữ liệu từ GitHub, ghim theo commit). Nếu không bật được thì dùng phương án Dataset (xem dưới) |
| Input | Không bắt buộc. Phương án dự phòng: Dataset riêng chứa `propara-raw-v1.zip` (đã giải nén) |
| Thư viện | Chỉ thư viện chuẩn của Python + `matplotlib` (Kaggle có sẵn). Không cần `pip install` |
| Thời gian | Toàn bộ dữ liệu chỉ ~1 MB; trên máy CPU thử nghiệm, notebook chạy xong dưới 1 phút (chưa đo trên Kaggle) |
| Output | `/kaggle/working/propara_audit_v1/` và `/kaggle/working/propara_audit_v1.zip` |

**Cách chạy:** `Run All`. Đọc bảng **CỔNG NGHIỆM THU** ở ô cuối. Nếu ghi `GATE_STATUS = PASS` thì `Save Version` → `Save & Run All (Commit)`.

**Điều kiện thành công:** không có dòng `FAIL`. Các dòng `WARN` là phát hiện cần ghi nhận, **không chặn** bước tiếp theo.

**Gửi lại cho trợ lý:** toàn bộ output của **ô cuối cùng** (khối `=== GỬI LẠI ===`) và file `propara_audit_v1/reports/audit_summary.json`.

> Quy ước trong notebook: `sentence_id` đánh số từ 1; `states[i][0]` là trạng thái ban đầu, `states[i][t]` là trạng thái **sau** câu *t*. Thực thể được định danh bằng **chỉ số** (không bằng tên), vì ProPara có thực thể trùng tên.
""")

# =============================================================================
md("## 1. Cấu hình")
code(r'''
import os, sys, json, time, platform, shutil, hashlib, csv, zipfile, random, statistics, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SMOKE_TEST     = False   # True: chỉ chạy 15 đoạn/tập để thử luồng (bỏ qua đối chiếu số liệu báo cáo)
ALLOW_DOWNLOAD = True    # cho phép tải từ GitHub nếu không thấy dữ liệu trong /kaggle/input
SEED           = 2026
RUN_TAG        = "propara_audit_v1"

# Biến môi trường chỉ dùng khi chạy NGOÀI Kaggle (kiểm thử cục bộ); trên Kaggle để mặc định.
INPUT_ROOT = Path(os.environ.get("PROPARA_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT  = Path(os.environ.get("PROPARA_WORK_ROOT", "/kaggle/working"))
OUT = WORK_ROOT / RUN_TAG

PIN_PROPARA = "51a3cccbedcbe5f92276f9925e0ce1e6ec4247bb"   # allenai/propara (2019-08-30)
PIN_LEADER  = "60d5be31b5a7d36f29d13223ece29a8d2bfa8b5f"   # allenai/aristo-leaderboard (2021-05-17)
_GH = "https://raw.githubusercontent.com/allenai"

# tên file cục bộ -> (URL ghim commit, sha256, các hậu tố đường dẫn chấp nhận khi dò /kaggle/input)
RAW_FILES = {
    "grids.v1.train.json": (f"{_GH}/propara/{PIN_PROPARA}/data/emnlp18/grids.v1.train.json",
        "6dc048d09c532ef906ba37d5ab814f54c54e56f230184029912eabe415de225d", ["grids.v1.train.json"]),
    "grids.v1.dev.json": (f"{_GH}/propara/{PIN_PROPARA}/data/emnlp18/grids.v1.dev.json",
        "580ddc66d86a89c747d3e7f2da5cca0eee95444cca66b60b345fe29c93c3fef6", ["grids.v1.dev.json"]),
    "grids.v1.test.json": (f"{_GH}/propara/{PIN_PROPARA}/data/emnlp18/grids.v1.test.json",
        "cf239a45d3eb203aba09ec392cffc81812e5c6fde69d782b01d068efd9e6f370", ["grids.v1.test.json"]),
    "naacl18.partition.paraid.tsv": (f"{_GH}/propara/{PIN_PROPARA}/data/emnlp18/naacl18.partition.paraid.tsv",
        "9fecf07296c58b195ae778943fc0b4d38ea489ec71e19045bd8730460175de44", ["naacl18.partition.paraid.tsv"]),
    "gold-full-grids.v3.tsv": (f"{_GH}/propara/{PIN_PROPARA}/data/naacl18/gold-full-grids.v3.tsv",
        "8eff6e8fb066869d9d0c0fe18e3d994eb3078caa0a3a1be98f716e8a79dbe0aa", ["gold-full-grids.v3.tsv"]),
    "leaderboard_answers_train.tsv": (f"{_GH}/aristo-leaderboard/{PIN_LEADER}/propara/data/train/answers.tsv",
        "2f2b014aa071583f9180cd8487c855438821abe1c9aae5474ac685854934e8cd", ["leaderboard_answers_train.tsv", "train/answers.tsv"]),
    "leaderboard_answers_dev.tsv": (f"{_GH}/aristo-leaderboard/{PIN_LEADER}/propara/data/dev/answers.tsv",
        "a496afe5bb2b81ad79b0da7a935f81d5d0a6b4cefc89d865ba09fece38a30217", ["leaderboard_answers_dev.tsv", "dev/answers.tsv"]),
    "leaderboard_answers_test.tsv": (f"{_GH}/aristo-leaderboard/{PIN_LEADER}/propara/data/test/answers.tsv",
        "cf1a71f38ccad7609e54b1dcf5cbc6e342ea3d2513490c0c09883d5bd80625d6", ["leaderboard_answers_test.tsv", "test/answers.tsv"]),
}
SPLITS = ("train", "dev", "test")

# Con số trong báo cáo 16/10/2026 (trang ghi kèm), dùng để đối chiếu.
REPORT = {
    "n_paragraphs": 488, "n_sentences": 3302, "n_entities": 1915,                       # tr.1
    "table1": {  # (ngầm, tổng) theo tập và nhãn — tr.2, Bảng 1
        "train": {"CREATE": (78, 747), "DESTROY": (81, 502), "MOVE": (355, 1211)},
        "dev":   {"CREATE": (11, 85),  "DESTROY": (15, 68),  "MOVE": (41, 157)},
        "test":  {"CREATE": (15, 128), "DESTROY": (22, 121), "MOVE": (53, 152)},
    },
    "gold_pairs": 533, "gold_pairs_split": {"train": 378, "dev": 58, "test": 97},       # tr.2, (3)
    "decision_points": {"DESTROY": 57, "CREATE": 76},                                     # tr.2
    "avg_candidates": {"DESTROY": 2.6, "CREATE": 2.9},                                    # tr.3
    "random_acc_pct": {"DESTROY": 51, "CREATE": 40},                                      # tr.3, Bảng 3
    "lift": {  # (lift, số ô) — tr.2, Bảng 2; hàng = nhãn ngầm A, cột = tín hiệu B
        ("CREATE", "CREATE"): (0.3, 10), ("CREATE", "DESTROY"): (6.6, 63), ("CREATE", "MOVE"): (0.3, 10),
        ("DESTROY", "CREATE"): (9.0, 85), ("DESTROY", "DESTROY"): (1.4, 19), ("DESTROY", "MOVE"): (0.0, 2),
        ("MOVE", "CREATE"): (1.0, 97), ("MOVE", "DESTROY"): (1.3, 69), ("MOVE", "MOVE"): (1.2, 141),
    },
    "pp1_label_questions": 52936, "pp1_pair_questions": 48070,                            # tr.4, (6)(7)
    "pp2_kept": 11144, "pp2_gold_kept": 533,                                              # tr.4, (12)
    "cooc_gold_pct": 74.7, "cooc_src_implicit_pct": 18.9,                                 # tr.4
    "mean_entities": 3.9,                                                                 # tr.6
}
# Số đoạn theo tập lấy từ file phân chia chính thức (báo cáo không nêu)
EXPECTED_SPLIT_PARAS = {"train": 391, "dev": 43, "test": 54}

random.seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)
for sub in ["raw", "data/processed", "data/splits", "data/audit", "reports/figures"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
T0 = time.time()
print("OUT =", OUT, "| SMOKE_TEST =", SMOKE_TEST)
''')

# =============================================================================
md("## 2. Kiểm tra môi trường")
code(r'''
def _internet_ok(url="https://raw.githubusercontent.com", timeout=6):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception as e:  # mạng tắt hoặc bị chặn
        return f"không truy cập được: {type(e).__name__}"

ENV = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "on_kaggle": Path("/kaggle").exists() and "PROPARA_INPUT_ROOT" not in os.environ,
    "cpu_count": os.cpu_count(),
    "disk_free_gb": round(shutil.disk_usage(WORK_ROOT).free / 1e9, 1),
    "gpu_present": bool(shutil.which("nvidia-smi")),
    "internet": _internet_ok(),
}
try:
    with open("/proc/meminfo") as f:
        ENV["ram_total_gb"] = round(int(f.readline().split()[1]) / 1e6, 1)
except OSError:
    ENV["ram_total_gb"] = None
try:
    import matplotlib
    ENV["matplotlib"] = matplotlib.__version__
except ImportError:
    ENV["matplotlib"] = None
for k, v in ENV.items():
    print(f"{k:14s}: {v}")
if ENV["gpu_present"]:
    print("\n[CẢNH BÁO] Đang bật GPU. Notebook này chỉ cần CPU, nên tắt GPU (Settings → Accelerator → None) để giữ quota.")
if ENV["disk_free_gb"] < 1:
    raise RuntimeError("Dung lượng trống < 1 GB trong thư mục làm việc.")
''')

# =============================================================================
md("## 3. Mô-đun lõi `propara_core` (nhúng nguyên văn, ghi ra output để notebook sau dùng lại)")
code("CORE_SRC = r'''\n" + CORE + "'''\n\n"
     "TESTS_SRC = r'''\n" + TESTS + "'''\n\n"
     r'''(OUT / "propara_core.py").write_text(CORE_SRC, encoding="utf-8")
(OUT / "propara_core_tests.py").write_text(TESTS_SRC, encoding="utf-8")
sys.dont_write_bytecode = True   # không để __pycache__ lọt vào output
sys.path.insert(0, str(OUT))
import importlib
import propara_core as pc
import propara_core_tests as pct
importlib.reload(pc); importlib.reload(pct)
print("propara_core", pc.CORE_VERSION, "| sha256", hashlib.sha256(CORE_SRC.encode()).hexdigest()[:16])
''')

# =============================================================================
md("## 4. Kiểm thử đơn vị (phải đạt hết trước khi đụng vào dữ liệu)")
code(r'''
n_tests, failed_tests = pct.run_all(verbose=True)
print(f"\n{n_tests - len(failed_tests)}/{n_tests} kiểm thử đạt")
if failed_tests:
    raise RuntimeError(f"Kiểm thử đơn vị thất bại: {failed_tests}")
''')

# =============================================================================
md("## 5. Nạp dữ liệu gốc: dò `/kaggle/input`, nếu không có thì tải theo commit ghim; kiểm tra SHA-256")
code(r'''
def find_in_input(suffixes):
    if not INPUT_ROOT.exists():
        return None
    for suf in suffixes:
        base = suf.split("/")[-1]
        for p in sorted(INPUT_ROOT.rglob(base)):
            if p.is_file() and p.as_posix().endswith(suf):
                return p
    return None

RAW, RAW_SOURCE, CHECKSUM = {}, {}, {}
for name, (url, sha, suffixes) in RAW_FILES.items():
    p = find_in_input(suffixes)
    if p is not None:
        RAW_SOURCE[name] = f"input:{p}"
    else:
        p = OUT / "raw" / name
        if not p.exists():
            if not ALLOW_DOWNLOAD or ENV["internet"] is not True:
                raise FileNotFoundError(
                    f"Không tìm thấy {name} trong {INPUT_ROOT} và không tải được (internet={ENV['internet']}).\n"
                    "Cách sửa: bật Settings → Internet → On, HOẶC Add Input dataset chứa propara-raw-v1.zip đã giải nén.")
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(url, p)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"Tải thất bại {url}: {e}")
                    time.sleep(2)
        RAW_SOURCE[name] = f"download:{url}"
    RAW[name] = p
    got = pc.sha256_file(str(p))
    CHECKSUM[name] = {"expected": sha, "observed": got, "match": got == sha}
    print(f"{'OK ' if got == sha else 'LỆCH'}  {name:32s} {RAW_SOURCE[name][:90]}")
''')

# =============================================================================
md("## 6. Chuẩn hoá và kiểm tra cấu trúc")
code(r'''
TOPICS = pc.load_topics_from_naacl_grid(str(RAW["gold-full-grids.v3.tsv"]))
PARTITION = {}
with open(RAW["naacl18.partition.paraid.tsv"], encoding="utf-8") as f:
    for line in f:
        if line.strip():
            sp, pid = line.split("\t")[:2]
            PARTITION[pid.strip()] = sp.strip()

RAWS = {sp: pc.load_jsonl(str(RAW[f"grids.v1.{sp}.json"])) for sp in SPLITS}
if SMOKE_TEST:
    RAWS = {sp: rows[:15] for sp, rows in RAWS.items()}
RECS = [pc.build_record(r, sp, TOPICS.get(str(r["para_id"]))) for sp in SPLITS for r in RAWS[sp]]
BY_SPLIT = {sp: [r for r in RECS if r["split"] == sp] for sp in SPLITS}

# --- kiểm tra cấu trúc từng đoạn
ISSUES = [(r["paragraph_id"], msg) for r in RECS for msg in pc.validate_record(r)]
DUP_ISSUES = [x for x in ISSUES if "trùng tên" in x[1]]
HARD_ISSUES = [x for x in ISSUES if "trùng tên" not in x[1]]

# --- trùng lặp / rò rỉ giữa các tập
pid_split = defaultdict(set)
for r in RECS:
    pid_split[r["paragraph_id"]].add(r["split"])
PID_DUP_WITHIN = [p for p, c in Counter(r["paragraph_id"] for r in RECS).items() if c > 1]
PID_CROSS = [p for p, s in pid_split.items() if len(s) > 1]
PARTITION_MISMATCH = [r["paragraph_id"] for r in RECS if PARTITION.get(r["paragraph_id"]) != r["split"]]
topic_split = defaultdict(set)
for r in RECS:
    topic_split[r["topic"]].add(r["split"])
TOPIC_CROSS = [t for t, s in topic_split.items() if len(s) > 1]
NO_TOPIC = [r["paragraph_id"] for r in RECS if not r["topic"]]
text_split = defaultdict(set)
sent_split = defaultdict(set)
for r in RECS:
    text_split[" ".join(r["sentences"]).strip().lower()].add(r["split"])
    for s in r["sentences"]:
        sent_split[s.strip().lower()].add(r["split"])
TEXT_CROSS = [t for t, s in text_split.items() if len(s) > 1]
SENT_CROSS = sorted(t for t, s in sent_split.items() if len(s) > 1)

# --- ràng buộc tồn tại trên nhãn vàng (phải = 0 theo cấu tạo)
GOLD_VIOL = sum(pc.count_constraint_violations(r["labels"], [e[0] for e in r["existence"]])[0] for r in RECS)

COUNTS = {sp: {"paragraphs": len(BY_SPLIT[sp]),
               "sentences": sum(len(r["sentences"]) for r in BY_SPLIT[sp]),
               "entities": sum(len(r["entities"]) for r in BY_SPLIT[sp])} for sp in SPLITS}
COUNTS["all"] = {k: sum(COUNTS[sp][k] for sp in SPLITS) for k in ("paragraphs", "sentences", "entities")}
print(json.dumps(COUNTS, indent=1))
print("Lỗi cấu trúc nghiêm trọng:", len(HARD_ISSUES), HARD_ISSUES[:5])
print("Thực thể trùng tên:", DUP_ISSUES)
print("Trùng mã đoạn trong/giữa tập:", PID_DUP_WITHIN, PID_CROSS, "| lệch file phân chia:", PARTITION_MISMATCH[:5])
print("Chủ đề xuất hiện ở >1 tập:", len(TOPIC_CROSS), "| đoạn thiếu chủ đề:", len(NO_TOPIC), "| số chủ đề:", len(topic_split))
print("Đoạn văn trùng nguyên văn giữa các tập:", len(TEXT_CROSS), "| câu trùng nguyên văn giữa các tập:", len(SENT_CROSS))
print("Vi phạm ràng buộc tồn tại trên nhãn vàng:", GOLD_VIOL)
''')

# =============================================================================
md("## 7. Đối chiếu với đáp án của bộ chấm chính thức (`aristo-leaderboard/propara/data/*/answers.tsv`)")
code(r'''
ANSWER_CMP = {}
for sp in SPLITS:
    official = pc.load_action_file(str(RAW[f"leaderboard_answers_{sp}.tsv"]))
    if SMOKE_TEST:
        keep = {r["paragraph_id"] for r in BY_SPLIT[sp]}
        official = [x for x in official if x[0] in keep]
    derived = [row for r in BY_SPLIT[sp] for row in pc.to_action_rows(r)]
    ANSWER_CMP[sp] = pc.compare_action_rows(official, derived)
    print(sp, ANSWER_CMP[sp])
''')

# =============================================================================
md("## 8. Suy nhãn và thay đổi ngầm (đối chiếu Bảng 1)")
code(r'''
TABLE1 = {sp: {} for sp in SPLITS}
for sp in SPLITS:
    for lab in ("CREATE", "DESTROY", "MOVE", "NONE"):
        tot = imp = 0
        for r in BY_SPLIT[sp]:
            for i in range(len(r["entities"])):
                for t, y in enumerate(r["labels"][i]):
                    if y == lab:
                        tot += 1
                        imp += int(r["implicit"][i][t])
        TABLE1[sp][lab] = (imp, tot)
print(f"{'tập':6s} {'nhãn':8s} {'ngầm/tổng (đo)':>16s} {'báo cáo':>12s}")
for sp in SPLITS:
    for lab in ("CREATE", "DESTROY", "MOVE"):
        imp, tot = TABLE1[sp][lab]
        rep = REPORT["table1"][sp][lab]
        print(f"{sp:6s} {lab:8s} {f'{imp}/{tot} ({imp/max(tot,1):.0%})':>16s} {f'{rep[0]}/{rep[1]}':>12s} {'✓' if (imp, tot) == rep else '✗'}")
''')

# =============================================================================
md("## 9. Cặp vàng (định nghĩa báo cáo) và conversion theo bộ chấm chính thức")
code(r'''
PAIRS = [p for r in RECS for p in pc.gold_pairs_report(r)]
PAIRS_SPLIT = Counter(p["split"] for p in PAIRS)
OFFICIAL_CONV = [dict(c, paragraph_id=r["paragraph_id"], split=r["split"]) for r in RECS for c in pc.official_q3_conversions(r)]
OFFICIAL_SPLIT = Counter(c["split"] for c in OFFICIAL_CONV)
DUP_NAME_PIDS = sorted({pid for pid, _ in DUP_ISSUES})

shape = Counter()
for r in RECS:
    for t in range(1, len(r["sentences"]) + 1):
        D, C = pc.destroyed_created_at(r, t)
        if D and C:
            shape[(len(D), len(C))] += 1
PAIR_STATS = {
    "pairs_by_split": dict(PAIRS_SPLIT),
    "source_implicit": sum(p["source_implicit"] for p in PAIRS),
    "target_implicit": sum(p["target_implicit"] for p in PAIRS),
    "any_side_implicit": sum(p["source_implicit"] or p["target_implicit"] for p in PAIRS),
    "both_sides_implicit": sum(p["source_implicit"] and p["target_implicit"] for p in PAIRS),
    "sentences_with_pairs": sum(shape.values()),
    "sentences_1x1": shape[(1, 1)],
    "pairs_from_multi_entity_sentences": sum(p["n_destroy_in_sentence"] * p["n_create_in_sentence"] > 1 for p in PAIRS),
    "sentence_shapes_DxC": {f"{d}x{c}": v for (d, c), v in sorted(shape.items())},
    "official_q3_conversions_replica_by_split": dict(OFFICIAL_SPLIT),
    "official_q3_rules": dict(Counter(c["rule"] for c in OFFICIAL_CONV)),
}
for sp in SPLITS:
    PAIR_STATS[f"pair_slices_{sp}"] = {
        "n": PAIRS_SPLIT[sp],
        "source_implicit": sum(p["source_implicit"] for p in PAIRS if p["split"] == sp),
        "target_implicit": sum(p["target_implicit"] for p in PAIRS if p["split"] == sp),
        "any_side_implicit": sum(p["source_implicit"] or p["target_implicit"] for p in PAIRS if p["split"] == sp),
    }
print(json.dumps(PAIR_STATS, indent=1, ensure_ascii=False))
print("\nĐoạn có thực thể trùng tên NGUYÊN VĂN (evaluator chính thức gộp làm một):", DUP_NAME_PIDS)
''')

# =============================================================================
md("## 10. Điểm quyết định ngầm, PP2, lọc đồng xuất hiện, lift (Bảng 2)")
code(r'''
DPS = [d for r in RECS for d in pc.decision_points(r)]
DP_STATS = {}
for lab in ("DESTROY", "CREATE"):
    x = [d for d in DPS if d["label"] == lab]
    DP_STATS[lab] = {"n": len(x),
                     "by_split": dict(Counter(d["split"] for d in x)),
                     "avg_candidates": round(statistics.mean(d["n_candidates"] for d in x), 3) if x else None,
                     "random_acc": round(statistics.mean(d["random_acc"] for d in x), 4) if x else None}
PP2 = Counter()
for r in RECS:
    PP2.update(pc.pp2_counts(r))
COOC = Counter()
for r in RECS:
    COOC.update(pc.cooccurrence_filter_coverage(r))
COOC_PCT = {
    "same_sentence_gold": 100 * COOC["same_sentence"] / max(COOC["gold"], 1),
    "same_sentence_src_implicit": 100 * COOC["same_sentence_src_implicit"] / max(COOC["gold_src_implicit"], 1),
    "any_sentence_gold": 100 * COOC["any_sentence"] / max(COOC["gold"], 1),
    "any_sentence_src_implicit": 100 * COOC["any_sentence_src_implicit"] / max(COOC["gold_src_implicit"], 1),
}
LIFT = pc.lift_table(RECS)
print("Điểm quyết định:", json.dumps(DP_STATS, ensure_ascii=False))
print("PP1/PP2:", dict(PP2), f"| tỉ lệ giữ = {PP2['kept_gold_existence']/max(PP2['all_pairs'],1):.1%}")
print("Lọc đồng xuất hiện (%):", {k: round(v, 1) for k, v in COOC_PCT.items()})
print("\nLift (đo | báo cáo):")
for A in ("CREATE", "DESTROY", "MOVE"):
    row = []
    for B in ("CREATE", "DESTROY", "MOVE"):
        v = LIFT.get((A, B), {"lift": float("nan"), "n_implicit_A_with_B": 0})
        rep = REPORT["lift"][(A, B)]
        row.append(f"{v['lift']:.1f}({v['n_implicit_A_with_B']}) | {rep[0]}({rep[1]})")
    print(f"  {A:8s}", "   ".join(row))
''')

# =============================================================================
md("## 11. Độ nhạy của cờ nhắc và khả thi của PP4 (giải mã chính xác)")
code(r'''
# (a) Cờ nhắc kiểu 'extended' (bắt thêm -es/-ies) thay đổi bao nhiêu ô ngầm? Chỉ để ghi nhận độ nhạy.
SENS = Counter()
for sp in SPLITS:
    for raw in RAWS[sp]:
        rx = pc.build_record(raw, sp, mention_mode="extended")
        rr = next(r for r in BY_SPLIT[sp] if r["paragraph_id"] == str(raw["para_id"]))
        for i in range(len(rr["entities"])):
            for t in range(len(rr["labels"][i])):
                if rr["implicit"][i][t] != rx["implicit"][i][t]:
                    SENS[(sp, rr["labels"][i][t])] += 1
SENS_D = {f"{sp}/{lab}": v for (sp, lab), v in sorted(SENS.items())}
print("Số ô ngầm thay đổi khi dùng cờ nhắc 'extended':", SENS_D or "0")

# (b) Số thực thể/đoạn → kích thước không gian giải mã chính xác 4^n (cận trên trong báo cáo, mục 4.4)
NS = [len(r["entities"]) for r in RECS]
TS = [len(r["sentences"]) for r in RECS]
PP4 = {"mean_entities": round(statistics.mean(NS), 3), "max_entities": max(NS),
       "entities_hist": dict(sorted(Counter(NS).items())), "mean_sentences": round(statistics.mean(TS), 3),
       "max_sentences": max(TS), "max_states_per_step_4^n": 4 ** max(NS),
       "max_T_x_4^n": max(len(r["sentences"]) * 4 ** len(r["entities"]) for r in RECS),
       "paragraphs_n_gt_8": sum(n > 8 for n in NS)}
print(json.dumps(PP4, indent=1))
''')

# =============================================================================
md("## 12. Chia chéo 5 phần theo chủ đề (khoá trước) và mẫu kiểm tra nhãn vàng cho người chấm")
code(r'''
FOLDS = pc.cv_folds_by_topic(RECS, k=5, seed=SEED)
FOLD_STATS = defaultdict(Counter)
for r in RECS:
    f = FOLDS[r["paragraph_id"]]
    FOLD_STATS[f]["paragraphs"] += 1
    FOLD_STATS[f]["gold_pairs"] += len(pc.gold_pairs_report(r))
    for d in pc.decision_points(r):
        FOLD_STATS[f][f"dp_{d['label']}"] += 1
for f in sorted(FOLD_STATS):
    print("fold", f, dict(FOLD_STATS[f]))

AUDIT_QUOTAS = {"source_implicit": 30, "target_implicit": 20, "multi_pair": 20, "regular": 10}
AUDIT = pc.gold_audit_sample(PAIRS, AUDIT_QUOTAS, seed=SEED)
print("Mẫu kiểm tra nhãn vàng:", dict(Counter(a["audit_group"] for a in AUDIT)), "| theo tập:", dict(Counter(a["split"] for a in AUDIT)))
''')

# =============================================================================
md("## 13. Biểu đồ")
code(r'''
FIGS = []
if ENV["matplotlib"]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C3 = {"train": "#2a78d6", "dev": "#eb6834", "test": "#1baf7a"}   # 3 màu phân loại đầu, cố định theo tập
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

    def _style(ax):
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # Hình 1: tỉ lệ thay đổi ngầm theo nhãn và tập
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    labs = ["CREATE", "DESTROY", "MOVE"]
    w = 0.26
    for k, sp in enumerate(SPLITS):
        vals = [100 * TABLE1[sp][l][0] / max(TABLE1[sp][l][1], 1) for l in labs]
        xs = [j + (k - 1) * (w + 0.02) for j in range(len(labs))]
        bars = ax.bar(xs, vals, width=w, color=C3[sp], label=sp, edgecolor="white", linewidth=2)
        for x, v, l in zip(xs, vals, labs):
            ax.text(x, v + 0.8, f"{TABLE1[sp][l][0]}/{TABLE1[sp][l][1]}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ymax = max(100 * TABLE1[sp][l][0] / max(TABLE1[sp][l][1], 1) for sp in SPLITS for l in labs)
    ax.set_ylim(0, ymax * 1.22)
    ax.set_xticks(range(len(labs)), labs, color=INK)
    ax.set_ylabel("% thay đổi là ngầm", color=MUTED)
    ax.set_title("Thay đổi ngầm chiếm 10–35% số thay đổi; CREATE/DESTROY ngầm là nhóm nhỏ", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _style(ax)
    fig.tight_layout()
    p = OUT / "reports/figures/fig1_implicit_rate_by_label.png"
    fig.savefig(p); FIGS.append(p.name); plt.show(); plt.close(fig)

    # Hình 2: số thực thể mỗi đoạn (quyết định chi phí giải mã chính xác của PP4)
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=130)
    hist = Counter(NS)
    xs = sorted(hist)
    ax.bar(xs, [hist[x] for x in xs], width=0.7, color="#2a78d6", edgecolor="white", linewidth=2)
    for x in xs:
        ax.text(x, hist[x] + 1, str(hist[x]), ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_ylim(0, max(hist.values()) * 1.15)
    ax.set_xticks(xs)
    ax.set_xlabel("số thực thể n trong đoạn", color=MUTED)
    ax.set_ylabel("số đoạn", color=MUTED)
    ax.set_title(f"n tối đa = {max(NS)} → 4^n ≤ {4 ** max(NS):,} cấu hình/câu: giải mã chính xác khả thi", loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    p = OUT / "reports/figures/fig2_entities_per_paragraph.png"
    fig.savefig(p); FIGS.append(p.name); plt.show(); plt.close(fig)
else:
    print("Không có matplotlib — bỏ qua biểu đồ (không ảnh hưởng cổng nghiệm thu).")
''')

# =============================================================================
md("## 14. Cổng nghiệm thu: đối chiếu số liệu báo cáo + kiểm tra cấu trúc")
code(r'''
CHECKS = []
def check(cid, desc, expected, observed, ok, severity="FAIL", note=""):
    status = "PASS" if ok else severity
    if SMOKE_TEST and cid.startswith("R"):
        status = "SKIP(smoke)"
    CHECKS.append({"id": cid, "check": desc, "expected": str(expected), "observed": str(observed), "status": status, "note": note})

# --- S: cấu trúc, toàn vẹn, rò rỉ
check("S01", "Kiểm thử đơn vị", f"{n_tests}/{n_tests}", f"{n_tests - len(failed_tests)}/{n_tests}", not failed_tests)
check("S02", "SHA-256 file gốc khớp giá trị ghim", "8/8", f"{sum(v['match'] for v in CHECKSUM.values())}/8",
      all(v["match"] for v in CHECKSUM.values()), "WARN", "lệch = bản dữ liệu khác; các kiểm tra R quyết định")
check("S03", "Không có lỗi cấu trúc (độ dài trạng thái, câu rỗng, trạng thái rỗng)", 0, len(HARD_ISSUES), not HARD_ISSUES)
check("S04", "Thực thể trùng tên trong cùng đoạn", 0, len(DUP_ISSUES), not DUP_ISSUES, "WARN",
      f"đoạn {sorted({p for p, _ in DUP_ISSUES})}; giữ nguyên, định danh thực thể bằng chỉ số")
check("S05", "Mã đoạn duy nhất, không chéo tập, khớp file phân chia", "0/0/0",
      f"{len(PID_DUP_WITHIN)}/{len(PID_CROSS)}/{len(PARTITION_MISMATCH)}", not (PID_DUP_WITHIN or PID_CROSS or PARTITION_MISMATCH))
check("S06", "Không có chủ đề (prompt) nằm ở >1 tập", 0, len(TOPIC_CROSS), not TOPIC_CROSS, "WARN")
check("S07", "Không có đoạn văn trùng nguyên văn giữa các tập", 0, len(TEXT_CROSS), not TEXT_CROSS)
check("S08", "Câu trùng nguyên văn giữa các tập (thông tin)", "ghi nhận", len(SENT_CROSS), True, "INFO",
      "câu ngắn chung chung kiểu 'the magma cools.'; không phải rò rỉ nhãn")
check("S09", "Nhãn vàng không vi phạm ràng buộc tồn tại", 0, GOLD_VIOL, GOLD_VIOL == 0)
ans_bad = {sp: ANSWER_CMP[sp]["n_mismatched_steps"] for sp in SPLITS}
check("S10", "Hành động suy từ grids.v1 = answers.tsv chính thức (so đa tập)", "0/0/0",
      "/".join(str(ans_bad[sp]) for sp in SPLITS), sum(ans_bad.values()) == 0)
check("S11", "Mọi đoạn có chủ đề (để chia chéo theo chủ đề)", 0, len(NO_TOPIC), not NO_TOPIC, "WARN")

# --- R: con số trong báo cáo
c = COUNTS["all"]
check("R01", "Số đoạn / câu / thực thể (tr.1)", "488/3302/1915", f"{c['paragraphs']}/{c['sentences']}/{c['entities']}",
      (c["paragraphs"], c["sentences"], c["entities"]) == (488, 3302, 1915))
check("R02", "Số đoạn train/dev/test (file phân chia)", "391/43/54", "/".join(str(COUNTS[sp]["paragraphs"]) for sp in SPLITS),
      all(COUNTS[sp]["paragraphs"] == EXPECTED_SPLIT_PARAS[sp] for sp in SPLITS))
t1_tot_ok = all(TABLE1[sp][l][1] == REPORT["table1"][sp][l][1] for sp in SPLITS for l in ("CREATE", "DESTROY", "MOVE"))
t1_imp_ok = all(TABLE1[sp][l][0] == REPORT["table1"][sp][l][0] for sp in SPLITS for l in ("CREATE", "DESTROY", "MOVE"))
check("R03", "Bảng 1: tổng số nhãn theo tập (9 ô)", "khớp 9/9",
      f"khớp {sum(TABLE1[sp][l][1] == REPORT['table1'][sp][l][1] for sp in SPLITS for l in ('CREATE','DESTROY','MOVE'))}/9", t1_tot_ok)
check("R04", "Bảng 1: số thay đổi ngầm (9 ô, cờ nhắc kiểu báo cáo)", "khớp 9/9",
      f"khớp {sum(TABLE1[sp][l][0] == REPORT['table1'][sp][l][0] for sp in SPLITS for l in ('CREATE','DESTROY','MOVE'))}/9", t1_imp_ok)
check("R05", "Tổng cặp vàng D×C (3)", 533, len(PAIRS), len(PAIRS) == 533)
check("R06", "Cặp vàng train/dev/test", "378/58/97", "/".join(str(PAIRS_SPLIT[sp]) for sp in SPLITS),
      all(PAIRS_SPLIT[sp] == REPORT["gold_pairs_split"][sp] for sp in SPLITS))
check("R07", "Điểm quyết định DESTROY/CREATE (tr.2)", "57/76", f"{DP_STATS['DESTROY']['n']}/{DP_STATS['CREATE']['n']}",
      (DP_STATS["DESTROY"]["n"], DP_STATS["CREATE"]["n"]) == (57, 76))
ac = (DP_STATS["DESTROY"]["avg_candidates"] or 0, DP_STATS["CREATE"]["avg_candidates"] or 0)
check("R08", "Số ứng viên TB DESTROY/CREATE (tr.3, làm tròn)", "2.6/2.9", f"{ac[0]:.2f}/{ac[1]:.2f}",
      abs(ac[0] - 2.6) <= 0.06 and abs(ac[1] - 2.9) <= 0.06, "WARN", "báo cáo có vẻ cắt bớt chữ số thay vì làm tròn")
ra = (100 * (DP_STATS["DESTROY"]["random_acc"] or 0), 100 * (DP_STATS["CREATE"]["random_acc"] or 0))
check("R09", "Chọn ngẫu nhiên kỳ vọng DESTROY/CREATE (Bảng 3)", "51%/40%", f"{ra[0]:.1f}%/{ra[1]:.1f}%",
      round(ra[0]) == 51 and round(ra[1]) == 40)
lift_ok = all(abs(LIFT[k]["lift"] - v[0]) < 0.051 and LIFT[k]["n_implicit_A_with_B"] == v[1] for k, v in REPORT["lift"].items())
check("R10", "Bảng 2: lift + số ô (9 ô)", "khớp 9/9",
      f"khớp {sum(abs(LIFT[k]['lift'] - v[0]) < 0.051 and LIFT[k]['n_implicit_A_with_B'] == v[1] for k, v in REPORT['lift'].items())}/9", lift_ok)
check("R11", "PP1: câu hỏi nhãn 4ΣTn / cặp ΣTn(n−1) (6)(7)", "52936/48070", f"{PP2['label_questions']}/{PP2['all_pairs']}",
      (PP2["label_questions"], PP2["all_pairs"]) == (52936, 48070))
check("R12", "PP2 với tồn tại VÀNG: số cặp còn lại / cặp vàng giữ lại (12)", "11144/533", f"{PP2['kept_gold_existence']}/{PP2['gold_kept']}",
      (PP2["kept_gold_existence"], PP2["gold_kept"]) == (11144, 533))
check("R13", "Lọc đồng xuất hiện: % cặp vàng / % cặp nguồn ngầm (tr.4)", "74.7/18.9",
      f"{COOC_PCT['any_sentence_gold']:.1f}/{COOC_PCT['any_sentence_src_implicit']:.1f}",
      round(COOC_PCT["any_sentence_gold"], 1) == 74.7 and round(COOC_PCT["any_sentence_src_implicit"], 1) == 18.9, "WARN",
      "chỉ khớp nếu hiểu 'đồng xuất hiện' = cùng được nhắc trong MỘT câu bất kỳ của đoạn; nếu hiểu là cùng câu t thì "
      f"{COOC_PCT['same_sentence_gold']:.1f}/{COOC_PCT['same_sentence_src_implicit']:.1f}")
check("R14", "Số thực thể TB mỗi đoạn (tr.6)", 3.9, PP4["mean_entities"], round(PP4["mean_entities"], 1) == 3.9)

# --- D: phát hiện cần ghi nhận (không chặn)
check("D01", "Số conversion vàng theo bộ chấm CHÍNH THỨC ≠ số cặp D×C", "khác 533 (định nghĩa khác)",
      f"{len(OFFICIAL_CONV)} ({'/'.join(str(OFFICIAL_SPLIT[sp]) for sp in SPLITS)})", False, "WARN",
      "evaluator gộp mọi D,C của một câu thành 1 conversion + luật ghép câu kế tiếp + chấm Jaccard mềm, macro theo đoạn. "
      "Bản sao này đếm cả thực thể trùng tên; evaluator gốc gộp chúng ở 4 đoạn train → notebook 01 đối chiếu với mã gốc")
check("D02", "Cặp ở câu 1×1 (cặp được quyết định hoàn toàn bởi nhãn)", "ghi nhận",
      f"{PAIR_STATS['sentences_1x1']}/{PAIR_STATS['sentences_with_pairs']} câu; {len(PAIRS) - PAIR_STATS['pairs_from_multi_entity_sentences']}/{len(PAIRS)} cặp",
      True, "INFO", "với định nghĩa D×C, tập cặp là hàm của nhãn CREATE/DESTROY")
check("D03", "Không có đoạn n>8 (giải mã chính xác 4^n khả thi)", 0, PP4["paragraphs_n_gt_8"], PP4["paragraphs_n_gt_8"] == 0, "WARN")
check("D04", "Độ nhạy cờ nhắc (extended −es/−ies) — số ô ngầm thay đổi", "ghi nhận", sum(SENS.values()), True, "INFO", str(SENS_D))

n_fail = sum(c["status"] == "FAIL" for c in CHECKS)
n_warn = sum(c["status"] == "WARN" for c in CHECKS)
GATE_STATUS = "SMOKE_OK" if (SMOKE_TEST and n_fail == 0) else ("PASS" if n_fail == 0 else "FAIL")
w = max(len(c["check"]) for c in CHECKS)
print(f"{'ID':4s} {'TRẠNG THÁI':11s} {'KIỂM TRA':{w}s}  KỲ VỌNG → QUAN SÁT")
for c in CHECKS:
    print(f"{c['id']:4s} {c['status']:11s} {c['check']:{w}s}  {c['expected']} → {c['observed']}")
print(f"\nFAIL={n_fail}  WARN={n_warn}  →  GATE_STATUS = {GATE_STATUS}")
''')

# =============================================================================
md("## 15. Lưu đầu ra (dữ liệu chuẩn, báo cáo, manifest, zip)")
code(r'''
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})

P = OUT / "data/processed"
for sp in SPLITS:
    write_jsonl(P / f"{sp}.jsonl", BY_SPLIT[sp])
    # Đầu vào cho mô hình: KHÔNG có trạng thái/nhãn → tránh lộ đáp án ở các notebook Qwen/mô hình
    write_jsonl(P / f"model_inputs_{sp}.jsonl",
                [{"paragraph_id": r["paragraph_id"], "split": sp, "sentences": r["sentences"], "entities": r["entities"]} for r in BY_SPLIT[sp]])
write_jsonl(P / "gold_pairs_report.jsonl", PAIRS)
write_jsonl(P / "official_q3_gold_replica.jsonl", OFFICIAL_CONV)
write_jsonl(P / "decision_points.jsonl", DPS)
cells = []
for r in RECS:
    for i, ent in enumerate(r["entities"]):
        for t in range(1, len(r["sentences"]) + 1):
            cells.append({"paragraph_id": r["paragraph_id"], "split": r["split"], "sentence_id": t, "entity_idx": i,
                          "entity": ent, "state_before": r["states"][i][t - 1], "state_after": r["states"][i][t],
                          "exists_before": int(r["existence"][i][t - 1]), "exists_after": int(r["existence"][i][t]),
                          "label": r["labels"][i][t - 1], "mentioned": int(r["mentions"][i][t - 1]),
                          "implicit": int(r["implicit"][i][t - 1])})
write_csv(P / "cells.csv", cells)
assert not any(k in json.loads(open(P / "model_inputs_dev.jsonl", encoding="utf-8").readline())
               for k in ("states", "labels", "existence", "implicit")), "model_inputs bị lộ nhãn!"

write_csv(OUT / "data/splits/official_split.csv",
          [{"paragraph_id": r["paragraph_id"], "split": r["split"], "topic": r["topic"]} for r in RECS])
write_csv(OUT / "data/splits/cv5_by_topic.csv",
          [{"paragraph_id": r["paragraph_id"], "official_split": r["split"], "topic": r["topic"], "fold": FOLDS[r["paragraph_id"]]} for r in RECS])
AUDIT_FIELDS = ["audit_group", "split", "paragraph_id", "sentence_id", "source", "target", "source_implicit", "target_implicit",
                "n_destroy_in_sentence", "n_create_in_sentence", "sentence_text", "paragraph_text",
                "annotator_A_label", "annotator_A_note", "annotator_B_label", "annotator_B_note", "resolved_label"]
rec_by_pid = {r["paragraph_id"]: r for r in RECS}
for a in AUDIT:
    rr = rec_by_pid[a["paragraph_id"]]
    a["sentence_text"] = rr["sentences"][a["sentence_id"] - 1]
    a["paragraph_text"] = " ".join(f"[{k}] {s}" for k, s in enumerate(rr["sentences"], 1))
    for k in AUDIT_FIELDS[-5:]:
        a[k] = ""
write_csv(OUT / "data/audit/gold_audit_sample.csv", AUDIT, AUDIT_FIELDS)

write_csv(OUT / "reports/gate_checks.csv", CHECKS)
SUMMARY = {
    "run_tag": RUN_TAG, "core_version": pc.CORE_VERSION, "gate_status": GATE_STATUS, "n_fail": n_fail, "n_warn": n_warn,
    "smoke_test": SMOKE_TEST, "seed": SEED, "env": ENV, "raw_sources": RAW_SOURCE, "checksums": CHECKSUM,
    "counts": COUNTS, "table1_implicit_over_total": {sp: {l: list(v) for l, v in TABLE1[sp].items()} for sp in SPLITS},
    "pair_stats": PAIR_STATS, "decision_points": DP_STATS, "pp1_pp2": dict(PP2), "cooccurrence_pct": COOC_PCT,
    "lift": {f"{a}|{b}": v for (a, b), v in LIFT.items()}, "pp4_feasibility": PP4, "mention_sensitivity": SENS_D,
    "duplicate_entity_paragraphs": sorted({p for p, _ in DUP_ISSUES}), "cross_split_duplicate_sentences": SENT_CROSS,
    "folds": {str(f): dict(v) for f, v in sorted(FOLD_STATS.items())}, "checks": CHECKS,
    "runtime_sec": round(time.time() - T0, 1), "figures": FIGS,
}
(OUT / "reports/audit_summary.json").write_text(json.dumps(SUMMARY, indent=1, ensure_ascii=False), encoding="utf-8")

md_lines = [f"# dataset_validation — {RUN_TAG}", "",
            f"- GATE_STATUS: **{GATE_STATUS}** (FAIL={n_fail}, WARN={n_warn}); SMOKE_TEST={SMOKE_TEST}",
            f"- Nguồn: allenai/propara@{PIN_PROPARA[:7]} (grids.v1), allenai/aristo-leaderboard@{PIN_LEADER[:7]} (answers.tsv)",
            f"- Môi trường: Python {ENV['python']}, {ENV['platform']}, on_kaggle={ENV['on_kaggle']}", "",
            "| ID | Trạng thái | Kiểm tra | Kỳ vọng | Quan sát | Ghi chú |", "|---|---|---|---|---|---|"]
md_lines += [f"| {c['id']} | {c['status']} | {c['check']} | {c['expected']} | {c['observed']} | {c['note']} |" for c in CHECKS]
(OUT / "reports/dataset_validation.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

manifest = {"run_tag": RUN_TAG, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name != "manifest.json":
        manifest["files"][p.relative_to(OUT).as_posix()] = {"bytes": p.stat().st_size, "sha256": pc.sha256_file(str(p))}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

ZIP = WORK_ROOT / f"{RUN_TAG}.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            z.write(p, Path(RUN_TAG) / p.relative_to(OUT))
print(f"Đã ghi {len(manifest['files'])} file vào {OUT}\nZip: {ZIP} ({ZIP.stat().st_size/1e6:.2f} MB)")
''')

# =============================================================================
md("## 16. Kết quả để gửi lại (sao chép toàn bộ khối dưới đây)")
code(r'''
print("=== GỬI LẠI ===")
print(f"notebook=00_propara_data_audit run_tag={RUN_TAG} core={pc.CORE_VERSION} GATE_STATUS={GATE_STATUS} FAIL={n_fail} WARN={n_warn} smoke={SMOKE_TEST}")
print(f"env: python={ENV['python']} on_kaggle={ENV['on_kaggle']} gpu={ENV['gpu_present']} internet={ENV['internet']} runtime={SUMMARY['runtime_sec']}s")
print("nguồn:", sorted({v.split(':')[0] for v in RAW_SOURCE.values()}), "| checksum khớp:", sum(v['match'] for v in CHECKSUM.values()), "/8")
for c in CHECKS:
    if c["status"] != "PASS":
        print(f"  {c['id']} {c['status']}: {c['check']} | {c['expected']} → {c['observed']}")
print("cặp:", PAIR_STATS["pairs_by_split"], "| official conv (bản sao):", PAIR_STATS["official_q3_conversions_replica_by_split"])
print("điểm QĐ:", {k: (v['n'], v['by_split']) for k, v in DP_STATS.items()})
print("=== HẾT ===")
if GATE_STATUS == "FAIL":
    raise RuntimeError("Cổng nghiệm thu FAIL. Xem bảng ở mục 14; output đã được lưu để chẩn đoán.")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False},
}
out = ROOT / "notebooks" / "00_propara_data_audit.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.validate(nb)
nbf.write(nb, out)
print("Đã ghi", out, "| số ô:", len(cells))
