"""Khối ô dùng chung cho các notebook 01+ (tự chứa: tự tải dữ liệu, nhúng mã lõi).

Mỗi notebook sau 00 KHÔNG cần Add Input output của notebook trước: nó tự tải ProPara theo commit
ghim (hoặc dò /kaggle/input), tự dựng lại dữ liệu chuẩn bằng propara_core cùng phiên bản, và tự
kiểm tra lại các con số then chốt (488/3302/1915, 533 cặp) trước khi chạy tiếp.
"""
import base64
import hashlib
from pathlib import Path

import nbformat as nbf

RAW_FILES_SRC = r'''
PIN_PROPARA = "51a3cccbedcbe5f92276f9925e0ce1e6ec4247bb"   # allenai/propara (2019-08-30)
PIN_LEADER  = "60d5be31b5a7d36f29d13223ece29a8d2bfa8b5f"   # allenai/aristo-leaderboard (2021-05-17)
_GH = "https://raw.githubusercontent.com/allenai"
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
'''

ENV_SRC = r'''
def _internet_ok(url="https://raw.githubusercontent.com", timeout=6):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception as e:
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
for k, v in ENV.items():
    print(f"{k:14s}: {v}")
'''

LOAD_SRC = r'''
def find_in_input(suffixes):
    if not INPUT_ROOT.exists():
        return None
    for suf in suffixes:
        for p in sorted(INPUT_ROOT.rglob(suf.split("/")[-1])):
            if p.is_file() and p.as_posix().endswith(suf):
                return p
    return None

RAW, RAW_SOURCE, CHECKSUM = {}, {}, {}
(OUT / "raw").mkdir(parents=True, exist_ok=True)
for name, (url, sha, suffixes) in RAW_FILES.items():
    p = find_in_input(suffixes)
    if p is not None:
        RAW_SOURCE[name] = f"input:{p}"
    else:
        p = OUT / "raw" / name
        if not p.exists():
            if ENV["internet"] is not True:
                raise FileNotFoundError(f"Không tìm thấy {name} trong {INPUT_ROOT} và không có Internet. "
                                        "Bật Settings → Internet, hoặc Add Input dataset propara-raw-v1.")
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
    print(f"{'OK ' if got == sha else 'LỆCH'}  {name}")

TOPICS = pc.load_topics_from_naacl_grid(str(RAW["gold-full-grids.v3.tsv"]))
RECS = [pc.build_record(r, sp, TOPICS.get(str(r["para_id"])))
        for sp in SPLITS for r in pc.load_jsonl(str(RAW[f"grids.v1.{sp}.json"]))]
BY_SPLIT = {sp: [r for r in RECS if r["split"] == sp] for sp in SPLITS}
BY_PID = {r["paragraph_id"]: r for r in RECS}
_n = (len(RECS), sum(len(r["sentences"]) for r in RECS), sum(len(r["entities"]) for r in RECS))
_pairs = {sp: sum(len(pc.gold_pairs_report(r)) for r in BY_SPLIT[sp]) for sp in SPLITS}
print("đoạn/câu/thực thể:", _n, "| cặp vàng:", _pairs)
if _n != (488, 3302, 1915) or _pairs != {"train": 378, "dev": 58, "test": 97}:
    raise RuntimeError("Dữ liệu dựng lại KHÔNG khớp kết quả notebook 00 (488/3302/1915; 378/58/97). Dừng.")
'''


def embed_modules_cell(modules: dict, blobs: dict = None) -> str:
    """modules: {tên_file.py: nội_dung}; blobs: {tên_file: bytes} (nhúng base64 + sha256)."""
    parts = ["MODULES = {}"]
    for name, src in modules.items():
        assert "'''" not in src, f"{name} chứa '''"
        parts.append(f"MODULES[{name!r}] = r'''\n{src}'''")
    parts.append("BLOBS = {}")
    for name, data in (blobs or {}).items():
        b64 = base64.b64encode(data).decode()
        lines = "\n".join(b64[i:i + 120] for i in range(0, len(b64), 120))
        parts.append(f"BLOBS[{name!r}] = ({hashlib.sha256(data).hexdigest()!r}, '''\n{lines}\n''')")
    parts.append(r'''
import base64 as _b64, zipfile as _zf, io as _io, importlib
MOD_DIR = OUT / "code"
MOD_DIR.mkdir(parents=True, exist_ok=True)
for _name, _src in MODULES.items():
    (MOD_DIR / _name).write_text(_src, encoding="utf-8")
for _name, (_sha, _b) in BLOBS.items():
    _data = _b64.b64decode("".join(_b.split()))
    if hashlib.sha256(_data).hexdigest() != _sha:
        raise RuntimeError(f"Blob {_name} hỏng (sha256 lệch)")
    (MOD_DIR / _name).write_bytes(_data)
    if _name.endswith(".zip"):
        with _zf.ZipFile(_io.BytesIO(_data)) as z:
            z.extractall(MOD_DIR)
sys.dont_write_bytecode = True
if str(MOD_DIR) not in sys.path:
    sys.path.insert(0, str(MOD_DIR))
for _m in [n[:-3] for n in MODULES]:
    if _m in sys.modules:
        importlib.reload(sys.modules[_m])
print("Đã nạp:", ", ".join(MODULES), "|", ", ".join(BLOBS))
''')
    return "\n".join(parts)


def cfg_header(run_tag: str, extra: str = "") -> str:
    return (r'''
import os, sys, json, time, platform, shutil, hashlib, csv, zipfile, random, statistics, urllib.request, warnings
from collections import Counter, defaultdict
from pathlib import Path
warnings.filterwarnings("ignore", category=SyntaxWarning)   # stemmer.py gốc có chuỗi '\m' cũ

SEED = 2026
RUN_TAG = "''' + run_tag + r'''"
INPUT_ROOT = Path(os.environ.get("PROPARA_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT  = Path(os.environ.get("PROPARA_WORK_ROOT", "/kaggle/working"))
OUT = WORK_ROOT / RUN_TAG
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
''' + extra + RAW_FILES_SRC)


def save_notebook(cells, path: Path, gpu: bool = False, internet: bool = True):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isInternetEnabled": internet, "isGpuEnabled": gpu},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.validate(nb)
    nbf.write(nb, path)
    return path
