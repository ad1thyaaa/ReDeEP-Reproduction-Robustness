"""Stage the code + official dataset as a private Kaggle dataset, and the
kernel folder, for `kaggle datasets create` / `kaggle kernels push`.

    python kaggle/make_bundle.py --user <kaggle username>
"""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REDEEP = HERE.parent
OFFICIAL = REDEEP.parent / "ReDeEP_official_dataset"

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
args = ap.parse_args()

b = HERE / "bundle"
shutil.rmtree(b, ignore_errors=True)
(b / "redeep" / "data").mkdir(parents=True)
shutil.copytree(REDEEP / "redeep", b / "redeep" / "redeep", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(REDEEP / "scripts", b / "redeep" / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(OFFICIAL, b / "ReDeEP_official_dataset",
                ignore=shutil.ignore_patterns("response.jsonl", "source_info.jsonl", "dolly.json", "*_v1*.json",
                                              "llama2_7B_response_chunk.json"))
(b / "dataset-metadata.json").write_text(json.dumps(
    {"title": "redeep-bundle", "id": f"{args.user}/redeep-bundle", "licenses": [{"name": "other"}]}, indent=1))

k = HERE / "kernel_push"
shutil.rmtree(k, ignore_errors=True)
k.mkdir()
shutil.copy(HERE / "kernel.py", k / "kernel.py")
(k / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{args.user}/redeep-llama2-13b-llama3-8b", "title": "redeep-llama2-13b-llama3-8b",
    "code_file": "kernel.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{args.user}/redeep-bundle"],
    "competition_sources": [], "kernel_sources": []}, indent=1))
size = sum(f.stat().st_size for f in b.rglob("*") if f.is_file()) / 1e6
print(f"bundle {size:.0f} MB at {b}\nkernel at {k}")
