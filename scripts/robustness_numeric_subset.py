"""Post-hoc analysis for the 'contradictory' robustness condition: restrict
to examples where the number perturbation actually changed a number that
the response itself states (otherwise the corrupted context does not
contradict the response at all). CPU-only; reuses robustness_<tag>.csv.

    python scripts/robustness_numeric_subset.py --tag tinyllama_1.1b
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redeep.data import load_examples
from redeep.robustness import build_robustness_conditions

NUM = re.compile(r"\b\d+(?:\.\d+)?\b")

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
args = ap.parse_args()
data_dir = Path(__file__).resolve().parents[1] / "data"
prev = json.loads((data_dir / f"results_{args.tag}.json").read_text())["args"]
models = prev["response_models"].split(",") if prev.get("response_models") else None
pool = load_examples(split=prev["split"], models=models)
by_id = {e["id"]: e for e in pool}
df = pd.read_csv(data_dir / f"robustness_{args.tag}.csv", dtype={"id": str})

hits = []
for _, r in df.iterrows():
    ex = by_id[r["id"]]
    conds = build_robustness_conditions(ex, pool)
    orig_nums = set(NUM.findall(ex["prompt"]))
    new_nums = set(NUM.findall(conds["contradictory"]))
    resp_nums = set(NUM.findall(ex["response"]))
    # numbers the response copies from the real context that no longer appear in the corrupted one
    hits.append(len((resp_nums & orig_nums) - new_nums))
df["n_contradicted_numbers"] = hits

out = {}
for name, sub in (("all", df), ("response_numbers_contradicted", df[df.n_contradicted_numbers > 0])):
    diff = sub["score_contradictory"] - sub["score_accurate"]
    de = sub["E_A_contradictory"] - sub["E_A_accurate"]
    out[name] = {"n": int(len(sub)), "n_halluc": int(sub.label.sum()),
                 "score_accurate": float(sub.score_accurate.mean()),
                 "score_contradictory": float(sub.score_contradictory.mean()),
                 "score_shift": float(diff.mean()),
                 "score_shift_p": float(wilcoxon(diff).pvalue) if len(sub) > 5 else None,
                 "E_A_shift": float(de.mean()),
                 "frac_score_increased": float((diff > 0).mean())}
print(json.dumps(out, indent=2))
(data_dir / f"robustness_numeric_subset_{args.tag}.json").write_text(json.dumps(out, indent=2))
