"""Extension: robustness of ReDeEP under corrupted retrieval (see
redeep/robustness.py). Requires a finished run_pipeline.py run for the
same --tag (reuses its cached scores + copying heads to refit the
detector), then rescoring each response under accurate / irrelevant /
number-corrupted context.

    python scripts/run_robustness.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --tag tinyllama_1.1b --n 150

Writes data/robustness_<tag>.json (+ per-example .csv).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redeep.data import load_examples, subsample_balanced
from redeep.detect import ReDeEPDetector, aggregate_features
from redeep.mechanistic import LlamaMechInterp
from redeep.robustness import build_robustness_conditions

CONDS = ["accurate", "irrelevant", "contradictory"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="tag of the finished run_pipeline.py run")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / "data"
    prev = json.loads((data_dir / f"results_{args.tag}.json").read_text())
    pa = prev["args"]
    d = np.load(data_dir / f"scores_{args.tag}.npz")
    copying = np.load(data_dir / f"copying_heads_{args.tag}.npy")

    # Detector fitted on everything *except* the examples we perturb, so the
    # accurate-context condition is a genuine held-out score.
    models = pa["response_models"].split(",") if pa.get("response_models") else None
    pool = load_examples(split=pa["split"], models=models)
    chosen = subsample_balanced(pool, len(pool), max_prompt_chars=pa["max_prompt_chars"])
    rng = np.random.RandomState(1)
    eval_ids = set(rng.choice(d["ids"], size=min(args.n, len(d["ids"]) // 2), replace=False))
    train_mask = np.array([i not in eval_ids for i in d["ids"]])
    det = ReDeEPDetector(pa["n_copying_heads"], pa["n_knowledge_ffns"])
    det.fit(d["E"][train_mask], d["P"][train_mask], d["y"][train_mask], copying)
    examples = [e for e in chosen if e["id"] in eval_ids]
    print(f"detector fit on {train_mask.sum()} examples; perturbing {len(examples)}")

    dtype = torch.float16
    mech = LlamaMechInterp(args.model, device="cuda", dtype=dtype, load_in_4bit=args.load_in_4bit)

    rows = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        conds = build_robustness_conditions(ex, pool)
        row = {"id": ex["id"], "label": ex["label"], "task_type": ex["task_type"]}
        resp_ids = mech.tokenizer(ex["response"], return_tensors="pt", add_special_tokens=False).input_ids[0]
        for c in CONDS:
            if c not in conds or resp_ids.shape[0] == 0:
                continue
            p_ids = mech.tokenizer(conds[c], return_tensors="pt").input_ids[0]
            try:
                s = mech.score_example(p_ids, resp_ids, max_len=args.max_len)
            except torch.cuda.OutOfMemoryError:
                mech._topk_req = None
                torch.cuda.empty_cache()
                continue
            e_a, p_f = aggregate_features(s["E"][None], s["P"][None], det.heads_A, det.layers_F)
            row[f"E_A_{c}"] = float(e_a[0])
            row[f"P_F_{c}"] = float(p_f[0])
            row[f"score_{c}"] = float(det.score(s["E"][None], s["P"][None])[0])
            row[f"ppl_{c}"] = s["ppl"]
        rows.append(row)
        print(f"  [{i+1}/{len(examples)}] label={ex['label']} "
              + " ".join(f"{c}={row.get('score_'+c, float('nan')):.3f}" for c in CONDS)
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / f"robustness_{args.tag}.csv", index=False)

    summary = {"tag": args.tag, "model": args.model, "n": len(df), "conditions": {}}
    for c in CONDS:
        sub = df.dropna(subset=[f"score_{c}"])
        m = {"n": len(sub)}
        for feat in ("score", "E_A", "P_F", "ppl"):
            col = f"{feat}_{c}"
            m[f"{feat}_mean"] = float(sub[col].mean())
            m[f"{feat}_mean_faithful"] = float(sub[sub.label == 0][col].mean())
            m[f"{feat}_mean_halluc"] = float(sub[sub.label == 1][col].mean())
        if sub.label.nunique() > 1:
            m["AUC_vs_original_labels"] = float(roc_auc_score(sub.label, sub[f"score_{c}"]))
        if c != "accurate":
            both = df.dropna(subset=[f"score_{c}", "score_accurate"])
            for feat in ("score", "E_A", "P_F"):
                diff = both[f"{feat}_{c}"] - both[f"{feat}_accurate"]
                m[f"{feat}_shift_vs_accurate"] = float(diff.mean())
                m[f"{feat}_shift_wilcoxon_p"] = float(wilcoxon(diff).pvalue) if len(diff) > 5 else float("nan")
        summary["conditions"][c] = m

    print(json.dumps(summary, indent=2))
    (data_dir / f"robustness_{args.tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
