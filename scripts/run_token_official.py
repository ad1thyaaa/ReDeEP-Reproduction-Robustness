"""ReDeEP(token) on the authors' data files: RAGTruth responses of any
generator (incl. llama-3-8b-instruct) or Dolly(AC). Scores every response
(resumable), then evaluates with our leakage-free 5-fold CV.

    python scripts/run_token_official.py --model NousResearch/Llama-2-7b-chat-hf --load_in_4bit \
        --tag tok_llama2_7b_dolly --dataset dolly --response_model llama-2-7b-chat
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from redeep.mechanistic import LlamaMechInterp
from redeep.detect import ReDeEPDetector
from run_chunk import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", choices=["ragtruth", "dolly"], default="ragtruth")
    ap.add_argument("--response_model", default="llama-2-7b-chat")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ex = load(args.dataset, args.response_model)
    data = ROOT / "data"
    cache = data / f"scores_{args.tag}.npz"
    part = data / f"scores_{args.tag}.partial.jsonl"
    done = {}
    if part.exists():
        for l in open(part, encoding="utf-8"):
            r = json.loads(l); done[r["id"]] = r
    print(f"{len(ex)} responses ({sum(e['label'] for e in ex)} hallucinated), {len(done)} done")
    mech = LlamaMechInterp(args.model, device=args.device, dtype=getattr(torch, args.dtype), load_in_4bit=args.load_in_4bit)
    cop = mech.copying_head_scores(cache_path=data / f"copying_heads_{args.tag}.npy")
    with open(part, "a", encoding="utf-8") as f:
        for i, e in enumerate(ex):
            if e["id"] in done:
                continue
            t0 = time.time()
            p = mech.tokenizer(e["prompt"], return_tensors="pt").input_ids[0]
            r = mech.tokenizer(e["response"], return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(r) == 0:
                continue
            try:
                s = mech.score_example(p, r, max_len=args.max_len)
            except torch.cuda.OutOfMemoryError:
                mech._topk_req = None; torch.cuda.empty_cache(); print("  OOM skip", flush=True); continue
            row = {"id": e["id"], "label": e["label"], "task_type": e["task_type"], "E": s["E"].tolist(),
                   "P": s["P"].tolist(), "ppl": s["ppl"], "lnent": s["lnent"]}
            f.write(json.dumps(row) + "\n"); f.flush(); done[e["id"]] = row
            print(f"  [{i+1}/{len(ex)}] label={e['label']} {time.time()-t0:.1f}s", flush=True)

    rows = [done[e["id"]] for e in ex if e["id"] in done]
    E = np.array([r["E"] for r in rows]); P = np.array([r["P"] for r in rows]); y = np.array([r["label"] for r in rows])
    ppl = np.array([r["ppl"] for r in rows]); ln = np.array([r["lnent"] for r in rows])
    np.savez(cache, E=E, P=P, y=y, ppl=ppl, lnent=ln, ids=np.array([r["id"] for r in rows]))
    folds = {"AUC": [], "PPL_AUC": [], "LNEnt_AUC": []}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        det = ReDeEPDetector(8, 5).fit(E[tr], P[tr], y[tr], cop)
        m, _ = det.evaluate(E[te], P[te], y[te])
        folds["AUC"].append(m["AUC"])
        folds["PPL_AUC"].append(roc_auc_score(y[te], ppl[te]))
        folds["LNEnt_AUC"].append(roc_auc_score(y[te], ln[te]))
    res = {"model": args.model, "dataset": args.dataset, "response_model": args.response_model, "n": int(len(y)),
           "n_hallucinated": int(y.sum()), "cv": {k: [float(np.mean(v)), float(np.std(v))] for k, v in folds.items()},
           "load_in_4bit": args.load_in_4bit, "dtype": args.dtype}
    print(json.dumps(res, indent=1))
    (data / f"results_{args.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
