"""End-to-end ReDeEP reproduction on RAGTruth using a LLaMA-family
backbone. Usage:

    python scripts/run_pipeline.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --n_per_class 40 --max_len 768

    # paper-faithful: Llama-2-7B-chat scoring its own RAGTruth test responses,
    # 4-bit so it fits a 6GB GPU
    python scripts/run_pipeline.py --model NousResearch/Llama-2-7b-chat-hf \
        --load_in_4bit --split test --response_models llama-2-7b-chat --n_per_class 0

Produces:
  - data/scores_<tag>.npz    (cached E/P arrays + labels + ids, so re-runs are free)
  - data/results_<tag>.json  (metrics)
  - prints an AUC/PCC/Acc/Rec/F1 table, ReDeEP vs. Perplexity/LN-Entropy baselines,
    on a single held-out split and under stratified k-fold CV
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redeep.data import load_examples, subsample_balanced
from redeep.mechanistic import LlamaMechInterp
from redeep.detect import ReDeEPDetector


def build_prompt_response_ids(tok, example, max_len):
    prompt_ids = tok(example["prompt"], return_tensors="pt", add_special_tokens=True).input_ids[0]
    response_ids = tok(example["response"], return_tensors="pt", add_special_tokens=False).input_ids[0]
    return prompt_ids, response_ids


def score_all(mech, examples, args, cache_path):
    """Score every example, checkpointing to <cache>.partial.npz so a crash
    or Ctrl-C part-way through a multi-hour 7B run can be resumed."""
    partial_path = cache_path.with_suffix(".partial.npz")
    keys = ["E", "P", "y", "ppl", "lnent", "ids"]
    acc = {k: [] for k in keys}
    if partial_path.exists():
        d = np.load(partial_path, allow_pickle=False)
        acc = {k: list(d[k]) for k in keys}
        print(f"  resuming from checkpoint: {len(acc['ids'])} examples already scored")
    done = set(acc["ids"])

    t_start = time.time()
    n_new = 0
    for i, ex in enumerate(examples):
        if ex["id"] in done:
            continue
        t0 = time.time()
        prompt_ids, response_ids = build_prompt_response_ids(mech.tokenizer, ex, args.max_len)
        if response_ids.shape[0] == 0:
            continue
        try:
            res = mech.score_example(prompt_ids, response_ids, max_len=args.max_len)
        except torch.cuda.OutOfMemoryError:
            mech._topk_req = None
            torch.cuda.empty_cache()
            print(f"  [{i+1}/{len(examples)}] OOM -- skipped (n_tokens={len(prompt_ids)+len(response_ids)})")
            continue
        acc["E"].append(res["E"])
        acc["P"].append(res["P"])
        acc["y"].append(ex["label"])
        acc["ppl"].append(res["ppl"])
        acc["lnent"].append(res["lnent"])
        acc["ids"].append(ex["id"])
        n_new += 1
        eta = (time.time() - t_start) / n_new * (len(examples) - len(acc["ids"]))
        print(f"  [{i+1}/{len(examples)}] label={ex['label']} task={ex['task_type']} "
              f"n_prompt={res['n_prompt']} n_resp={res['n_response']} took {time.time()-t0:.1f}s "
              f"(eta {eta/60:.1f} min)", flush=True)
        if n_new % 25 == 0:
            np.savez(partial_path, **{k: np.array(v) for k, v in acc.items()})

    out = {k: np.array(v) for k, v in acc.items()}
    np.savez(cache_path, **out)
    if partial_path.exists():
        partial_path.unlink()
    return out


def eval_split(E_all, P_all, y, ppls, lnents, train_idx, test_idx, copying_scores, args):
    from sklearn.metrics import roc_auc_score

    det = ReDeEPDetector(n_copying_heads=args.n_copying_heads, n_knowledge_ffns=args.n_knowledge_ffns)
    det.fit(E_all[train_idx], P_all[train_idx], y[train_idx], copying_scores)
    metrics, _ = det.evaluate(E_all[test_idx], P_all[test_idx], y[test_idx])

    for name, vals in (("PPL_AUC", ppls), ("LNEnt_AUC", lnents)):
        v = vals[test_idx]
        ok = ~np.isnan(v)
        yt = y[test_idx][ok]
        metrics[name] = roc_auc_score(yt, v[ok]) if ok.sum() > 2 and len(set(yt)) > 1 else float("nan")
    return metrics, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--n_per_class", type=int, default=30, help="0 = use every eligible example")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--max_prompt_chars", type=int, default=4000)
    ap.add_argument("--n_copying_heads", type=int, default=8)
    ap.add_argument("--n_knowledge_ffns", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--cv", type=int, default=5, help="stratified k-fold CV folds (0 disables)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--response_models", default=None,
                    help="comma list of RAGTruth generator models to keep, e.g. llama-2-7b-chat")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, choices=[None, "float32", "float16", "bfloat16"])
    ap.add_argument("--load_in_4bit", action="store_true", help="bitsandbytes NF4 (needed for 7B on 6GB)")
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype] if args.dtype else (torch.float16 if args.device == "cuda" else torch.float32)

    tag = args.tag or args.model.replace("/", "_")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    cache_path = data_dir / f"scores_{tag}.npz"

    print(f"[1/4] Loading RAGTruth examples...")
    models = args.response_models.split(",") if args.response_models else None
    pool = load_examples(split=args.split, models=models)
    n_per_class = args.n_per_class if args.n_per_class > 0 else len(pool)
    small = subsample_balanced(pool, n_per_class, max_prompt_chars=args.max_prompt_chars)
    print(f"  -> {len(small)} examples ({sum(e['label'] for e in small)} hallucinated)")

    print(f"[2/4] Loading model {args.model} on {args.device} ({dtype}{', 4-bit' if args.load_in_4bit else ''}) ...")
    mech = LlamaMechInterp(args.model, device=args.device, dtype=dtype, load_in_4bit=args.load_in_4bit)
    ch_cache = data_dir / f"copying_heads_{tag}.npy"
    print("  computing copying-head scores (cached after first run)...")
    t0 = time.time()
    copying_scores = mech.copying_head_scores(cache_path=ch_cache)
    print(f"  done in {time.time()-t0:.1f}s. top head: {np.unravel_index(np.argmax(copying_scores), copying_scores.shape)}")

    if cache_path.exists():
        print("[3/4] Loading cached E/P scores...")
        d = np.load(cache_path)
    else:
        print("[3/4] Scoring examples...")
        t0 = time.time()
        d = score_all(mech, small, args, cache_path)
        print(f"  scored {len(d['y'])} examples in {(time.time()-t0)/60:.1f} min")
    E_all, P_all, y, ppls, lnents = d["E"], d["P"], d["y"], d["ppl"], d["lnent"]

    print(f"[4/4] Fitting ReDeEP detector and evaluating (n={len(y)})...")
    n_test = max(1, int(len(y) * args.test_frac))
    idx = np.arange(len(y))
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    single, det = eval_split(E_all, P_all, y, ppls, lnents, train_idx, test_idx, copying_scores, args)

    print(f"\n=== Single held-out split (n_test={n_test}) ===")
    for k, v in single.items():
        print(f"  {k}: {v:.4f}")
    print(f"  selected Copying Heads (layer,head): {det.heads_A}")
    print(f"  selected Knowledge FFN layers: {det.layers_F}")

    results = {"model": args.model, "tag": tag, "n": int(len(y)), "n_hallucinated": int(y.sum()),
               "args": vars(args), "single_split": single,
               "heads_A": [list(map(int, h)) for h in det.heads_A],
               "layers_F": [int(l) for l in det.layers_F],
               "test_ids": [str(x) for x in d["ids"][test_idx]] if "ids" in d else None}

    if args.cv and min(np.bincount(y)) >= args.cv:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=0)
        folds = [eval_split(E_all, P_all, y, ppls, lnents, tr, te, copying_scores, args)[0]
                 for tr, te in skf.split(np.zeros(len(y)), y)]
        cv = {k: (float(np.nanmean([f[k] for f in folds])), float(np.nanstd([f[k] for f in folds])))
              for k in folds[0]}
        print(f"\n=== {args.cv}-fold stratified CV (mean +/- std) ===")
        for k, (m, s) in cv.items():
            print(f"  {k}: {m:.4f} +/- {s:.4f}")
        results["cv"] = cv

    out_path = data_dir / f"results_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
