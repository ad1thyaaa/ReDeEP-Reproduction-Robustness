"""End-to-end ReDeEP reproduction on RAGTruth using a small LLaMA-family
backbone (CPU-friendly). Usage:

    python scripts/run_pipeline.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --n_per_class 40 --max_len 768

Produces:
  - data/scores_<tag>.npz   (cached E/P arrays + labels, so re-runs are free)
  - prints an AUC/PCC/Acc/Rec/F1 table, ReDeEP vs. Perplexity/LN-Entropy baselines
"""
import argparse
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


def perplexity_and_entropy(model, tok, prompt_ids, response_ids, device, max_len):
    full = torch.cat([prompt_ids, response_ids])[-max_len:]
    n_resp = min(len(response_ids), max_len)
    with torch.no_grad():
        out = model(input_ids=full.unsqueeze(0).to(device))
        logits = out.logits[0]
    # predict response tokens from preceding context
    start = full.shape[0] - n_resp
    losses = []
    entropies = []
    for i in range(start, full.shape[0]):
        if i == 0:
            continue
        logit = logits[i - 1]
        target = full[i]
        logp = torch.log_softmax(logit, dim=-1)
        losses.append(-logp[target].item())
        p = logp.exp()
        entropies.append(-(p * logp).sum().item())
    ppl = float(np.exp(np.mean(losses))) if losses else float("nan")
    ln_entropy = float(np.mean(entropies)) / max(1, len(entropies)) if entropies else float("nan")
    return ppl, ln_entropy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--n_per_class", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--max_prompt_chars", type=int, default=4000)
    ap.add_argument("--n_copying_heads", type=int, default=8)
    ap.add_argument("--n_knowledge_ffns", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, choices=[None, "float32", "float16", "bfloat16"])
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype] if args.dtype else (torch.float16 if args.device == "cuda" else torch.float32)

    tag = args.tag or args.model.replace("/", "_")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    cache_path = data_dir / f"scores_{tag}.npz"

    print(f"[1/4] Loading RAGTruth examples...")
    train_ex = load_examples(split="train")
    small = subsample_balanced(train_ex, args.n_per_class, max_prompt_chars=args.max_prompt_chars)
    print(f"  -> {len(small)} examples ({sum(e['label'] for e in small)} hallucinated)")

    print(f"[2/4] Loading model {args.model} on {args.device} ({dtype}) ...")
    mech = LlamaMechInterp(args.model, device=args.device, dtype=dtype)
    ch_cache = data_dir / f"copying_heads_{tag}.npy"
    print("  computing copying-head scores (cached after first run)...")
    t0 = time.time()
    copying_scores = mech.copying_head_scores(cache_path=ch_cache)
    print(f"  done in {time.time()-t0:.1f}s. top head: {np.unravel_index(np.argmax(copying_scores), copying_scores.shape)}")

    if cache_path.exists():
        print("[3/4] Loading cached E/P scores...")
        d = np.load(cache_path)
        E_all, P_all, y, ppls, lnents = d["E"], d["P"], d["y"], d["ppl"], d["lnent"]
    else:
        print("[3/4] Scoring examples (this is the slow part on CPU)...")
        E_list, P_list, y_list, ppl_list, ent_list = [], [], [], [], []
        for i, ex in enumerate(small):
            t0 = time.time()
            prompt_ids, response_ids = build_prompt_response_ids(mech.tokenizer, ex, args.max_len)
            if response_ids.shape[0] == 0:
                continue
            res = mech.score_example(prompt_ids, response_ids, max_len=args.max_len)
            ppl, ent = perplexity_and_entropy(mech.model, mech.tokenizer, prompt_ids, response_ids, args.device, args.max_len)
            E_list.append(res["E"])
            P_list.append(res["P"])
            y_list.append(ex["label"])
            ppl_list.append(ppl)
            ent_list.append(ent)
            print(f"  [{i+1}/{len(small)}] label={ex['label']} task={ex['task_type']} "
                  f"n_prompt={res['n_prompt']} n_resp={res['n_response']} took {time.time()-t0:.1f}s")
        E_all = np.stack(E_list)
        P_all = np.stack(P_list)
        y = np.array(y_list)
        ppls = np.array(ppl_list)
        lnents = np.array(ent_list)
        np.savez(cache_path, E=E_all, P=P_all, y=y, ppl=ppls, lnent=lnents)

    print(f"[4/4] Fitting ReDeEP detector and evaluating (n={len(y)})...")
    n_test = max(1, int(len(y) * args.test_frac))
    idx = np.arange(len(y))
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    det = ReDeEPDetector(n_copying_heads=args.n_copying_heads, n_knowledge_ffns=args.n_knowledge_ffns)
    det.fit(E_all[train_idx], P_all[train_idx], y[train_idx], copying_scores)
    metrics, scores = det.evaluate(E_all[test_idx], P_all[test_idx], y[test_idx])

    print("\n=== ReDeEP(token) on held-out split ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  selected Copying Heads (layer,head): {det.heads_A}")
    print(f"  selected Knowledge FFN layers: {det.layers_F}")

    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr
    valid = ~np.isnan(ppls[test_idx])
    if valid.sum() > 2 and len(set(y[test_idx][valid])) > 1:
        auc_ppl = roc_auc_score(y[test_idx][valid], ppls[test_idx][valid])
        print(f"\n=== Perplexity baseline AUC: {auc_ppl:.4f} ===")
    valid2 = ~np.isnan(lnents[test_idx])
    if valid2.sum() > 2 and len(set(y[test_idx][valid2])) > 1:
        auc_ent = roc_auc_score(y[test_idx][valid2], lnents[test_idx][valid2])
        print(f"=== LN-Entropy baseline AUC: {auc_ent:.4f} ===")


if __name__ == "__main__":
    main()
