"""How much does the official ReDeEP evaluation benefit from selecting
heads / layers / weights on the test set itself?

Uses the authors' own logged chunk-level scores for Llama-2-7B
(ReDEeP-ICLR/ReDeEP/log/test_llama2_7B/llama2_7B_response_chunk.json) and
compares:
  (a) their protocol: rank the 32 head features and 32 layer features by
      AUC on ALL test chunks, keep the top i / j, combine with weight k
      (i, j, k = 3, 4, 0.6 hard-coded for llama2-7b), report AUC on the
      same responses;
  (b) the same fixed i, j, k but feature ranking done on training folds only;
  (c) i, j, k also chosen by inner search on the training folds only.

    python scripts/leakage_check.py --log <path to llama2_7B_response_chunk.json>
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--i", type=int, default=3)
ap.add_argument("--j", type=int, default=4)
ap.add_argument("--k", type=float, default=0.6)
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "leakage_check.json"))
args = ap.parse_args()

resp = [r for r in json.load(open(args.log, encoding="utf-8")) if r["split"] == "test" and r.get("scores")]
# chunk-level matrices
E, P, yc, rid = [], [], [], []
for n, r in enumerate(resp):
    for s in r["scores"]:
        E.append(list(s["prompt_attention_score"].values())[:32])
        P.append(list(s["parameter_knowledge_scores"].values())[:32])
        yc.append(s["hallucination_label"])
        rid.append(n)
E, P, yc, rid = np.array(E), np.array(P), np.array(yc), np.array(rid)
yr = np.array([int(len(r["labels"]) > 0) for r in resp])   # response-level label
# official code uses max over chunk labels as the response label
yr_chunk = np.array([yc[rid == n].max() for n in range(len(resp))])
print(f"{len(resp)} responses, {len(yc)} chunks; response label agreement {np.mean(yr == yr_chunk):.3f}")


def feat_auc(X, y, sign):
    return np.array([roc_auc_score(y if sign > 0 else 1 - y, X[:, c]) if np.std(X[:, c]) > 0 else 0.5
                     for c in range(X.shape[1])])


def score(E, P, heads, layers, k, mm):
    """Official combination: min-max normalise summed features, P - k*E,
    then average over a response's chunks."""
    es, ps = E[:, heads].sum(1), P[:, layers].sum(1)
    (emin, emax), (pmin, pmax) = mm
    en = (es - emin) / max(emax - emin, 1e-12)
    pn = (ps - pmin) / max(pmax - pmin, 1e-12)
    return pn - k * en


def respond(chunk_scores, idx_chunks, n_resp_ids):
    return np.array([chunk_scores[idx_chunks == n].mean() for n in n_resp_ids])


def fit(train_chunks, i, j):
    ha = feat_auc(E[train_chunks], yc[train_chunks], -1)   # ECS: higher = truthful
    la = feat_auc(P[train_chunks], yc[train_chunks], +1)
    heads, layers = np.argsort(-ha)[:i], np.argsort(-la)[:j]
    es, ps = E[train_chunks][:, heads].sum(1), P[train_chunks][:, layers].sum(1)
    return heads, layers, ((es.min(), es.max()), (ps.min(), ps.max()))


out = {}
# (a) official protocol on all data
all_chunks = np.arange(len(yc))
h, l, mm = fit(all_chunks, args.i, args.j)
s = respond(score(E, P, h, l, args.k, mm), rid, range(len(resp)))
out["official_protocol"] = {"AUC": roc_auc_score(yr_chunk, s), "PCC": pearsonr(s, yr_chunk)[0]}

# (b) / (c) cross-validated over responses
skf = StratifiedKFold(5, shuffle=True, random_state=0)
grid = list(itertools.product([1, 2, 3, 4, 6, 8, 12, 16], [1, 2, 3, 4, 6, 8, 12], [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8]))
fold_b, fold_c, chosen = [], [], []
for tr, te in skf.split(np.zeros(len(resp)), yr_chunk):
    trc, tec = np.isin(rid, tr), np.isin(rid, te)
    # (b) fixed i, j, k; ranking on train only
    h, l, mm = fit(np.where(trc)[0], args.i, args.j)
    fold_b.append(roc_auc_score(yr_chunk[te], respond(score(E, P, h, l, args.k, mm), rid, te)))
    # (c) inner search of i, j, k on train responses (inner 3-fold)
    inner = StratifiedKFold(3, shuffle=True, random_state=1)
    best, best_auc = None, -1
    for i, j, k in grid:
        a = []
        for itr, ite in inner.split(np.zeros(len(tr)), yr_chunk[tr]):
            itr_r, ite_r = tr[itr], tr[ite]
            hh, ll, m2 = fit(np.where(np.isin(rid, itr_r))[0], i, j)
            a.append(roc_auc_score(yr_chunk[ite_r], respond(score(E, P, hh, ll, k, m2), rid, ite_r)))
        if np.mean(a) > best_auc:
            best, best_auc = (i, j, k), np.mean(a)
    h, l, mm = fit(np.where(trc)[0], best[0], best[1])
    fold_c.append(roc_auc_score(yr_chunk[te], respond(score(E, P, h, l, best[2], mm), rid, te)))
    chosen.append(best)
out["cv_fixed_ijk"] = {"AUC_mean": float(np.mean(fold_b)), "AUC_std": float(np.std(fold_b)), "folds": fold_b}
out["cv_nested"] = {"AUC_mean": float(np.mean(fold_c)), "AUC_std": float(np.std(fold_c)), "folds": fold_c,
                    "chosen_ijk": [list(map(float, c)) for c in chosen]}
out["n_responses"], out["n_chunks"] = len(resp), int(len(yc))
print(json.dumps(out, indent=1, default=float))
Path(args.out).write_text(json.dumps(out, indent=1, default=float))
