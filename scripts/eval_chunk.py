"""Evaluate ReDeEP(chunk) features: the official protocol (heads, layers and
weight selected on the evaluation set itself, as in the authors'
chunk_level_reg.py) and a leakage-free nested 5-fold CV.

Works on our data/chunk_<tag>.jsonl files and on the authors' logged
llama2_7B_response_chunk.json. When chunk labels are unavailable (Dolly)
features are averaged per response and ranked at response level.

    python scripts/eval_chunk.py --tag llama2_7b_ragtruth
    python scripts/eval_chunk.py --tag llama2_7b_dolly --response_level
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
GRID = list(itertools.product([1, 2, 3, 4, 6, 8, 12, 16], [1, 2, 3, 4, 6, 8, 12], [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8]))


def load_ours(tag):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"chunk_{tag}.jsonl", encoding="utf-8")]
    return rows


def build(rows, response_level):
    """Return chunk-level (or response-level) feature matrices and ids."""
    E, P, y, rid = [], [], [], []
    for n, r in enumerate(rows):
        Er, Pr = np.array(r["E"]), np.array(r["P"])
        if response_level:
            E.append(Er.mean(0)); P.append(Pr.mean(0)); y.append(r["label"]); rid.append(n)
        else:
            for c in range(len(Er)):
                E.append(Er[c]); P.append(Pr[c]); y.append(r["chunk_labels"][c]); rid.append(n)
    yr = np.array([r["label"] for r in rows])
    return np.array(E), np.array(P), np.array(y), np.array(rid), yr


def feat_auc(X, y, sign):
    if len(set(y)) < 2:
        return np.full(X.shape[1], 0.5)
    return np.array([roc_auc_score(y if sign > 0 else 1 - y, X[:, c]) if np.std(X[:, c]) > 0 else 0.5
                     for c in range(X.shape[1])])


class Model:
    def fit(self, E, P, y, i, j, k):
        ha, la = feat_auc(E, y, -1), feat_auc(P, y, +1)
        self.h, self.l, self.k = np.argsort(-ha)[:i], np.argsort(-la)[:j], k
        es, ps = E[:, self.h].sum(1), P[:, self.l].sum(1)
        self.mm = (es.min(), es.max(), ps.min(), ps.max())
        return self

    def chunk_score(self, E, P):
        emin, emax, pmin, pmax = self.mm
        en = (E[:, self.h].sum(1) - emin) / max(emax - emin, 1e-12)
        pn = (P[:, self.l].sum(1) - pmin) / max(pmax - pmin, 1e-12)
        return pn - self.k * en


def response_scores(model, E, P, rid, resp_ids):
    s = model.chunk_score(E, P)
    return np.array([s[rid == n].mean() for n in resp_ids])


def evaluate(E, P, y, rid, yr, ijk):
    out = {}
    n = len(yr)
    allc = np.arange(len(y))
    m = Model().fit(E, P, y, *ijk)
    s = response_scores(m, E, P, rid, range(n))
    out["official_protocol"] = {"ijk": list(ijk), "AUC": float(roc_auc_score(yr, s)), "PCC": float(pearsonr(s, yr)[0])}
    # best possible when also tuning i, j, k on the evaluation set (upper bound of the leak)
    best = max(GRID, key=lambda g: roc_auc_score(yr, response_scores(Model().fit(E, P, y, *g), E, P, rid, range(n))))
    s = response_scores(Model().fit(E, P, y, *best), E, P, rid, range(n))
    out["tuned_on_test"] = {"ijk": list(best), "AUC": float(roc_auc_score(yr, s))}

    folds, chosen, oof = [], [], np.zeros(n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), yr):
        best_g, best_a = None, -1
        for g in GRID:
            a = []
            for itr, ite in StratifiedKFold(3, shuffle=True, random_state=1).split(np.zeros(len(tr)), yr[tr]):
                ci = np.isin(rid, tr[itr])
                mm = Model().fit(E[ci], P[ci], y[ci], *g)
                a.append(roc_auc_score(yr[tr[ite]], response_scores(mm, E, P, rid, tr[ite])))
            if np.mean(a) > best_a:
                best_g, best_a = g, np.mean(a)
        ci = np.isin(rid, tr)
        mm = Model().fit(E[ci], P[ci], y[ci], *best_g)
        oof[te] = response_scores(mm, E, P, rid, te)
        folds.append(float(roc_auc_score(yr[te], oof[te])))
        chosen.append(list(best_g))
    out["nested_cv"] = {"AUC_mean": float(np.mean(folds)), "AUC_std": float(np.std(folds)), "folds": folds,
                        "chosen_ijk": chosen, "PCC_oof": float(pearsonr(oof, yr)[0])}
    out["n_responses"], out["n_hallucinated"], out["n_chunks"] = int(n), int(yr.sum()), int(len(y))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--official_log", help="authors' *_response_chunk.json instead of our --tag")
    ap.add_argument("--response_level", action="store_true")
    ap.add_argument("--ijk", default="3,4,0.6", help="official hyperparameters (llama2-7b ragtruth: 3,4,0.6)")
    args = ap.parse_args()
    if args.official_log:
        raw = [r for r in json.load(open(args.official_log, encoding="utf-8")) if r["split"] == "test" and r.get("scores")]
        rows = [{"label": max(s["hallucination_label"] for s in r["scores"]),
                 "E": [list(s["prompt_attention_score"].values())[:32] for s in r["scores"]],
                 "P": [list(s["parameter_knowledge_scores"].values())[:32] for s in r["scores"]],
                 "chunk_labels": [s["hallucination_label"] for s in r["scores"]]} for r in raw]
        name = "official_" + Path(args.official_log).stem
    else:
        rows, name = load_ours(args.tag), args.tag
    i, j, k = args.ijk.split(",")
    res = evaluate(*build(rows, args.response_level), (int(i), int(j), float(k)))
    res["response_level_ranking"] = args.response_level
    print(json.dumps({k: v for k, v in res.items() if k != "nested_cv"} | {"nested_cv_AUC": [res["nested_cv"]["AUC_mean"], res["nested_cv"]["AUC_std"]]}, indent=1))
    (ROOT / "data" / f"eval_chunk_{name}.json").write_text(json.dumps(res, indent=1))
