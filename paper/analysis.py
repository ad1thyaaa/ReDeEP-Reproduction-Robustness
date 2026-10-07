"""Extra CPU-only analyses on the cached scores for the full report:
mechanistic-finding replication, ROC curves, per-task AUC, ablations,
score distributions, compute cost. Writes paper/analysis.json + figures.

    python paper/analysis.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redeep.data import load_examples
from redeep.detect import ReDeEPDetector, aggregate_features, select_copying_heads, select_knowledge_ffns

DATA, FIG = ROOT / "data", ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)
TAGS = [("TinyLlama-1.1B", "tinyllama_1.1b"), ("Llama-3.2-1B", "llama3.2_1b"),
        ("Llama-3.2-3B", "llama3.2_3b_4bit"), ("Llama-2-7B", "llama2_7b_4bit")]
AMBER, GREY, LIGHT, BLUE, INK = "#E8912D", "#7F8C8B", "#BFC8C7", "#2E6F9E", "#1B1B1B"
COLS = ["#1F4E79", "#2E8B8B", "#8C6BB1", AMBER]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


pool = {e["id"]: e for e in load_examples(split="test", models=["llama-2-7b-chat"])}
out = {}


def oof_scores(d, cop, k=8, m=5, feats="both"):
    """Out-of-fold ReDeEP scores with the same 5-fold split as run_pipeline.py."""
    E, P, y = d["E"], d["P"], d["y"]
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        det = ReDeEPDetector(k, m).fit(E[tr], P[tr], y[tr], cop)
        if feats == "both":
            s[te] = det.score(E[te], P[te])
        else:
            from sklearn.linear_model import LogisticRegression
            e, p = aggregate_features(E, P, det.heads_A, det.layers_F)
            X = (p if feats == "pks" else -e).reshape(-1, 1)
            clf = LogisticRegression().fit(X[tr], y[tr])
            s[te] = clf.predict_proba(X[te])[:, 1]
    return s


scores = {}
for name, tag in TAGS:
    d = dict(np.load(DATA / f"scores_{tag}.npz"))
    cop = np.load(DATA / f"copying_heads_{tag}.npy")
    y, E, P = d["y"], d["E"], d["P"]
    tasks = np.array([pool[str(i)]["task_type"] for i in d["ids"]])
    r = {"n_layers": int(E.shape[1]), "n_heads": int(E.shape[2])}

    # --- replication of the paper's mechanistic findings
    e_t, e_h = E[y == 0].mean(0), E[y == 1].mean(0)            # [L, H]
    r["heads_ecs_truth_gt_halluc"] = int((e_t > e_h).sum())
    r["heads_total"] = int(e_t.size)
    A = select_copying_heads(cop, 8)
    r["copying_heads_ecs_truth_gt_halluc"] = int(sum(e_t[l, h] > e_h[l, h] for l, h in A))
    p_t, p_h = P[y == 0].mean(0), P[y == 1].mean(0)            # [L]
    r["pks_diff_by_layer"] = (p_h - p_t).tolist()
    half = E.shape[1] // 2
    r["pks_halluc_gt_truth_late_layers"] = int((p_h[half:] > p_t[half:]).sum())
    r["pks_halluc_gt_truth_early_layers"] = int((p_h[:half] > p_t[:half]).sum())
    _, corrs = select_knowledge_ffns(P, y, 5)
    r["pks_label_corr_by_layer"] = [float(c) for c in corrs]
    r["ecs_head_diff"] = (e_t - e_h).tolist()

    # --- out-of-fold scores, ROC, per-task AUC
    s = oof_scores(d, cop)
    scores[name] = (y, s, d["ppl"], d["lnent"], tasks)
    r["oof_auc"] = float(roc_auc_score(y, s))
    r["per_task"] = {}
    for t in ("QA", "Summary", "Data2txt"):
        msk = tasks == t
        r["per_task"][t] = {"n": int(msk.sum()), "n_halluc": int(y[msk].sum()),
                            "redeep": float(roc_auc_score(y[msk], s[msk])),
                            "ppl": float(roc_auc_score(y[msk], d["ppl"][msk])),
                            "lnent": float(roc_auc_score(y[msk], d["lnent"][msk]))}

    # --- ablations
    r["feature_ablation"] = {f: float(roc_auc_score(y, oof_scores(d, cop, feats=f))) for f in ("ecs", "pks", "both")}
    ks, ms = [1, 2, 4, 8, 16, 32], [1, 3, 5, 10]
    r["grid_k"], r["grid_m"] = ks, ms
    r["grid_auc"] = [[float(roc_auc_score(y, oof_scores(d, cop, k=k, m=m))) for m in ms] for k in ks]
    out[name] = r
    print(name, "done", r["heads_ecs_truth_gt_halluc"], "/", r["heads_total"], "oof", round(r["oof_auc"], 3))

# --- compute cost from logs
cost = {}
for name, tag in TAGS:
    txt = (DATA / "logs" / f"{tag}.log").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"scored (\d+) examples in ([\d.]+) min", txt)
    per = [float(x) for x in re.findall(r"took ([\d.]+)s", txt)]
    ch = re.search(r"done in ([\d.]+)s\. top head", txt)
    cost[name] = {"minutes": float(m.group(2)) if m else None, "sec_per_example_median": float(np.median(per)) if per else None,
                  "copying_head_seconds": float(ch.group(1)) if ch else None}
out["_cost"] = cost

# --- judge score distributions
def judge_scores(f):
    rows = [json.loads(l) for l in open(DATA / f, encoding="utf-8")]
    return [(r["label"], r["score"]) for r in rows if r.get("score") is not None]
out["_judge_score_values"] = {}
for name, f in (("Gemini 3.5 Flash-Lite", "judge_gemini_gemini-3.5-flash-lite.jsonl"),
                ("GPT gpt-oss-120b", "judge_groq_openai_gpt-oss-120b.jsonl"),
                ("Llama-3.2-3B", "judge_llama_unsloth_Llama-3.2-3B-Instruct.jsonl")):
    v = judge_scores(f)
    sc = np.array([s for _, s in v])
    out["_judge_score_values"][name] = {"n": len(v), "frac_0_or_100": float(np.mean((sc == 0) | (sc == 1))),
                                       "distinct_values": int(len(set(sc.round(2))))}
(ROOT / "paper" / "analysis.json").write_text(json.dumps(out, indent=1))

# ================================================================= figures
# ROC curves
fig, ax = plt.subplots(figsize=(3.45, 2.8))
for (name, _), c in zip(TAGS, COLS):
    y, s, *_ = scores[name]
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=c, lw=1.2, label=f"ReDeEP, {name} ({roc_auc_score(y, s):.3f})")
y, s, ppl, ln, _ = scores["Llama-2-7B"]
for v, lab, ls in ((ppl, "Perplexity, Llama-2-7B", "--"), (ln, "LN-Entropy, Llama-2-7B", ":")):
    fpr, tpr, _ = roc_curve(y, v)
    ax.plot(fpr, tpr, color=GREY, lw=1, ls=ls, label=f"{lab} ({roc_auc_score(y, v):.3f})")
ax.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(loc="lower right", frameon=False, fontsize=6)
clean(ax); fig.tight_layout(pad=0.3); fig.savefig(FIG / "roc.pdf"); plt.close(fig)

# Per-task AUC (grouped bars)
fig, ax = plt.subplots(figsize=(3.45, 2.1))
tasks = ["QA", "Summary", "Data2txt"]
w = 0.2
for i, ((name, _), c) in enumerate(zip(TAGS, COLS)):
    vals = [out[name]["per_task"][t]["redeep"] for t in tasks]
    ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, color=c, label=name)
ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
ax.set_xticks(range(3), [f"{t}\n(n={out['Llama-2-7B']['per_task'][t]['n']})" for t in tasks])
ax.set_ylim(0.4, 0.9); ax.set_ylabel("ReDeEP AUC (out-of-fold)")
ax.legend(ncol=2, frameon=False, fontsize=6, loc="upper left")
clean(ax); fig.tight_layout(pad=0.3); fig.savefig(FIG / "per_task.pdf"); plt.close(fig)

# PKS difference by normalised layer depth
fig, ax = plt.subplots(figsize=(3.45, 2.1))
for (name, _), c in zip(TAGS, COLS):
    dlt = np.array(out[name]["pks_diff_by_layer"])
    x = np.linspace(0, 1, len(dlt))
    ax.plot(x, dlt / np.abs(dlt).max(), color=c, lw=1.1, label=name)
ax.axhline(0, color=INK, lw=0.6)
ax.set_xlabel("Layer depth (0 = first, 1 = last)"); ax.set_ylabel("PKS(halluc.) $-$ PKS(truthful)\n(scaled per model)")
ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
clean(ax); fig.tight_layout(pad=0.3); fig.savefig(FIG / "pks_by_layer.pdf"); plt.close(fig)

# ECS truthful-minus-hallucinated heatmap, Llama-2-7B
dlt = np.array(out["Llama-2-7B"]["ecs_head_diff"])
fig, ax = plt.subplots(figsize=(3.45, 2.4))
lim = np.abs(dlt).max()
im = ax.imshow(dlt, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim, origin="lower")
ax.set_xlabel("Attention head"); ax.set_ylabel("Layer")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6)
cb.set_label("ECS(truthful) $-$ ECS(halluc.)", fontsize=7)
fig.tight_layout(pad=0.3); fig.savefig(FIG / "ecs_heatmap.pdf"); plt.close(fig)

# Ablation heatmap, Llama-2-7B
g = np.array(out["Llama-2-7B"]["grid_auc"])
fig, ax = plt.subplots(figsize=(3.2, 2.2))
im = ax.imshow(g, cmap="YlOrBr", aspect="auto")
ax.set_xticks(range(len(out["Llama-2-7B"]["grid_m"])), out["Llama-2-7B"]["grid_m"])
ax.set_yticks(range(len(out["Llama-2-7B"]["grid_k"])), out["Llama-2-7B"]["grid_k"])
ax.set_xlabel("Knowledge FFN layers $|\\mathcal{F}|$"); ax.set_ylabel("Copying heads $|\\mathcal{A}|$")
for i in range(g.shape[0]):
    for j in range(g.shape[1]):
        ax.text(j, i, f"{g[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                fontweight="bold" if (i, j) == (3, 2) else "normal")
fig.tight_layout(pad=0.3); fig.savefig(FIG / "ablation_grid.pdf"); plt.close(fig)

# Score distributions, Llama-2-7B
y, s, *_ = scores["Llama-2-7B"]
fig, ax = plt.subplots(figsize=(3.45, 1.9))
bins = np.linspace(0, 1, 21)
ax.hist(s[y == 0], bins, color=BLUE, alpha=0.6, label=f"Truthful (n={int((y == 0).sum())})")
ax.hist(s[y == 1], bins, color=AMBER, alpha=0.6, label=f"Hallucinated (n={int((y == 1).sum())})")
ax.set_xlabel("ReDeEP hallucination score (out-of-fold), Llama-2-7B"); ax.set_ylabel("Responses")
ax.legend(frameon=False)
clean(ax); fig.tight_layout(pad=0.3); fig.savefig(FIG / "score_hist.pdf"); plt.close(fig)
print("figures written")
