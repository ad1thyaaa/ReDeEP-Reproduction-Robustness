"""Builds the IEEE-format final report (paper/redeep_paper.tex + figures)
from the result files in data/. Compile with Tectonic:

    python paper/build_paper.py
    tectonic paper/redeep_paper.tex
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
load = lambda f: json.loads((DATA / f).read_text())

BACKBONES = [("TinyLlama-1.1B", "tinyllama_1.1b"), ("Llama-3.2-1B", "llama3.2_1b"),
             ("Llama-3.2-3B (4-bit)", "llama3.2_3b_4bit"), ("Llama-2-7B (4-bit)", "llama2_7b_4bit")]
R = {n: load(f"results_{t}.json") for n, t in BACKBONES}
l2 = R["Llama-2-7B (4-bit)"]
s2 = l2["single_split"]
robT = load("robustness_tinyllama_1.1b.json")["conditions"]
rob7 = load("robustness_llama2_7b_4bit.json")["conditions"]
numT = load("robustness_numeric_subset_tinyllama_1.1b.json")["response_numbers_contradicted"]
num7 = load("robustness_numeric_subset_llama2_7b_4bit.json")["response_numbers_contradicted"]
SPLIT = "same_test_split_as_llama2_7b_4bit"
JG = load("results_judge_gemini_gemini-3.5-flash-lite.json")
JO = load("results_judge_groq_openai_gpt-oss-120b.json")
JL = load("results_judge_llama_unsloth_Llama-3.2-3B-Instruct.json")
f3 = lambda v: f"{v:.3f}"
pm = lambda m: f"{m[0]:.3f} $\\pm$ {m[1]:.3f}"
pfmt = lambda p: "$p<0.001$" if p < 0.001 else f"$p={p:.2f}$"

AMBER, GREY, LIGHT, INK = "#E8912D", "#7F8C8B", "#BFC8C7", "#1B1B1B"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------- Fig. 2: detection AUC
fig, ax = plt.subplots(figsize=(3.45, 2.45))
names = [n for n, _ in BACKBONES]
for k, (lab, key, col) in enumerate([("ReDeEP", "AUC", AMBER), ("Perplexity", "PPL_AUC", GREY), ("LN-Entropy", "LNEnt_AUC", LIGHT)]):
    ys = [i + k * 0.24 for i in range(len(names))]
    m = [R[n]["cv"][key][0] for n in names]
    e = [R[n]["cv"][key][1] for n in names]
    ax.errorbar(m, ys, xerr=e, fmt="o", ms=4, color=col, ecolor=col, elinewidth=1, capsize=0, label=lab)
    if lab == "ReDeEP":
        for x, y, d in zip(m, ys, e):
            ax.text(x + d + 0.005, y, f3(x), va="center", fontsize=6.5, fontweight="bold")
ax.axvline(0.7325, color="#444444", ls="--", lw=0.8)
ax.text(0.7365, len(names) - 0.52, "← paper: 0.7325", ha="left", va="center", fontsize=6.5, color="#444444")
ax.set_yticks([i + 0.24 for i in range(len(names))], names)
ax.invert_yaxis()
ax.set_xlim(0.5, 0.86)
ax.set_xlabel("AUC (mean $\\pm$ std over 5 folds; 0.5 = chance)")
ax.grid(axis="x", color="#E6E6E6", lw=0.5)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, handletextpad=0.2)
clean(ax)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "detection_auc.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: robustness shift
fig, ax = plt.subplots(figsize=(3.45, 1.65))
y, ticks = 0, []
for name, c in (("TinyLlama-1.1B", robT), ("Llama-2-7B", rob7)):
    for cond, col, lab in (("irrelevant", GREY, "Irrelevant passage"), ("contradictory", AMBER, "Numbers corrupted")):
        v, p = c[cond]["score_shift_vs_accurate"], c[cond]["score_shift_wilcoxon_p"]
        ax.barh(y, v, color=col, height=0.6, label=lab if name == "TinyLlama-1.1B" else None)
        ax.text(max(v, 0) + 0.006, y, f"{v:+.3f} ({'p<0.001' if p < 0.001 else f'p={p:.2f}'})", va="center", fontsize=6.5)
        y += 0.7
    ticks.append(y - 1.05)
    y += 0.45
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(ticks, ["TinyLlama-1.1B", "Llama-2-7B"])
ax.invert_yaxis()
ax.set_xlim(-0.05, 0.47)
ax.set_xlabel("Change in mean hallucination score vs. accurate passage")
ax.legend(loc="center right", frameon=False)
clean(ax)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "robustness_shift.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: judges
rows = [("ReDeEP (Llama-2-7B)", s2["AUC"]), ("GPT: gpt-oss-120b", JO[SPLIT]["AUC"]),
        ("Gemini 3.5 Flash-Lite", JG[SPLIT]["AUC"]), ("LN-Entropy", s2["LNEnt_AUC"]),
        ("Perplexity", s2["PPL_AUC"]), ("Llama-3.2-3B judge", JL[SPLIT]["AUC"])]
fig, ax = plt.subplots(figsize=(3.45, 1.65))
for i, (n, v) in enumerate(rows):
    ax.barh(i, v - 0.4, left=0.4, color=AMBER if i == 0 else GREY, height=0.6)
    ax.text(v + 0.006, i, f3(v), va="center", fontsize=6.5)
ax.axvline(0.5, color="#444444", ls=":", lw=0.8)
ax.set_yticks(range(len(rows)), [n for n, _ in rows])
ax.invert_yaxis()
ax.set_xlim(0.4, 0.9)
ax.set_xlabel("AUC on the same 124 held-out responses")
clean(ax)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "judges_auc.pdf")
plt.close(fig)

# ---------------------------------------------------------------- LaTeX
det_rows = "\n".join(
    f"{n} & \\textbf{{{f3(R[n]['cv']['AUC'][0])}}} & {f3(R[n]['cv']['PCC'][0])} & {f3(R[n]['cv']['F1'][0])} & "
    f"{f3(R[n]['cv']['PPL_AUC'][0])} & {f3(R[n]['cv']['LNEnt_AUC'][0])} \\\\" for n, _ in BACKBONES)
rob_rows = []
for name, c in (("TinyLlama", robT), ("Llama-2-7B", rob7)):
    for cond, lab in (("accurate", "Accurate"), ("irrelevant", "Irrelevant"), ("contradictory", "Numbers corr.")):
        m = c[cond]
        rob_rows.append(f"{name} & {lab} & {f3(m['score_mean'])} & {f3(m['E_A_mean'])} & {f3(m['P_F_mean'])} & "
                        f"{f3(m['AUC_vs_original_labels'])} \\\\")
    rob_rows.append("\\midrule")
rob_rows = "\n".join(rob_rows[:-1])
j = lambda d: d[SPLIT]
judge_rows = "\n".join([
    f"Gemini 3.5 Flash-Lite & Judge & \\textbf{{{f3(j(JG)['AUC'])}}} & {f3(j(JG)['PCC'])} & {f3(j(JG)['Rec'])} & {f3(j(JG)['F1'])} \\\\",
    f"GPT: gpt-oss-120b & Judge & {f3(j(JO)['AUC'])} & {f3(j(JO)['PCC'])} & {f3(j(JO)['Rec'])} & {f3(j(JO)['F1'])} \\\\",
    f"ReDeEP (Llama-2-7B) & Internals & {f3(s2['AUC'])} & {f3(s2['PCC'])} & {f3(s2['Rec'])} & {f3(s2['F1'])} \\\\",
    f"LN-Entropy & Uncertainty & {f3(s2['LNEnt_AUC'])} & -- & -- & -- \\\\",
    f"Perplexity & Uncertainty & {f3(s2['PPL_AUC'])} & -- & -- & -- \\\\",
    f"Llama-3.2-3B (local) & Judge & {f3(j(JL)['AUC'])} & {f3(j(JL)['PCC'])} & {f3(j(JL)['Rec'])} & {f3(j(JL)['F1'])} \\\\"])

V = dict(auc7=f3(l2["cv"]["AUC"][0]), sd7=f3(l2["cv"]["AUC"][1]), pcc7=f3(l2["cv"]["PCC"][0]), n=l2["n"],
         nh=l2["n_hallucinated"], aucG=f3(j(JG)["AUC"]), aucO=f3(j(JO)["AUC"]), aucR=f3(s2["AUC"]),
         aucL=f3(j(JL)["AUC"]), aucGall=f3(JG["all"]["AUC"]), nG=JG["n"],
         tIrr=f"{robT['irrelevant']['score_shift_vs_accurate']:+.3f}", sIrr=f"{rob7['irrelevant']['score_shift_vs_accurate']:+.3f}",
         tCon=f"{robT['contradictory']['score_shift_vs_accurate']:+.3f}", sCon=f"{rob7['contradictory']['score_shift_vs_accurate']:+.3f}",
         tConP=pfmt(robT["contradictory"]["score_shift_wilcoxon_p"]), sConP=pfmt(rob7["contradictory"]["score_shift_wilcoxon_p"]),
         eAcc7=f3(rob7["accurate"]["E_A_mean"]), eCon7=f3(rob7["contradictory"]["E_A_mean"]),
         eAccT=f3(robT["accurate"]["E_A_mean"]), eIrrT=f3(robT["irrelevant"]["E_A_mean"]),
         pAcc7=f3(rob7["accurate"]["P_F_mean"]), pIrr7=f3(rob7["irrelevant"]["P_F_mean"]),
         nNum=num7["n"], fT=round(100 * numT["frac_score_increased"]), f7=round(100 * num7["frac_score_increased"]),
         det_rows=det_rows, rob_rows=rob_rows, judge_rows=judge_rows)

exec((HERE / "report_tail.py").read_text(encoding="utf-8"))
