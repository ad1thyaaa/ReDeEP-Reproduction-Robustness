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

TEX = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage[hidelinks]{hyperref}
\usepackage{cite}

\begin{document}

\title{ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability\\[4pt]
\large A Reproduction and Robustness Study}

\author{\IEEEauthorblockN{Gourilakshmi S}
\IEEEauthorblockA{23BCS21\\
gourilakshmis78@gmail.com}
\and
\IEEEauthorblockN{Adithya K}
\IEEEauthorblockA{23BCS51\\
aditkrishna95@gmail.com}}

\maketitle

\begin{abstract}
We reproduce ReDeEP, a mechanistic-interpretability method for detecting hallucinations in retrieval-augmented generation (RAG), on the RAGTruth benchmark using a single 6\,GB laptop GPU. On the paper's own backbone, Llama-2-7B-chat, our detector reaches an AUC of <<auc7>> against the published 0.7325. Across four LLaMA-family models from 1.1B to 7B parameters, ReDeEP outperforms perplexity and length-normalised entropy baselines by 0.06 to 0.11 AUC. We then stress-test the method: replacing the retrieved passage with an irrelevant one raises the hallucination score sharply, but corrupting the passage's numbers, so that the response contradicts it, leaves the score unchanged or even lowers it. ReDeEP therefore measures whether a response is \emph{about} its context, not whether it \emph{agrees} with it. Finally, prompted black-box judges from the Gemini and GPT families (AUC <<aucG>> and <<aucO>>) outperform ReDeEP (<<aucR>>) on the same held-out responses, while a small 3B LLaMA judge performs at chance (<<aucL>>).
\end{abstract}

\begin{IEEEkeywords}
hallucination detection, retrieval-augmented generation, mechanistic interpretability, large language models, robustness
\end{IEEEkeywords}

\section{Introduction}
Retrieval-augmented generation grounds a language model's answer in a retrieved passage, yet models still produce claims that the passage does not support. Such RAG hallucinations are difficult to catch with output-only signals such as perplexity, because a fluent, confident answer can still ignore its context.

ReDeEP~\cite{sun2025redeep} explains RAG hallucinations mechanistically. \emph{Copying heads}---attention heads that move context tokens into the residual stream---carry external knowledge, while \emph{knowledge FFNs}---feed-forward layers---inject the model's parametric knowledge. A hallucination occurs when the model under-uses the former and over-relies on the latter. ReDeEP converts both into scores and regresses the hallucination label on them.

Our earlier review of the method identified a gap: ReDeEP assumes the retrieved context is accurate and relevant, and is never tested under noisy or wrong retrieval, a common real-world failure. This project reproduces the method and then targets that gap directly. It makes three contributions:
\begin{enumerate}
\item A faithful reproduction of ReDeEP's token-level detector on RAGTruth~\cite{niu2024ragtruth} on consumer hardware (RTX~4050, 6\,GB VRAM), including memory optimisations that make 7B models feasible.
\item A transfer study across four LLaMA-family backbones~\cite{touvron2023llama2} from 1.1B to 7B parameters.
\item An extension that stress-tests ReDeEP under corrupted retrieval, and a comparison against black-box LLM judges from the GPT, Gemini and LLaMA families.
\end{enumerate}

\section{Method}
\begin{figure}[t]
\centering
\begin{tikzpicture}[font=\footnotesize, box/.style={draw=black!60, rounded corners=2pt, align=center, minimum height=8mm, inner sep=3pt}, >={Stealth[length=4pt]}]
\node[box, text width=1.45cm] (in) {Prompt +\\response};
\node[box, text width=2.0cm, right=0.45cm of in, yshift=6.5mm] (ecs) {Copying heads\\$\rightarrow$ ECS $E^{l,h}$};
\node[box, text width=2.0cm, right=0.45cm of in, yshift=-6.5mm] (pks) {Knowledge FFNs\\$\rightarrow$ PKS $P^{l}$};
\node[box, text width=1.6cm, right=0.45cm of ecs, yshift=-6.5mm, fill=orange!20] (out) {Logistic\\regression};
\draw[->] (in.east) -- (ecs.west); \draw[->] (in.east) -- (pks.west);
\draw[->] (ecs.east) -- (out.west); \draw[->] (pks.east) -- (out.west);
\end{tikzpicture}
\caption{ReDeEP pipeline: one forward pass yields an external-context score per head and a parametric-knowledge score per FFN layer; a logistic regression combines them into a hallucination score.}
\label{fig:pipeline}
\end{figure}

Fig.~\ref{fig:pipeline} summarises the detector. All quantities are computed from a single forward pass over the prompt (instruction and retrieved passage) followed by the response; $R$ denotes the set of response token positions.

\subsection{External Context Score}
For head $h$ in layer $l$ and response token $n$, let $\mathcal{T}_n$ be the 10\% of context tokens the head attends to most. With $x^{L}_i$ the final-layer hidden state of token $i$,
\begin{equation}
E^{l,h} = \frac{1}{|R|}\sum_{n\in R}\cos\!\Big(\tfrac{1}{|\mathcal{T}_n|}\textstyle\sum_{i\in\mathcal{T}_n} x^{L}_{i},\; x^{L}_{n}\Big).
\end{equation}
A high ECS means the response is built from what the head read in the context.

\subsection{Parametric Knowledge Score}
For layer $l$, the residual stream just before ($x^{\mathrm{mid},l}$) and after ($x^{l}$) the FFN block is projected to the vocabulary with the LogitLens~\cite{logitlens}:
\begin{equation}
P^{l} = \frac{1}{|R|}\sum_{n\in R}\mathrm{JSD}\big(p^{\mathrm{mid},l}_n,\; p^{l}_n\big),
\end{equation}
where $p_n = \sigma(W_U\,\mathrm{norm}(x_n))$ is the LogitLens next-token distribution of a residual state, $\sigma$ the softmax and JSD the Jensen--Shannon divergence. A large shift indicates that the FFN injected its own knowledge.

\subsection{Detector and Baselines}
Copying heads are ranked by their OV circuit, scored on 500 sampled vocabulary tokens (large trace, few Gershgorin-disc outliers); the top 8 form set $\mathcal{A}$. On each training fold, the 5 layers whose PKS correlates most positively with the label form set $\mathcal{F}$. A logistic regression is fitted on the standardised features $\bar{P}_{\mathcal{F}}$ and $-\bar{E}_{\mathcal{A}}$, with the decision threshold set at the best-F1 point on the training fold. Baselines are the response perplexity and LN-Entropy~\cite{malinin2021uncertainty} (mean per-token predictive entropy) from the same forward pass.

\subsection{Black-Box Judges}
Each judge receives the RAG input and the response and must reply in JSON with the probability (0--100) that the response contains a claim the passage contradicts or does not support, judging against the passage only. The stated probability is used as the score.

\section{Experimental Setup}
All models score the same <<n>> responses: the RAGTruth test split restricted to responses generated by llama-2-7b-chat, the paper's setting, capped at 6{,}000 characters; <<nh>> are hallucinated. Table~\ref{tab:backbones} lists the backbones.

\begin{table}[t]
\caption{Backbones (ungated Hugging Face mirrors with identical weights)}
\label{tab:backbones}
\centering\footnotesize
\begin{tabular}{@{}lll@{}}
\toprule
Backbone & Params & Precision \\
\midrule
TinyLlama-1.1B-Chat & 1.1B & fp16 \\
Llama-3.2-1B-Instruct & 1.2B & fp16 \\
Llama-3.2-3B-Instruct & 3.2B & 4-bit NF4 \\
Llama-2-7B-chat & 6.7B & 4-bit NF4 \\
\bottomrule
\end{tabular}
\end{table}

Experiments ran on a Lenovo LOQ laptop (RTX~4050, 6\,GB VRAM, 16\,GB RAM) with PyTorch~2.11, transformers~5.17 and bitsandbytes~0.50. Sequences are capped at 1{,}024 tokens, truncating the prompt from the left so that the whole response is scored. We report the mean and standard deviation over stratified 5-fold cross-validation; heads and layers are re-selected inside each training fold, so no test labels leak into selection.

\textbf{Engineering for 6\,GB.} The reference implementation requests every full attention map, which does not fit for 7B models. We replaced it with forward hooks that keep only each layer's top-$k$ context indices, and compute the attended mean as a sparse matrix product. On a RAGTruth example the rewritten ECS matches the original to within $6\times10^{-8}$. Scoring takes about 0.6\,s per example for TinyLlama and 4--7\,s for Llama-2-7B.

\section{Results}
\subsection{Hallucination Detection}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/detection_auc.pdf}
\caption{Detection AUC per backbone. ReDeEP (orange) beats both baselines on every model; the dashed line marks the paper's Llama-2-7B result.}
\label{fig:detection}
\end{figure}

\begin{table}[t]
\caption{Detection results, 5-fold CV mean ($n=<<n>>$)}
\label{tab:detection}
\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
& \multicolumn{3}{c}{ReDeEP} & PPL & LN-Ent \\
\cmidrule(lr){2-4}
Backbone & AUC & PCC & F1 & AUC & AUC \\
\midrule
<<det_rows>>
\midrule
Paper (Llama-2-7B) & 0.7325 & 0.3979 & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

ReDeEP reproduces on the paper's backbone: Llama-2-7B reaches an AUC of <<auc7>> ($\pm$<<sd7>>) against the published 0.7325, and a PCC of <<pcc7>> against 0.3979 (Table~\ref{tab:detection}, Fig.~\ref{fig:detection}). ReDeEP beats both baselines on every backbone, by 0.06 (Llama-3.2-3B, vs.\ LN-Entropy) to 0.10 (Llama-3.2-1B, vs.\ perplexity) over the better baseline. Detection quality is roughly flat in model size: the 1.1B TinyLlama scores highest, and the 3B and 7B models lie within one standard deviation of each other. The baselines also vary more across folds (std 0.06--0.08) than ReDeEP does (0.03--0.05).

\subsection{Black-Box LLM Judges}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/judges_auc.pdf}
\caption{All detectors on the same 124 held-out responses. The API judges are more accurate; a small local judge is at chance.}
\label{fig:judges}
\end{figure}

\begin{table}[t]
\caption{Judges vs.\ ReDeEP on the same 124 held-out responses}
\label{tab:judges}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
Detector & Kind & AUC & PCC & Rec. & F1 \\
\midrule
<<judge_rows>>
\bottomrule
\end{tabular}
\end{table}

Large prompted judges outperform ReDeEP (Table~\ref{tab:judges}, Fig.~\ref{fig:judges}): Gemini 3.5 Flash-Lite reaches <<aucG>> and OpenAI's open-weights gpt-oss-120b~<<aucO>>, against <<aucR>> for ReDeEP on the same 124 responses. On all <<nG>> responses the Gemini judge scores <<aucGall>>, consistent with the held-out result. Because the judges read the passage and the response as text, they can spot a contradicted fact directly---exactly ReDeEP's blind spot (Section~\ref{sec:robust}). A small prompted judge fails: Llama-3.2-3B answers ``no hallucination'' almost always (AUC <<aucL>>), whereas reading the internals of a model of the same size (ReDeEP on Llama-3.2-3B, AUC 0.727) works far better. ReDeEP's advantage is cost and access: one forward pass of a small open model, with no API, no per-call cost and no data leaving the machine.

\section{Extension: Robustness Under Corrupted Retrieval}
\label{sec:robust}
We hold each response fixed and change only the retrieved passage: (i) \emph{accurate}, the real passage; (ii) \emph{irrelevant}, a passage from a different source of the same task type; (iii) \emph{numbers corrupted}, every number in the real passage multiplied by a random factor in $[1.3, 3.0]$, so that numbers the response copied are now contradicted. The detector is refitted on the remaining 265 examples, so all 150 perturbed responses are held out.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/robustness_shift.pdf}
\caption{Change in mean ReDeEP score relative to the accurate passage (paired Wilcoxon test, 150 responses per backbone).}
\label{fig:robust}
\end{figure}

\begin{table}[t]
\caption{Robustness: mean score and signals per condition}
\label{tab:robust}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
Backbone & Condition & Score & ECS & PKS & AUC \\
\midrule
<<rob_rows>>
\bottomrule
\end{tabular}
\end{table}

\textbf{Irrelevant passages work as intended.} The score jumps for nearly every response (<<tIrr>> for TinyLlama, <<sIrr>> for Llama-2-7B, both $p<0.001$), so the original labels stop separating the responses (AUC $\approx$0.55; Fig.~\ref{fig:robust}, Table~\ref{tab:robust}). This is correct behaviour, because nothing in the response is supported by the new passage. The mechanism differs by size: in TinyLlama ECS falls (<<eAccT>> to <<eIrrT>>), whereas in Llama-2-7B ECS is flat and the increase comes from PKS (<<pAcc7>> to <<pIrr7>>).

\textbf{Corrupted numbers go unnoticed.} For TinyLlama the score does not move (<<tCon>>, <<tConP>>). For Llama-2-7B it falls slightly (<<sCon>>, <<sConP>>) while ECS rises (<<eAcc7>> to <<eCon7>>): the detector rates a contradicted response as \emph{more} faithful. Restricted to the <<nNum>> responses that state a number the corrupted passage now contradicts, the score rose in only <<fT>>\% of cases for TinyLlama and <<f7>>\% for Llama-2-7B.

\textbf{Interpretation.} ECS is a cosine similarity between hidden states: it measures whether the response is built from what the copying heads read, not whether the facts match. A plausible explanation is that a changed number barely moves the hidden state, so ReDeEP cannot see it. The method thus catches topical hallucinations but needs a complementary fact-level check for numeric or entity contradictions.

\section{Limitations}
\begin{itemize}
\item \emph{Quantisation:} the 3B and 7B models ran in 4-bit NF4, so their ECS and PKS approximate full precision; the 7B match with the paper suggests the effect is small.
\item \emph{Sampled copying-head scores:} OV circuits are scored on 500 sampled vocabulary tokens rather than the full vocabulary.
\item \emph{One generator:} all responses come from llama-2-7b-chat, so the other backbones score text they did not write.
\item \emph{Sample size:} fold standard deviations are 0.03--0.05 AUC; smaller differences are within noise.
\item \emph{Robustness scope:} only numbers are corrupted; entity swaps and negations are untested.
\item \emph{Judge access:} judges ran on free API tiers; the GPT-family judge is the open-weights gpt-oss-120b, evaluated on the 124 held-out responses because of rate limits.
\end{itemize}

\section{Conclusion}
ReDeEP reproduces on a laptop GPU (AUC <<auc7>> vs.\ 0.7325) and beats uncertainty baselines on four LLaMA models from 1.1B to 7B parameters. Its signals catch responses that are off-topic for their context but not responses that contradict their context's numbers. Large prompted judges are more accurate but require an API; small prompted judges do not work. ReDeEP is therefore best used as a cheap, private first-line detector for topical hallucinations, paired with a fact-level check. Future work: entity and negation corruptions, a numeric-consistency feature combined with ECS, full-precision runs on larger GPUs with backbone-matched responses, and judge evaluation on all responses with repeated sampling.

\begin{thebibliography}{9}
\bibitem{sun2025redeep} Z.~Sun \emph{et al.}, ``ReDeEP: Detecting hallucination in retrieval-augmented generation via mechanistic interpretability,'' in \emph{Proc. Int. Conf. Learning Representations (ICLR)}, 2025.
\bibitem{niu2024ragtruth} C.~Niu \emph{et al.}, ``RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models,'' in \emph{Proc. Annu. Meeting Assoc. Comput. Linguistics (ACL)}, 2024.
\bibitem{touvron2023llama2} H.~Touvron \emph{et al.}, ``Llama 2: Open foundation and fine-tuned chat models,'' arXiv:2307.09288, 2023.
\bibitem{malinin2021uncertainty} A.~Malinin and M.~Gales, ``Uncertainty estimation in autoregressive structured prediction,'' in \emph{Proc. ICLR}, 2021.
\bibitem{logitlens} nostalgebraist, ``Interpreting GPT: the logit lens,'' LessWrong, 2020.
\end{thebibliography}

\end{document}
"""
for k, v in V.items():
    TEX = TEX.replace(f"<<{k}>>", str(v))
assert "<<" not in TEX, TEX[TEX.index("<<") - 40: TEX.index("<<") + 40]
(HERE / "redeep_paper.tex").write_text(TEX, encoding="utf-8")
print("wrote", HERE / "redeep_paper.tex")
