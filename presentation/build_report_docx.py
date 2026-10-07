"""Builds presentation/ReDeEP_Final_Report.docx (an offline copy of the
final report) from the result files in data/.

    python presentation/build_report_docx.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "ReDeEP_Final_Report.docx"
load = lambda f: json.loads((DATA / f).read_text())

TEAL, AMBER, GREY, LIGHT = "#0F3D3E", "#F2A541", "#8C9796", "#C9D1D0"
BACKBONES = [("TinyLlama-1.1B", "tinyllama_1.1b"), ("Llama-3.2-1B", "llama3.2_1b"),
             ("Llama-3.2-3B (4-bit)", "llama3.2_3b_4bit"), ("Llama-2-7B (4-bit)", "llama2_7b_4bit")]
R = {name: load(f"results_{tag}.json") for name, tag in BACKBONES}
l2 = R["Llama-2-7B (4-bit)"]
robT = load("robustness_tinyllama_1.1b.json")["conditions"]
rob7 = load("robustness_llama2_7b_4bit.json")["conditions"]
numT = load("robustness_numeric_subset_tinyllama_1.1b.json")["response_numbers_contradicted"]
num7 = load("robustness_numeric_subset_llama2_7b_4bit.json")["response_numbers_contradicted"]
SPLIT = "same_test_split_as_llama2_7b_4bit"
JG = load("results_judge_gemini_gemini-3.5-flash-lite.json")
JO = load("results_judge_groq_openai_gpt-oss-120b.json")
JL = load("results_judge_llama_unsloth_Llama-3.2-3B-Instruct.json")
f3 = lambda v: f"{v:.3f}"


# ------------------------------------------------------------------ charts
def chart_detection(path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    names = [n for n, _ in BACKBONES]
    rows = [("ReDeEP", "AUC", AMBER), ("Perplexity", "PPL_AUC", GREY), ("LN-Entropy", "LNEnt_AUC", LIGHT)]
    for k, (label, key, col) in enumerate(rows):
        ys = [i * 1.0 + k * 0.25 for i in range(len(names))]
        m = [R[n]["cv"][key][0] for n in names]
        s = [R[n]["cv"][key][1] for n in names]
        ax.errorbar(m, ys, xerr=s, fmt="o", color=col, ecolor=col, elinewidth=1.5, capsize=0, ms=7, label=label)
        if label == "ReDeEP":
            for x, y, e in zip(m, ys, s):
                ax.text(x + e + 0.006, y, f3(x), va="center", fontsize=9, fontweight="bold")
    ax.axvline(0.7325, color="#555555", ls="--", lw=1)
    ax.text(0.7325, -0.55, "Paper, Llama-2-7B: 0.7325", ha="center", fontsize=8.5, color="#555555")
    ax.set_yticks([i + 0.25 for i in range(len(names))], names)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 0.86)
    ax.set_xlabel("AUC, mean and std over 5 folds (0.50 = chance)")
    ax.set_title("ReDeEP beats both baselines on every backbone", loc="left", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_robustness(path):
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=200)
    rows = [("TinyLlama-1.1B", robT), ("Llama-2-7B (4-bit)", rob7)]
    y = 0
    ticks, labels = [], []
    for name, c in rows:
        for cond, col, lab in (("irrelevant", GREY, "Irrelevant passage"), ("contradictory", AMBER, "Numbers corrupted")):
            v, p = c[cond]["score_shift_vs_accurate"], c[cond]["score_shift_wilcoxon_p"]
            ax.barh(y, v, color=col, height=0.6, label=lab if name == rows[0][0] else None)
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
            ax.text(max(v, 0) + 0.006, y, f"{v:+.3f}  ({ptxt})", va="center", fontsize=9)
            y += 0.7
        ticks.append(y - 1.05)
        labels.append(name)
        y += 0.5
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 0.45)
    ax.set_xlabel("Change in mean ReDeEP hallucination score vs. accurate passage (150 responses)")
    ax.set_title("An irrelevant passage raises the score; corrupted numbers do not", loc="left", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ doc helpers
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
for lvl, size in ((1, 15), (2, 13)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Cambria"
    h.font.size = Pt(size)
    h.font.color.rgb = RGBColor(0x0F, 0x3D, 0x3E)


def para(text, bold_lead=None, italic=False):
    p = doc.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead + " ").bold = True
    r = p.add_run(text)
    r.italic = italic
    return p


def bullets(items, style="List Bullet"):
    for it in items:
        if isinstance(it, tuple):
            p = doc.add_paragraph(style=style)
            p.add_run(it[0] + " ").bold = True
            p.add_run(it[1])
        else:
            doc.add_paragraph(it, style=style)


def table(header, rows, bold_col=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            if bold_col is not None and i == bold_col:
                for r in cells[i].paragraphs[0].runs:
                    r.bold = True
    doc.add_paragraph()


def formula(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name = "Cambria Math"
    r.italic = True


# ------------------------------------------------------------------ content
title = doc.add_heading("ReDeEP Reproduction & Robustness Study — Final Report", 0)
para("Adithya K · October 2026", italic=True)

doc.add_heading("Abstract", 1)
para(f"We reproduced ReDeEP (Sun et al., ICLR 2025) on the RAGTruth benchmark using a single 6 GB laptop GPU. "
     f"On the paper's own backbone, Llama-2-7B-chat, our detector reaches an AUC of {f3(l2['cv']['AUC'][0])}, against 0.7325 "
     f"reported in the paper. We repeated the experiment on three more LLaMA-family models (TinyLlama-1.1B, Llama-3.2-1B, "
     f"Llama-3.2-3B). ReDeEP beat the perplexity and LN-entropy baselines on all four, by 0.06 to 0.11 AUC.")
para("Our extension tests ReDeEP under corrupted retrieval. Swapping in an irrelevant passage raises the hallucination score "
     "sharply, as it should. Changing the numbers in the passage, so that the response now contradicts it, leaves the score "
     "unchanged on TinyLlama and slightly lowers it on Llama-2-7B. ReDeEP therefore measures whether a response is about its "
     f"context, not whether it agrees with it. Finally, prompted black-box judges from the Gemini and GPT families (AUC "
     f"{f3(JG[SPLIT]['AUC'])} and {f3(JO[SPLIT]['AUC'])}) outperform ReDeEP ({f3(l2['single_split']['AUC'])}) on the same "
     f"held-out responses, while a small 3B LLaMA judge is at chance ({f3(JL[SPLIT]['AUC'])}). ReDeEP's advantage is that it "
     "needs only one forward pass of a small open model, with no API.")

doc.add_heading("1. Introduction", 1)
para("Retrieval-augmented generation (RAG) gives a language model a retrieved passage to answer from, but models still produce "
     "claims the passage does not support. These RAG hallucinations are hard to catch with output-only signals such as "
     "perplexity, because a fluent, confident answer can still ignore its context.")
para("ReDeEP explains RAG hallucinations mechanistically. Copying Heads, attention heads that move context tokens into the "
     "residual stream, carry external knowledge. Knowledge FFNs, feed-forward layers, inject the model's parametric "
     "(memorised) knowledge. A hallucination happens when the model under-uses the first and over-relies on the second. "
     "ReDeEP turns both into scores and regresses the hallucination label on them.")
para("This project had three goals:")
bullets(["Reproduce ReDeEP's token-level detector on RAGTruth on consumer hardware (an RTX 4050 laptop GPU with 6 GB of VRAM).",
         "Test how the method transfers across the LLaMA family, from 1.1B to 7B parameters.",
         "Extend it: test whether its signals survive corrupted retrieval, and compare it against black-box LLM judges "
         "(GPT, Gemini, LLaMA) that cannot see model internals."], "List Number")

doc.add_heading("2. Method", 1)
para("The detector combines two internal signals, one per attention head and one per FFN layer, computed in a single forward "
     "pass over the prompt plus the response.")
para("For head h in layer l and each response token n, take the 10% of context tokens the head attends to most. Average their "
     "final-layer hidden states, then measure the cosine similarity with the response token's own final hidden state. High "
     "ECS means the response is built from what the head read in the context.", "External Context Score (ECS).")
formula("E^{l,h} = (1/|R|) Σ_{n∈R} cos( mean of x^L over top-k attended context tokens of n ,  x^L_n )")
para("For layer l, project the residual stream just before and just after the FFN block to the vocabulary (LogitLens). Then "
     "take the Jensen-Shannon divergence between the two next-token distributions. A large shift means the FFN injected its "
     "own knowledge.", "Parametric Knowledge Score (PKS).")
formula("P^l = (1/|R|) Σ_{n∈R} JSD( softmax(W_U norm(x^{mid,l}_n)) ,  softmax(W_U norm(x^l_n)) )")
bullets([("Copying-head selection.", "Each head's OV circuit is scored through 500 sampled vocabulary tokens; heads with large "
          "trace and few Gershgorin-disc outliers rank highest. The top 8 heads form set A."),
         ("Knowledge-FFN selection.", "On each training fold, the 5 layers whose PKS correlates most positively with the "
          "hallucination label form set F."),
         ("Detector.", "A logistic regression on two standardised features: mean PKS over F and negated mean ECS over A. The "
          "threshold is the best-F1 point on the training fold. AUC and PCC are threshold-free."),
         ("Baselines.", "Perplexity of the response, and LN-Entropy (mean per-token predictive entropy), from the same forward pass."),
         ("Black-box judges.", "Each judge sees the RAG input and the response and replies in JSON with the probability (0 to "
          "100) that the response contains a claim the passage contradicts or does not support, judging against the passage only.")])

doc.add_heading("3. Experimental setup", 1)
para(f"All models score the same {l2['n']} responses: the RAGTruth test split, restricted to responses generated by "
     f"llama-2-7b-chat (the paper's setting). Each example is capped at 6,000 characters of prompt plus response; "
     f"{l2['n_hallucinated']} of the {l2['n']} responses are hallucinated.")
table(["Backbone", "Parameters", "Precision", "Hugging Face checkpoint"],
      [["TinyLlama-1.1B-Chat", "1.1B", "fp16", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
       ["Llama-3.2-1B-Instruct", "1.2B", "fp16", "unsloth/Llama-3.2-1B-Instruct"],
       ["Llama-3.2-3B-Instruct", "3.2B", "4-bit NF4", "unsloth/Llama-3.2-3B-Instruct"],
       ["Llama-2-7B-chat", "6.7B", "4-bit NF4", "NousResearch/Llama-2-7b-chat-hf"]])
bullets([("Hardware:", "Lenovo LOQ laptop, NVIDIA RTX 4050 (6 GB VRAM), 16 GB RAM, Windows 11, PyTorch 2.11 (CUDA 12.8), "
          "transformers 5.17, bitsandbytes 0.50."),
         ("Sequence length:", "up to 1,024 tokens; long prompts are truncated from the left so the whole response is scored."),
         ("Evaluation:", "stratified 5-fold cross-validation, mean and standard deviation across folds. Copying heads and "
          "Knowledge FFNs are re-selected inside each training fold, so no test labels leak into selection."),
         ("Checkpoints:", "public mirrors with identical weights, so no gated licence or access token is needed.")])
para("The original code requested every full attention matrix, which does not fit for 7B models. We replaced this with "
     "forward hooks that keep only each layer's top-k context positions, and compute the mean of attended states as a sparse "
     "matrix product. On a RAGTruth example the rewritten ECS matched the original to within 6 × 10⁻⁸. "
     "Scoring took about 0.6 s per example for TinyLlama and 4 to 7 s for Llama-2-7B.", "Engineering for a 6 GB GPU.")

doc.add_heading("4. Results: hallucination detection", 1)
para(f"ReDeEP reproduces on the paper's backbone and beats both baselines on every LLaMA model we tried. Llama-2-7B reaches "
     f"{f3(l2['cv']['AUC'][0])} AUC against the paper's 0.7325, well inside one fold's standard deviation "
     f"({f3(l2['cv']['AUC'][1])}).")
chart_detection(HERE / "_fig_detection.png")
doc.add_picture(str(HERE / "_fig_detection.png"), width=Inches(6.3))
para("Source: data/results_*.json from run_pipeline.py; 5-fold stratified CV; whiskers = ±1 std across folds.", italic=True)
table(["Backbone", "ReDeEP AUC", "ReDeEP PCC", "ReDeEP F1", "Perplexity AUC", "LN-Entropy AUC"],
      [[n, f"{f3(R[n]['cv']['AUC'][0])} ± {f3(R[n]['cv']['AUC'][1])}", f3(R[n]["cv"]["PCC"][0]), f3(R[n]["cv"]["F1"][0]),
        f3(R[n]["cv"]["PPL_AUC"][0]), f3(R[n]["cv"]["LNEnt_AUC"][0])] for n, _ in BACKBONES]
      + [["Paper: Llama-2-7B-chat", "0.7325", "0.3979", "", "", ""]], bold_col=1)
para("The margin over the better baseline ranges from 0.06 (Llama-3.2-3B, against LN-Entropy) to 0.10 (Llama-3.2-1B, against "
     "perplexity). Detection quality is roughly flat in model size: the 1.1B TinyLlama scores highest, and the 3B and 7B "
     "models are within one standard deviation of each other. The baselines vary more across folds (std 0.06 to 0.08) than "
     "ReDeEP does (0.03 to 0.05).")

doc.add_heading("5. Black-box LLM judges", 1)
para("Large prompted judges detect hallucinations better than ReDeEP on our held-out split, but a small prompted judge is no "
     "better than chance. All detectors below are scored on the same 124 held-out test responses.")
s2 = l2["single_split"]
table(["Detector", "Kind", "AUC", "PCC", "Recall", "F1"],
      [["Gemini 3.5 Flash-Lite", "Prompted judge (API)", f3(JG[SPLIT]["AUC"]), f3(JG[SPLIT]["PCC"]), f3(JG[SPLIT]["Rec"]), f3(JG[SPLIT]["F1"])],
       ["GPT: gpt-oss-120b (OpenAI open weights, via Groq)", "Prompted judge (API)", f3(JO[SPLIT]["AUC"]), f3(JO[SPLIT]["PCC"]), f3(JO[SPLIT]["Rec"]), f3(JO[SPLIT]["F1"])],
       ["ReDeEP on Llama-2-7B (4-bit)", "Model internals", f3(s2["AUC"]), f3(s2["PCC"]), f3(s2["Rec"]), f3(s2["F1"])],
       ["LN-Entropy, Llama-2-7B", "Output uncertainty", f3(s2["LNEnt_AUC"]), "", "", ""],
       ["Perplexity, Llama-2-7B", "Output uncertainty", f3(s2["PPL_AUC"]), "", "", ""],
       ["Llama-3.2-3B-Instruct, local", "Prompted judge", f3(JL[SPLIT]["AUC"]), f3(JL[SPLIT]["PCC"]), f3(JL[SPLIT]["Rec"]), f3(JL[SPLIT]["F1"])]],
      bold_col=2)
bullets([("Large judges win on accuracy.", "Both API judges beat ReDeEP by 0.07 to 0.11 AUC. They read the passage and answer "
          "as text, so they can spot a contradicted number directly, which is exactly ReDeEP's blind spot (section 6)."),
         ("Small judges fail.", "The 3B LLaMA judge answers “no hallucination” almost every time, catching 7% of "
          "hallucinations. Reading the internals of a model of the same size (ReDeEP on Llama-3.2-3B, AUC 0.727) works far better."),
         ("ReDeEP wins on cost and access.", "It needs one forward pass of a 7B open model on a laptop GPU: no API, no per-call "
          "cost, no rate limits, and no data leaving the machine.")])
para(f"On all {JG['n']} responses, the Gemini judge's AUC is {f3(JG['all']['AUC'])}, consistent with the 124-response result. "
     "The judges were run once at default sampling settings; a difference of about 0.05 AUC is within the noise of 124 responses.")

doc.add_heading("6. Extension: robustness under corrupted retrieval", 1)
para("ReDeEP flags responses whose context is off-topic, but it does not notice when the context contradicts the response. We "
     "kept each response fixed and changed only the retrieved passage:")
bullets([("Accurate:", "the real RAGTruth passage (control)."),
         ("Irrelevant passage:", "a passage from a different source of the same task type."),
         ("Numbers corrupted:", "every number in the real passage multiplied by a random factor between 1.3 and 3.0, so "
          "numbers the response copied are now contradicted.")], "List Number")
para("The detector was refit on the other 265 examples, so all 150 perturbed responses are held out. We tested TinyLlama-1.1B "
     "and Llama-2-7B.")
chart_robustness(HERE / "_fig_robustness.png")
doc.add_picture(str(HERE / "_fig_robustness.png"), width=Inches(6.3))
para("Source: data/robustness_*.json from run_robustness.py; paired Wilcoxon signed-rank test.", italic=True)
rows = []
for name, c in (("TinyLlama-1.1B", robT), ("Llama-2-7B", rob7)):
    for cond, lab in (("accurate", "Accurate"), ("irrelevant", "Irrelevant"), ("contradictory", "Numbers corrupted")):
        m = c[cond]
        rows.append([name, lab, f3(m["score_mean"]), f3(m["E_A_mean"]), f3(m["P_F_mean"]), f3(m["AUC_vs_original_labels"])])
table(["Backbone", "Condition", "Mean score", "ECS (copying heads)", "PKS (knowledge FFNs)", "AUC vs. original labels"], rows)
para("The score jumps for nearly every response, so the original labels stop separating them (AUC falls to about 0.55). That "
     "is correct, because nothing in the response is supported by the new passage. In TinyLlama, ECS falls; in Llama-2-7B, ECS "
     "is flat and the jump comes from PKS.", "Irrelevant passages work as intended.")
para(f"For TinyLlama the score does not move ({robT['contradictory']['score_shift_vs_accurate']:+.3f}, p = "
     f"{robT['contradictory']['score_shift_wilcoxon_p']:.2f}). For Llama-2-7B it falls slightly "
     f"({rob7['contradictory']['score_shift_vs_accurate']:+.3f}, p < 0.001) and ECS rises: the detector rates a contradicted "
     f"response as more faithful. On the {num7['n']} responses that state a number the corrupted passage now contradicts, the "
     f"score rose in only {round(100 * numT['frac_score_increased'])}% of cases for TinyLlama and "
     f"{round(100 * num7['frac_score_increased'])}% for Llama-2-7B.", "Corrupted numbers go unnoticed.")
para("ECS is a cosine similarity between hidden states. It measures whether the response is about what the copying heads read, "
     "not whether the facts match. A plausible explanation is that a changed number barely moves the hidden state, so ReDeEP "
     "cannot see it. The method catches topical hallucinations but needs a complementary fact-level check for numeric or "
     "entity contradictions.", "Interpretation.")

doc.add_heading("7. Discussion and limitations", 1)
bullets([("4-bit quantisation.", "Llama-3.2-3B and Llama-2-7B ran in 4-bit NF4 to fit 6 GB, so their ECS and PKS approximate the "
          "full-precision values. That Llama-2-7B still matches the paper suggests the effect is small."),
         ("Sampled copying-head scores.", "Each head's OV circuit is scored on 500 sampled vocabulary tokens instead of the "
          "full vocabulary, so the head ranking may differ slightly."),
         ("One response generator.", "All 415 responses come from llama-2-7b-chat, so the other three backbones score text "
          "they did not write, unlike the paper's matched setting."),
         ("Sample size.", "415 responses give fold-level standard deviations of 0.03 to 0.05 AUC; differences below about 0.05 "
          "are within noise."),
         ("Robustness design.", "Only numbers are corrupted; entity swaps and negations may behave differently."),
         ("Judge access.", "The judges ran on free API tiers. The GPT-family model is OpenAI's open-weights gpt-oss-120b served "
          "by Groq, evaluated on the 124 held-out responses because of rate limits; Gemini is 3.5 Flash-Lite because the newest "
          "Flash model allowed 20 free requests per day.")])

doc.add_heading("8. Conclusion and future work", 1)
para(f"ReDeEP reproduces on a laptop: AUC {f3(l2['cv']['AUC'][0])} on Llama-2-7B against 0.7325 in the paper. It beats "
     "perplexity and entropy on four LLaMA models from 1.1B to 7B parameters. Its signals catch responses that are off-topic "
     "for their context, but not responses that contradict their context's numbers. Large prompted judges are more accurate "
     "but need an API; small prompted judges do not work at all.")
para("The practical reading: ReDeEP is a cheap, private first-line detector for topical hallucinations, and needs a fact-level "
     "check alongside it for numeric or entity conflicts.")
bullets(["Extend the robustness test to entity swaps and negations.",
         "Combine ECS with a numeric-consistency feature and measure the gain on the contradicted subset.",
         "Rerun Llama-2-7B and Llama-2-13B at full precision on a larger GPU, scoring responses generated by each backbone.",
         "Run the GPT judge on all 415 responses and stronger judge tiers, with repeated sampling to measure variance."], "List Number")

doc.add_heading("References", 1)
bullets(["Sun, Z. et al. ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability. ICLR 2025.",
         "Niu, C. et al. RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. ACL 2024.",
         "Touvron, H. et al. Llama 2: Open Foundation and Fine-Tuned Chat Models. 2023.",
         "Malinin, A. and Gales, M. Uncertainty Estimation in Autoregressive Structured Prediction. ICLR 2021.",
         "nostalgebraist. Interpreting GPT: the logit lens. LessWrong, 2020."], "List Number")

doc.save(OUT)
for f in ("_fig_detection.png", "_fig_robustness.png"):
    (HERE / f).unlink()
print("wrote", OUT)
