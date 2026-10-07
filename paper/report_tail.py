# Appended to build_paper.py's figure code: fills the full-length IEEE report.
import collections
import sys as _sys
import numpy as np
_sys.path.insert(0, str(HERE.parent))
from redeep.data import load_examples
from redeep.llm_judge import JUDGE_SYSTEM

A = json.loads((HERE / "analysis.json").read_text())
NAMES = {"TinyLlama-1.1B": "TinyLlama-1.1B", "Llama-3.2-1B": "Llama-3.2-1B",
         "Llama-3.2-3B (4-bit)": "Llama-3.2-3B", "Llama-2-7B (4-bit)": "Llama-2-7B"}

# ---------------------------------------------------------------- dataset statistics
allr = [json.loads(l) for l in open(HERE.parent.parent / "RAGTruth" / "dataset" / "response.jsonl", encoding="utf-8")]
srcs = [json.loads(l) for l in open(HERE.parent.parent / "RAGTruth" / "dataset" / "source_info.jsonl", encoding="utf-8")]
task_of = {s["source_id"]: s["task_type"] for s in srcs}
n_resp, n_src = len(allr), len(srcs)
ds = collections.Counter()
for r in allr:
    t = task_of.get(r["source_id"])
    ds[(t, "n")] += 1
    ds[(t, "h")] += int(len(r.get("labels", [])) > 0)
pool = {e["id"]: e for e in load_examples(split="test", models=["llama-2-7b-chat"])}
ids7 = [str(i) for i in np.load(DATA / "scores_llama2_7b_4bit.npz")["ids"]]
sub = collections.Counter()
for i in ids7:
    e = pool[i]
    sub[(e["task_type"], "n")] += 1
    sub[(e["task_type"], "h")] += e["label"]
TASKN = {"QA": "Question answering", "Summary": "Summarization", "Data2txt": "Data-to-text"}
TSRC = {"QA": "MS MARCO", "Summary": "CNN/DM, news", "Data2txt": "Yelp"}
ds_rows = "\n".join(
    f"{TASKN[t]} & {TSRC[t]} & {ds[(t, 'n')]:,} & {100 * ds[(t, 'h')] / ds[(t, 'n')]:.0f}\\% & {sub[(t, 'n')]} & "
    f"{100 * sub[(t, 'h')] / sub[(t, 'n')]:.0f}\\% \\\\" for t in ("QA", "Summary", "Data2txt"))

# ---------------------------------------------------------------- analysis tables
mech_rows = "\n".join(
    f"{NAMES[n]} & {A[NAMES[n]]['n_layers']}$\\times${A[NAMES[n]]['n_heads']} & "
    f"{A[NAMES[n]]['heads_ecs_truth_gt_halluc']}/{A[NAMES[n]]['heads_total']} & "
    f"{A[NAMES[n]]['pks_halluc_gt_truth_early_layers']}/{A[NAMES[n]]['n_layers'] // 2} & "
    f"{A[NAMES[n]]['pks_halluc_gt_truth_late_layers']}/{A[NAMES[n]]['n_layers'] - A[NAMES[n]]['n_layers'] // 2} \\\\"
    for n, _ in BACKBONES)
task_rows = []
for n, _ in BACKBONES:
    pt = A[NAMES[n]]["per_task"]
    cells = []
    for t in ("QA", "Summary", "Data2txt"):
        rd, best = pt[t]["redeep"], max(pt[t]["ppl"], pt[t]["lnent"])
        cells.append((f"\\textbf{{{rd:.3f}}}" if rd >= best else f"{rd:.3f}") + f" / {best:.3f}")
    task_rows.append(f"{NAMES[n]} & " + " & ".join(cells) + " \\\\")
task_rows = "\n".join(task_rows)
abl_rows = "\n".join(
    f"{NAMES[n]} & {A[NAMES[n]]['feature_ablation']['ecs']:.3f} & {A[NAMES[n]]['feature_ablation']['pks']:.3f} & "
    f"\\textbf{{{A[NAMES[n]]['feature_ablation']['both']:.3f}}} \\\\" for n, _ in BACKBONES)
cost_rows = "\n".join(
    f"{NAMES[n]} & {A['_cost'][NAMES[n]]['copying_head_seconds']:.1f} & {A['_cost'][NAMES[n]]['sec_per_example_median']:.1f} & "
    f"{A['_cost'][NAMES[n]]['minutes']:.1f} \\\\" for n, _ in BACKBONES)
g7 = np.array(A["Llama-2-7B"]["grid_auc"])
gv = A["_judge_score_values"]
m7 = A["Llama-2-7B"]
pt7 = m7["per_task"]
pt_rng = [v["redeep"] for n in A if not n.startswith("_") for v in A[n]["per_task"].values()]

# judge prompt for the appendix, wrapped for a two-column verbatim block
import textwrap
prompt_wrapped = "\n".join(textwrap.wrap(JUDGE_SYSTEM, 52))

V.update(dict(
    nResp=f"{n_resp:,}", nSrc=f"{n_src:,}", ds_rows=ds_rows, mech_rows=mech_rows, task_rows=task_rows,
    abl_rows=abl_rows, cost_rows=cost_rows,
    h7=m7["heads_ecs_truth_gt_halluc"], ht7=m7["heads_total"],
    lateP=m7["pks_halluc_gt_truth_late_layers"], earlyP=m7["pks_halluc_gt_truth_early_layers"], halfL=m7["n_layers"] // 2,
    qa7=f3(pt7["QA"]["redeep"]), su7=f3(pt7["Summary"]["redeep"]), d2t7=f3(pt7["Data2txt"]["redeep"]),
    hQA=f"{100 * sub[('QA', 'h')] / sub[('QA', 'n')]:.0f}", hSU=f"{100 * sub[('Summary', 'h')] / sub[('Summary', 'n')]:.0f}",
    hD2T=f"{100 * sub[('Data2txt', 'h')] / sub[('Data2txt', 'n')]:.0f}",
    ptmin=f"{min(pt_rng):.2f}", ptmax=f"{max(pt_rng):.2f}",
    gmin=f3(g7.min()), gmax=f3(g7.max()), gdef=f3(g7[3, 2]),
    fe7=f3(m7["feature_ablation"]["ecs"]), fp7=f3(m7["feature_ablation"]["pks"]), fb7=f3(m7["feature_ablation"]["both"]),
    gemBin=f"{100 * gv['Gemini 3.5 Flash-Lite']['frac_0_or_100']:.1f}", llBin=f"{100 * gv['Llama-3.2-3B']['frac_0_or_100']:.1f}",
    gptDistinct=gv["GPT gpt-oss-120b"]["distinct_values"],
    t7=f"{A['_cost']['Llama-2-7B']['minutes']:.0f}", tT=f"{A['_cost']['TinyLlama-1.1B']['minutes']:.1f}",
    prompt=prompt_wrapped,
))

TEX = (HERE / "report_template.tex").read_text(encoding="utf-8")
for k, v in V.items():
    TEX = TEX.replace(f"<<{k}>>", str(v))
assert "<<" not in TEX, TEX[TEX.index("<<") - 60: TEX.index("<<") + 60]
(HERE / "redeep_paper.tex").write_text(TEX, encoding="utf-8")
print("wrote", HERE / "redeep_paper.tex")
