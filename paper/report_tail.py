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

# ---------------------------------------------------------------- official code/data reproduction
from scipy.stats import binomtest
opt = lambda f: json.loads((DATA / f).read_text()) if (DATA / f).exists() else None
ev_off = opt("eval_chunk_official_llama2_7B_response_chunk.json")
BIG = [("Llama-2-7B", "llama2_7b"), ("Llama-2-13B", "llama2_13b"), ("Llama-3-8B", "llama3_8b")]
cell = lambda m, s=None: "--" if m is None else (f"{m:.3f}" + (f" ({s:.3f})" if s is not None else ""))
def tok_auc(tag, ds):
    if tag == "llama2_7b" and ds == "ragtruth":
        return R["Llama-2-7B (4-bit)"]["cv"]["AUC"][0]
    r = opt(f"results_tok_{tag}_{ds}.json")
    return r["cv"]["AUC"][0] if r else None
def ch_auc(tag, ds):
    e = opt(f"eval_chunk_{tag}_{ds}.json")
    return (e["nested_cv"]["AUC_mean"], e["official_protocol"]["AUC"]) if e else (None, None)
official_rows = []
for name, tag in BIG:
    cr, cd = ch_auc(tag, "ragtruth"), ch_auc(tag, "dolly")
    official_rows.append(f"{name} & {cell(tok_auc(tag, 'ragtruth'))} & {cell(*cr)} & {cell(tok_auc(tag, 'dolly'))} & {cell(*cd)} \\\\")
official_rows = "\n".join(official_rows)

e7r, e7d = opt("eval_chunk_llama2_7b_ragtruth.json"), opt("eval_chunk_llama2_7b_dolly.json")
audit = [("Authors' logs, RAGTruth", ev_off), ("Ours, RAGTruth", e7r), ("Ours, Dolly (AC)", e7d)]
for name, tag in BIG[1:]:
    for ds, dn in (("ragtruth", "RAGTruth"), ("dolly", "Dolly (AC)")):
        e = opt(f"eval_chunk_{tag}_{ds}.json")
        if e:
            audit.append((f"Ours {name}, {dn}", e))
audit_rows = "\n".join(f"{n} & {e['official_protocol']['AUC']:.3f} & {e['tuned_on_test']['AUC']:.3f} & "
                       f"\\textbf{{{e['nested_cv']['AUC_mean']:.3f}}} \\\\" for n, e in audit)
td7 = opt("results_tok_llama2_7b_dolly.json")

big_have = [(n, t) for n, t in BIG[1:] if opt(f"results_tok_{t}_ragtruth.json")]
if big_have:
    parts = []
    for n, t in big_have:
        r = opt(f"results_tok_{t}_ragtruth.json")
        cr = ch_auc(t, "ragtruth")[0]
        d = opt(f"results_tok_{t}_dolly.json")
        s = (f"{n} reaches a token-level AUC of {r['cv']['AUC'][0]:.3f} on its own {r['n']} RAGTruth responses "
             f"(perplexity {r['cv']['PPL_AUC'][0]:.3f}, LN-Entropy {r['cv']['LNEnt_AUC'][0]:.3f})")
        if cr is not None:
            s += f", {cr:.3f} at chunk level"
        if d:
            s += (f", and {d['cv']['AUC'][0]:.3f} on Dolly (AC) (LN-Entropy {d['cv']['LNEnt_AUC'][0]:.3f})")
        parts.append(s)
    big_para = ("\\textbf{Larger backbones.} Llama-2-13B-chat and Llama-3-8B-Instruct do not fit a 6\\,GB GPU, so we ran them in "
                "4-bit on a free Kaggle T4 GPU with the same code, each scoring its own responses. " + "; ".join(parts) + ". "
                "Llama-2-13B is the strongest detector we tested. On Llama-3-8B, ReDeEP's margin over LN-Entropy is small on "
                "RAGTruth and reversed on Dolly (AC), consistent with the paper's observation that heads and layers must be "
                "re-tuned per model; Llama-3-8B's released labels are response-level only, so its chunk-level score is "
                "evaluated per response.")
    big_contrib = "the paper's two larger backbones (Llama-2-13B, Llama-3-8B), "
    aarf_lim = ""
else:
    big_para = ("\\textbf{Larger backbones.} Llama-2-13B and Llama-3-8B do not fit a 6\\,GB GPU; their rows are left empty here.")
    big_contrib = ""
    aarf_lim = ""

# AARF
ar = opt("aarf_result_llama2_7b.json")
arows = [json.loads(l) for l in open(DATA / "aarf_llama2_7b.jsonl", encoding="utf-8")]
gates = np.concatenate([np.array(r["gate_trace"])[:, 2] for r in arows if r.get("gate_trace")])
jrows = [json.loads(l) for l in open(DATA / "aarf_judge_llama2_7b.jsonl", encoding="utf-8")]
byid = {}
for x in jrows:
    byid.setdefault(x["id"], {})[x["order"]] = x["aarf_won"]
both_parsed = [v for v in byid.values() if v.get("aarf_first") is not None and v.get("baseline_first") is not None]
flips = sum(v["aarf_first"] != v["baseline_first"] for v in both_parsed)
bA = sum(v["aarf_first"] and v["baseline_first"] for v in both_parsed)
bB = sum((not v["aarf_first"]) and (not v["baseline_first"]) for v in both_parsed)
first = [v.get("aarf_first") for v in byid.values()]


def aarf_stats(tag):
    gf, jf = DATA / f"aarf_{tag}.jsonl", DATA / f"aarf_judge_{tag}.jsonl"
    if not gf.exists():
        return None
    rows = [json.loads(l) for l in open(gf, encoding="utf-8")]
    g = np.concatenate([np.array(r["gate_trace"])[:, 2] for r in rows if r.get("gate_trace")])
    by = {}
    if jf.exists():
        for x in (json.loads(l) for l in open(jf, encoding="utf-8")):
            by.setdefault(x["id"], {})[x["order"]] = x["aarf_won"]
    bp = [v for v in by.values() if v.get("aarf_first") is not None and v.get("baseline_first") is not None]
    return {"n": len(rows), "changed": sum(r["baseline"] != r["aarf"] for r in rows), "fire": 100 * (g > 0.6).mean(),
            "offA": sum(v.get("aarf_first") is True for v in by.values()),
            "offB": sum(v.get("aarf_first") is False for v in by.values()),
            "bothA": sum(v["aarf_first"] and v["baseline_first"] for v in bp),
            "bothB": sum((not v["aarf_first"]) and (not v["baseline_first"]) for v in bp), "judged": bool(by)}


aarf_rows, other = [], []
for name, tag in BIG:
    s = aarf_stats(tag)
    if s is None:
        continue
    j = (f"{s['offA']} / {s['offB']} & {s['bothA']} / {s['bothB']}" if s["judged"] else "-- & --")
    aarf_rows.append(f"{name} & {s['changed']}/{s['n']} & {s['fire']:.0f}\\% & {j} \\\\")
    if tag != "llama2_7b":
        other.append((name, s))
aarf_rows = "\n".join(aarf_rows)
if other:
    bits = []
    for name, s in other:
        b = (f"{name}: the gate fired on {s['fire']:.0f}\\% of steps and changed {s['changed']} of {s['n']} answers")
        if s["judged"] and s["changed"]:
            b += f"; with both orders required to agree, AARF won {s['bothA']} and the baseline {s['bothB']}"
        bits.append(b)
    aarf_other_para = ("\\textbf{Other backbones.} With each model's released hyperparameters, run on a Kaggle T4 GPU: "
                       + "; ".join(bits) + ". The gate is thus either always on or almost never on, depending on the "
                       "model's stored ranges, and in neither case does AARF measurably improve truthfulness.")
    aarf_lim = ""
else:
    aarf_other_para = ""
V.update(dict(aarf_rows=aarf_rows, aarfOtherPara=aarf_other_para))
V.update(dict(
    chR7=f3(e7r["nested_cv"]["AUC_mean"]), chR7off=f3(e7r["official_protocol"]["AUC"]),
    chOff=f3(ev_off["nested_cv"]["AUC_mean"]), chOffoff=f3(ev_off["official_protocol"]["AUC"]),
    offPcc=f3(ev_off["official_protocol"]["PCC"]),
    leakR=f3(ev_off["official_protocol"]["AUC"] - ev_off["nested_cv"]["AUC_mean"]),
    leakD=f3(e7d["official_protocol"]["AUC"] - e7d["nested_cv"]["AUC_mean"]),
    tokD7=f3(td7["cv"]["AUC"][0]), tokD7sd=f3(td7["cv"]["AUC"][1]),
    pplD7=f3(td7["cv"]["PPL_AUC"][0]), lnD7=f3(td7["cv"]["LNEnt_AUC"][0]),
    official_rows=official_rows, audit_rows=audit_rows, bigModelsPara=big_para, bigModelsContrib=big_contrib,
    aarfN=len(arows), aarfSteps=f"{len(gates):,}", aarfMin=f"{gates.min():.2f}", aarfMed=f"{np.median(gates):.2f}",
    aarfFrac=f"{100 * (gates > 0.6).mean():.0f}",
    aarfOffA=sum(x is True for x in first), aarfOffB=sum(x is False for x in first), aarfUnp=sum(x is None for x in first),
    aarfBothA=bA, aarfBothB=bB, aarfDiff=len(byid), aarfSame=len(arows) - len(byid),
    aarfFlip=flips, aarfParsed=len(both_parsed), aarfFlipPct=f"{100 * flips / max(len(both_parsed), 1):.0f}",
    aarfP=f"{binomtest(bA, bA + bB, 0.5).pvalue:.2f}", aarfLimitModels=aarf_lim,
))

TEX = (HERE / "report_template.tex").read_text(encoding="utf-8")
for k, v in V.items():
    TEX = TEX.replace(f"<<{k}>>", str(v))
assert "<<" not in TEX, TEX[TEX.index("<<") - 60: TEX.index("<<") + 60]
(HERE / "redeep_paper.tex").write_text(TEX, encoding="utf-8")
print("wrote", HERE / "redeep_paper.tex")
