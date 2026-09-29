"""Collect every results_*.json / robustness_*.json in data/ into one
markdown report (data/RESULTS.md). Re-run after any new run.

    python scripts/summarize.py
"""
import json
from pathlib import Path

data_dir = Path(__file__).resolve().parents[1] / "data"
lines = ["# Results", ""]

# --- ReDeEP across the LLaMA family -----------------------------------
lines += ["## ReDeEP(token) vs. baselines — LLaMA family",
          "",
          "RAGTruth test split, llama-2-7b-chat responses. 5-fold stratified CV, mean ± std.",
          "",
          "| Backbone | n | ReDeEP AUC | ReDeEP PCC | ReDeEP F1 | Perplexity AUC | LN-Entropy AUC |",
          "|---|---|---|---|---|---|---|"]
pipe = []
for p in sorted(data_dir.glob("results_*.json")):
    r = json.loads(p.read_text())
    if "cv" in r and r.get("args", {}).get("split") == "test":
        pipe.append(r)
for r in pipe:
    cv = r["cv"]
    f = lambda k: f"{cv[k][0]:.3f} ± {cv[k][1]:.3f}"
    q = " (4-bit)" if r["args"].get("load_in_4bit") else ""
    lines.append(f"| {r['model']}{q} | {r['n']} | **{f('AUC')}** | {f('PCC')} | {f('F1')} | "
                 f"{f('PPL_AUC')} | {f('LNEnt_AUC')} |")
lines += ["| *Paper: Llama-2-7B-chat (Sun et al. 2025, Table 1)* | | *0.7325* | *0.3979* | | | |", ""]

# --- black-box judges -------------------------------------------------
judges = [json.loads(p.read_text()) for p in sorted(data_dir.glob("results_judge_*.json"))]
if judges:
    lines += ["## Black-box LLM judges (GPT / Gemini / LLaMA)", "",
              "Same examples, no training. Score = judge's stated hallucination probability.", "",
              "| Judge | n | AUC | PCC | Acc | Rec | F1 |", "|---|---|---|---|---|---|---|"]
    for j in judges:
        a = j["all"]
        lines.append(f"| {j['provider']}: {j['judge_model']} | {j['n']} | {a['AUC']:.3f} | {a['PCC']:.3f} | "
                     f"{a['Acc']:.3f} | {a['Rec']:.3f} | {a['F1']:.3f} |")
    lines.append("")

# --- robustness extension ---------------------------------------------
rob = [json.loads(p.read_text()) for p in sorted(data_dir.glob("robustness_*.json"))
       if not p.name.startswith("robustness_numeric")]
if rob:
    lines += ["## Extension: robustness under corrupted retrieval", "",
              "Same response, different context. Detector refit excluding the perturbed examples.", "",
              "| Backbone | Condition | Mean ReDeEP score | ECS (E_A) | PKS (P_F) | AUC vs. orig. labels | Score shift vs. accurate (Wilcoxon p) |",
              "|---|---|---|---|---|---|---|"]
    for r in rob:
        for c, m in r["conditions"].items():
            shift = (f"{m['score_shift_vs_accurate']:+.3f} (p={m['score_shift_wilcoxon_p']:.1e})"
                     if "score_shift_vs_accurate" in m else "—")
            lines.append(f"| {r['tag']} | {c} | {m['score_mean']:.3f} | {m['E_A_mean']:.3f} | "
                         f"{m['P_F_mean']:.3f} | {m.get('AUC_vs_original_labels', float('nan')):.3f} | {shift} |")
    lines.append("")
    for p in sorted(data_dir.glob("robustness_numeric_subset_*.json")):
        s = json.loads(p.read_text())["response_numbers_contradicted"]
        tag = p.stem.replace("robustness_numeric_subset_", "")
        lines.append(f"- **{tag}**, restricted to the {s['n']} examples whose response states a number the "
                     f"corrupted context now contradicts: score shift {s['score_shift']:+.3f} "
                     f"(p={s['score_shift_p']:.1e}), score increased in only {100*s['frac_score_increased']:.0f}% "
                     f"of cases, ECS shift {s['E_A_shift']:+.3f}.")
    lines.append("")

(data_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
