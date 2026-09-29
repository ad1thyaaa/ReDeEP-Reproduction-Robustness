"""Run black-box LLM judges (GPT / Gemini / local LLaMA) on the same
RAGTruth examples the ReDeEP pipeline scored, and report metrics.

    # free options (see redeep/llm_judge.py for the key env vars)
    python scripts/run_judges.py --provider github --judge_model openai/gpt-4o-mini --only_test_split --min_interval 4.5
    python scripts/run_judges.py --provider groq   --judge_model openai/gpt-oss-120b --min_interval 2.5
    python scripts/run_judges.py --provider gemini --judge_model gemini-2.5-flash --min_interval 6.5
    python scripts/run_judges.py --provider llama  --judge_model unsloth/Llama-3.2-3B-Instruct

Example selection mirrors run_pipeline.py's defaults for the paper setting
(test split, llama-2-7b-chat responses), so numbers line up with
results_<tag>.json from the LLaMA-family runs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redeep.data import load_examples, subsample_balanced
from redeep.llm_judge import make_judge, run_judge


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s)
    preds = (s >= 0.5).astype(int)
    return {"AUC": roc_auc_score(y, s), "PCC": pearsonr(s, y)[0] if s.std() > 0 else float("nan"),
            "Acc": accuracy_score(y, preds), "Rec": recall_score(y, preds), "F1": f1_score(y, preds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["openai", "github", "groq", "gemini", "llama"])
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--response_models", default="llama-2-7b-chat")
    ap.add_argument("--n_per_class", type=int, default=0)
    ap.add_argument("--max_prompt_chars", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=0, help="only judge the first N examples (smoke test)")
    ap.add_argument("--min_interval", type=float, default=0.0,
                    help="seconds between API calls (stay under free-tier requests/minute)")
    ap.add_argument("--only_test_split", action="store_true",
                    help="judge only the --compare_tag held-out test ids (fits free daily quotas)")
    ap.add_argument("--compare_tag", default="llama2_7b_4bit",
                    help="results_<tag>.json whose test_ids give a like-for-like held-out subset")
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / "data"
    models = args.response_models.split(",") if args.response_models else None
    pool = load_examples(split=args.split, models=models)
    n_per_class = args.n_per_class if args.n_per_class > 0 else len(pool)
    examples = subsample_balanced(pool, n_per_class, max_prompt_chars=args.max_prompt_chars)
    cmp_path = data_dir / f"results_{args.compare_tag}.json"
    test_ids = set(json.loads(cmp_path.read_text()).get("test_ids") or []) if cmp_path.exists() else set()
    if args.only_test_split:
        if not test_ids:
            sys.exit(f"--only_test_split needs {cmp_path} (run the LLaMA pipeline first)")
        examples = [e for e in examples if e["id"] in test_ids]
    if args.limit:
        examples = examples[: args.limit]
    print(f"{len(examples)} examples ({sum(e['label'] for e in examples)} hallucinated)")

    tag = f"judge_{args.provider}_{args.judge_model.replace('/', '_')}"
    judge = make_judge(args.provider, args.judge_model)
    rows = run_judge(judge, examples, data_dir / f"{tag}.jsonl", min_interval=args.min_interval)

    ok = [r for r in rows if r["score"] is not None]
    if len({r["label"] for r in ok}) < 2:
        sys.exit(f"only {len(ok)} usable replies so far -- re-run to resume")
    print(f"\nparsed {len(ok)}/{len(rows)} judge replies")
    res = {"provider": args.provider, "judge_model": args.judge_model, "n": len(ok),
           "all": metrics([r["label"] for r in ok], [r["score"] for r in ok])}

    if test_ids:
        sub = [r for r in ok if r["id"] in test_ids]
        if len(sub) > 2 and len({r["label"] for r in sub}) > 1:
            res["same_test_split_as_" + args.compare_tag] = metrics(
                [r["label"] for r in sub], [r["score"] for r in sub])
            res["n_same_test_split"] = len(sub)

    for name, m in res.items():
        if isinstance(m, dict):
            print(f"=== {name} ===")
            for k, v in m.items():
                print(f"  {k}: {v:.4f}")
    (data_dir / f"results_{tag}.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
