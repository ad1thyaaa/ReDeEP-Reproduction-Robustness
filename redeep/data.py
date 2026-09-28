"""Load and prepare RAGTruth examples for ReDeEP scoring.

Merges response.jsonl (per-response hallucination annotations) with
source_info.jsonl (prompt/context) on source_id, exactly as the original
RAGTruth baseline does, and produces a flat list of examples with a
binary response-level hallucination label.
"""
import json
import random
from pathlib import Path

RAGTRUTH_DIR = Path(__file__).resolve().parents[2] / "RAGTruth" / "dataset"


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_examples(ragtruth_dir=RAGTRUTH_DIR, split=None, task_types=None,
                   quality_good_only=True, models=None):
    """Return a list of dicts: id, source_id, task_type, model, prompt,
    response, label (1 = hallucination present), split.
    """
    responses = _read_jsonl(Path(ragtruth_dir) / "response.jsonl")
    sources = {s["source_id"]: s for s in _read_jsonl(Path(ragtruth_dir) / "source_info.jsonl")}

    examples = []
    for r in responses:
        if split is not None and r.get("split") != split:
            continue
        if quality_good_only and r.get("quality") != "good":
            continue
        if models is not None and r.get("model") not in models:
            continue
        src = sources.get(r["source_id"])
        if src is None:
            continue
        if task_types is not None and src.get("task_type") not in task_types:
            continue
        label = 1 if len(r.get("labels", [])) > 0 else 0
        examples.append({
            "id": r["id"],
            "source_id": r["source_id"],
            "task_type": src["task_type"],
            "model": r.get("model"),
            "prompt": src["prompt"],
            "response": r["response"],
            "labels": r.get("labels", []),
            "label": label,
            "split": r.get("split"),
        })
    return examples


def subsample_balanced(examples, n_per_class, max_prompt_chars=6000, seed=2024):
    """Pick a small, roughly class-balanced, length-capped subset so that
    CPU-only forward passes stay tractable."""
    rng = random.Random(seed)
    filtered = [e for e in examples if len(e["prompt"]) + len(e["response"]) <= max_prompt_chars]
    pos = [e for e in filtered if e["label"] == 1]
    neg = [e for e in filtered if e["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    picked = pos[:n_per_class] + neg[:n_per_class]
    rng.shuffle(picked)
    return picked


if __name__ == "__main__":
    ex = load_examples(split="train")
    print("train examples (good quality):", len(ex))
    print("label balance:", sum(e["label"] for e in ex), "/", len(ex))
    small = subsample_balanced(ex, 5)
    for e in small:
        print(e["task_type"], e["model"], e["label"], len(e["prompt"]), len(e["response"]))
