"""Score ReDeEP(chunk) features for RAGTruth or Dolly(AC), following the
official implementation; resumable (appends to data/chunk_<tag>.jsonl).

    python scripts/run_chunk.py --model NousResearch/Llama-2-7b-chat-hf --load_in_4bit \
        --tag llama2_7b_ragtruth --dataset ragtruth --response_model llama-2-7b-chat \
        --copy_heads official:test_llama2_7B
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redeep.mechanistic import LlamaMechInterp
from redeep.chunk_level import ChunkScorer, BGE

OFFICIAL = ROOT.parent / "ReDeEP_official_dataset"


def load(dataset, response_model):
    rd = lambda f: [json.loads(l) for l in open(OFFICIAL / f, encoding="utf-8")]
    if dataset == "ragtruth":
        src = {s["source_id"]: s for s in rd("source_info_spans.jsonl")}
        resp_file = "response_with_llama3_8b_spans.jsonl" if "llama-3" in response_model else "response_spans.jsonl"
        rows = [r for r in rd(resp_file) if r["model"] == response_model and r["split"] == "test"]
        lab = lambda r: int(len(r["labels"]) > 0)
    else:
        src = {s["source_id"]: s for s in rd("source_info_dolly_spans.jsonl")}
        rows = [r for r in rd("response_dolly_spans.jsonl") if r["model"] == response_model and r["split"] == "test"]
        lab = lambda r: int(r["label"] == ["Yes"])
    return [dict(id=str(r["id"]), source_id=r["source_id"], task_type=src[r["source_id"]].get("task_type"),
                 prompt=src[r["source_id"]]["prompt"], prompt_spans=src[r["source_id"]]["prompt_spans"],
                 response=r["response"], response_spans=r["response_spans"],
                 labels=r["labels"] if dataset == "ragtruth" else [], label=lab(r)) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", choices=["ragtruth", "dolly"], default="ragtruth")
    ap.add_argument("--response_model", default="llama-2-7b-chat")
    ap.add_argument("--copy_heads", default="official:test_llama2_7B",
                    help="official:<log dir> (authors' top-32 heads) or ours:<run_pipeline tag>")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    kind, ref = args.copy_heads.split(":", 1)
    if kind == "official":
        heads = json.load(open(OFFICIAL / "official_logs" / ref / "topk_heads.json"))[:32]
    else:
        cs = np.load(ROOT / "data" / f"copying_heads_{ref}.npy")
        heads = [list(np.unravel_index(i, cs.shape)) for i in np.argsort(-cs.flatten())[:32]]
    heads = [[int(a), int(b)] for a, b in heads]

    ex = load(args.dataset, args.response_model)
    if args.limit:
        ex = ex[: args.limit]
    out_path = ROOT / "data" / f"chunk_{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
    print(f"{len(ex)} responses ({sum(e['label'] for e in ex)} hallucinated), {len(done)} already scored")

    mech = LlamaMechInterp(args.model, device=args.device, dtype=torch.float16, load_in_4bit=args.load_in_4bit)
    scorer = ChunkScorer(mech, heads, BGE(device=args.device), max_tokens=args.max_tokens)
    t0, n = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, e in enumerate(ex):
            if e["id"] in done:
                continue
            t = time.time()
            try:
                s = scorer.score(e["prompt"], e["prompt_spans"], e["response"], e["response_spans"], e["labels"])
            except torch.cuda.OutOfMemoryError:
                scorer._req = None
                torch.cuda.empty_cache()
                print(f"  [{i+1}/{len(ex)}] OOM -- skipped", flush=True)
                continue
            if s is None:
                print(f"  [{i+1}/{len(ex)}] no usable chunks -- skipped", flush=True)
                continue
            f.write(json.dumps({"id": e["id"], "label": e["label"], "task_type": e["task_type"],
                                "heads": heads, "E": s["E"].tolist(), "P": s["P"].tolist(),
                                "chunk_labels": s["chunk_labels"], "n_tokens": s["n_tokens"],
                                "truncated": s["truncated"]}) + "\n")
            f.flush()
            n += 1
            eta = (time.time() - t0) / n * (len(ex) - i - 1) / 60
            print(f"  [{i+1}/{len(ex)}] label={e['label']} chunks={len(s['chunk_labels'])} "
                  f"tokens={s['n_tokens']} {time.time()-t:.1f}s (eta {eta:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
