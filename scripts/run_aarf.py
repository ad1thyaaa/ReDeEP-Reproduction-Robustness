"""Generate Dolly(AC) answers with and without gated AARF (as in the
authors' AARF/truthful_answer_generate.py), resumable.

    python scripts/run_aarf.py --model NousResearch/Llama-2-7b-chat-hf --load_in_4bit \
        --hp official:test_llama2_7B --tag llama2_7b
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redeep.aarf_gen import GatedAARF
from redeep.chunk_level import chat_prefix

OFFICIAL = ROOT.parent / "ReDeEP_official_dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hp", default="official:test_llama2_7B", help="official:<log dir> -> token_hyperparameter_dolly.json")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--add", type=float, default=1.2)
    ap.add_argument("--reduce", type=float, default=0.8)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"dtype": torch.float16, "attn_implementation": "eager", "device_map": {"": "cuda"}}
    if args.load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    hp = json.load(open(OFFICIAL / "official_logs" / args.hp.split(":", 1)[1] / "token_hyperparameter_dolly.json"))
    gen = GatedAARF(model, hp, args.add, args.reduce, args.threshold)

    src = [json.loads(l) for l in open(OFFICIAL / "source_info_dolly.jsonl", encoding="utf-8")]
    test_ids = {json.loads(l)["source_id"] for l in open(OFFICIAL / "response_dolly.jsonl", encoding="utf-8")
                if json.loads(l)["split"] == "test"}
    src = sorted([s for s in src if s["source_id"] in test_ids], key=lambda s: s["source_id"])
    if args.limit:
        src = src[: args.limit]
    out_path = ROOT / "data" / f"aarf_{args.tag}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    eos = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<|eot_id|>")      # Llama-3 end-of-turn
    if isinstance(eot, int) and eot != tok.unk_token_id:
        eos.add(eot)
    print(f"{len(src)} Dolly(AC) prompts, {len(done)} done; hp={hp}")
    with open(out_path, "a", encoding="utf-8") as f:
        for i, s in enumerate(src):
            if s["source_id"] in done:
                continue
            t0 = time.time()
            ids = tok(chat_prefix(tok, s["prompt"][:8000]), return_tensors="pt").input_ids.cuda()
            try:
                base, _ = gen.generate(ids, args.max_new_tokens, aarf=False, eos_ids=eos)
                arf, n_int = gen.generate(ids, args.max_new_tokens, aarf=True, eos_ids=eos)
            except torch.cuda.OutOfMemoryError:
                gen._capture = False
                torch.cuda.empty_cache()
                print(f"  [{i+1}/{len(src)}] OOM ({ids.shape[1]} prompt tokens) -- skipped", flush=True)
                continue
            tr = gen.trace
            row = {"id": s["source_id"], "prompt": s["prompt"],
                   "baseline": tok.decode(base, skip_special_tokens=True).strip(),
                   "aarf": tok.decode(arf, skip_special_tokens=True).strip(),
                   "aarf_interventions": n_int, "aarf_tokens": len(arf),
                   "gate_trace": [[round(a, 4), round(b, 4), round(c, 4)] for a, b, c in tr]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i+1}/{len(src)}] {len(base)} vs {len(arf)} tokens, {n_int} AARF steps, "
                  f"identical={row['baseline'] == row['aarf']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
