# Kaggle job: ReDeEP on Llama-2-13B-chat and Llama-3-8B-Instruct (the paper's
# other two backbones), one model per T4 GPU, in parallel. Each model runs
#   token-level ReDeEP on its own RAGTruth test responses,
#   chunk-level ReDeEP (official copying heads), token + chunk on Dolly(AC),
#   and AARF generation on Dolly(AC).
# Results land in /kaggle/working/out (downloaded back to the laptop).
import os
import shutil
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5", "accelerate",
                "bitsandbytes", "scikit-learn", "scipy"], check=True)

import glob
# locate the bundle wherever Kaggle mounted it (the folder holding redeep/scripts)
hits = glob.glob("/kaggle/input/**/redeep/scripts/run_chunk.py", recursive=True)
assert hits, f"bundle not found under /kaggle/input: {os.listdir('/kaggle/input')}"
SRC = os.path.dirname(os.path.dirname(os.path.dirname(hits[0])))
print("bundle at", SRC, os.listdir(SRC))
WORK = "/kaggle/working/Deep_Learning_for_LOQ"
if not os.path.exists(WORK):
    shutil.copytree(SRC, WORK)
R = f"{WORK}/redeep"
os.makedirs(f"{R}/data/logs", exist_ok=True)

JOBS = {
    "0": ("NousResearch/Llama-2-13b-chat-hf", "llama2_13b", "llama-2-13b-chat", "test_llama2_13B"),
    "1": ("NousResearch/Meta-Llama-3-8B-Instruct", "llama3_8b", "llama-3-8b-instruct", "test_llama3_8B"),
}


def chain(gpu, model, tag, resp, heads):
    py = sys.executable
    steps = [
        f"{py} scripts/run_token_official.py --model {model} --load_in_4bit --tag tok_{tag}_ragtruth --dataset ragtruth --response_model {resp}",
        f"{py} scripts/run_chunk.py --model {model} --load_in_4bit --tag {tag}_ragtruth --dataset ragtruth --response_model {resp} --copy_heads official:{heads}",
        f"{py} scripts/run_token_official.py --model {model} --load_in_4bit --tag tok_{tag}_dolly --dataset dolly --response_model {resp}",
        f"{py} scripts/run_chunk.py --model {model} --load_in_4bit --tag {tag}_dolly --dataset dolly --response_model {resp} --copy_heads official:{heads}",
        f"{py} scripts/run_aarf.py --model {model} --load_in_4bit --hp official:{heads} --tag {tag}",
    ]
    cmd = " ; ".join(f"echo '>>> {s}' ; {s}" for s in steps)
    log = open(f"{R}/data/logs/kaggle_{tag}.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
    return subprocess.Popen(["bash", "-c", cmd], cwd=R, stdout=log, stderr=subprocess.STDOUT, env=env)


import torch
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
if n_gpu >= 2:                      # one model per GPU, in parallel
    procs = [chain(g, *j) for g, j in JOBS.items()]
    for p in procs:
        p.wait()
else:                               # single GPU: one model after the other
    for j in JOBS.values():
        chain("0", *j).wait()

out = "/kaggle/working/out"
os.makedirs(out, exist_ok=True)
for f in os.listdir(f"{R}/data"):
    if any(f.startswith(p) for p in ("scores_tok_", "results_tok_", "copying_heads_tok_", "chunk_", "aarf_")):
        shutil.copy(f"{R}/data/{f}", out)
shutil.copytree(f"{R}/data/logs", f"{out}/logs", dirs_exist_ok=True)
shutil.rmtree(WORK, ignore_errors=True)   # keep only results in the output
print("done:", sorted(os.listdir(out)))
