"""Fallback path: run the ReDeEP pipeline on a rented GPU via modal.com,
for whenever the local RTX 4050 (6GB VRAM) isn't enough -- e.g. Llama-2-13B,
or scoring thousands of examples for a tighter AUC estimate.

Setup (one-time, on whichever machine you run this from):
    pip install modal
    modal setup                      # opens a browser to link your modal.com account
    modal secret create huggingface HF_TOKEN=hf_...   # only needed for gated models

Usage:
    modal run scripts/modal_run.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --n-per-class 200 --max-len 1024 --tag modal_tinyllama

    modal run scripts/modal_run.py --model meta-llama/Llama-2-7b-chat-hf \
        --n-per-class 100 --max-len 1024 --gpu A10G --tag modal_llama2_7b

Results are written into the container's /root/redeep/data and copied back
to the local ./data directory via modal's volume mount, so they land in the
same place `scripts/run_pipeline.py` expects for the detection step.
"""
import modal

app = modal.App("redeep-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "accelerate", "scikit-learn", "scipy",
        "pandas", "sentencepiece", "protobuf", "bitsandbytes",
    )
)

volume = modal.Volume.from_name("redeep-data", create_if_missing=True)

# The RAGTruth dataset + redeep/ package are mounted read-only from the
# local checkout so the container sees exactly the same code as your local
# runs (no code duplication / drift).
repo_mount = modal.Mount.from_local_dir(
    local_path="..",  # the `redeep/` project root, one level up from scripts/
    remote_path="/root/redeep",
)


GPU_TYPE = "A10G"  # change this and re-run to switch card (e.g. "T4", "A100")


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    volumes={"/root/redeep/data": volume},
    mounts=[repo_mount],
    # Needed only for gated models (e.g. meta-llama/Llama-2-*): run
    # `modal secret create huggingface HF_TOKEN=hf_...` once, first, or
    # comment this line out for ungated models (TinyLlama, etc).
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(model: str, n_per_class: int, max_len: int, tag: str, max_prompt_chars: int = 6000):
    import subprocess
    subprocess.run(
        [
            "python", "/root/redeep/scripts/run_pipeline.py",
            "--model", model,
            "--n_per_class", str(n_per_class),
            "--max_len", str(max_len),
            "--max_prompt_chars", str(max_prompt_chars),
            "--tag", tag,
            "--device", "cuda",
            "--dtype", "float16",
        ],
        cwd="/root/redeep",
        check=True,
    )
    volume.commit()


@app.local_entrypoint()
def main(model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
         n_per_class: int = 100,
         max_len: int = 1024,
         max_prompt_chars: int = 6000,
         tag: str = "modal_run"):
    run.remote(model=model, n_per_class=n_per_class, max_len=max_len,
                tag=tag, max_prompt_chars=max_prompt_chars)
    print(f"Done. Pull results with: modal volume get redeep-data /{tag}* ./data/")
