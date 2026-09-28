# ReDeEP reproduction + extension

Reproduces the core method of "ReDeEP: Detecting Hallucination in
Retrieval-Augmented Generation via Mechanistic Interpretability"
(Sun et al., ICLR 2025) on the RAGTruth benchmark, scoped to run on
CPU-only hardware, plus a robustness-testing extension.

## Layout

- `redeep/mechanistic.py` — Copying Head Score, External Context Score
  (ECS), Parametric Knowledge Score (PKS), computed via forward hooks on
  any LLaMA-architecture HF model (TinyLlama, Llama-2, Llama-3, ...).
- `redeep/detect.py` — ReDeEP(token) detector: selects Copying Heads /
  Knowledge FFN layers, fits a logistic regression over the decoupled
  covariates, reports AUC/PCC/Acc/Rec/F1.
- `redeep/aarf.py` — AARF intervention (amplify Copying Heads, damp
  Knowledge FFNs at decode time) for hallucination mitigation.
- `redeep/robustness.py` — **extension**: feeds the model deliberately
  wrong/irrelevant retrieved context and checks whether ECS/PKS still
  behave sensibly (tests whether the signals track "faithfulness to
  context" vs. secretly tracking "matches world knowledge").
- `redeep/data.py` — loads/merges RAGTruth's `response.jsonl` +
  `source_info.jsonl` into (prompt, response, label) triples.
- `scripts/run_pipeline.py` — end-to-end: load data → score with a
  backbone model → fit detector → report metrics.

## Day-1 results (CPU, TinyLlama-1.1B, 50 examples)

| Method | AUC | PCC | Acc | F1 |
|---|---|---|---|---|
| ReDeEP(token) | 0.870 | 0.758 | 0.867 | 0.833 |
| Perplexity baseline | 0.926 | — | — | — |
| LN-Entropy baseline | 0.704 | — | — | — |

Both signals clearly separate hallucinated/truthful responses, confirming
the mechanistic pipeline is producing real signal. Perplexity edging out
ReDeEP here (opposite of the paper's Table 1) is most likely noise from a
15-example test split -- needs a bigger run to confirm either way.

Pipeline is now GPU-ready: `--device cuda --dtype float16` (auto-detected
if a CUDA GPU is present). `scripts/modal_run.py` is an **untested**
scaffold for running on a rented GPU via modal.com if the local GPU isn't
enough (e.g. Llama-2-13B) -- verify against your installed `modal` CLI
version before relying on it.

## Status / plan

1. **Backbone for fast iteration**: TinyLlama-1.1B-Chat (LLaMA
   architecture — RMSNorm/RoPE/SwiGLU — so the paper's formulas transfer
   exactly; small enough for CPU).
2. **Paper-faithful reproduction**: swap in `meta-llama/Llama-2-7b-chat-hf`
   / `Llama-2-13b-chat-hf` once HF gated access is set up (needs the
   user to accept Meta's license on huggingface.co and run
   `hf auth login`). Same code, only `--model` changes. Expect much
   longer runtimes on CPU — plan to run on a small, length-capped subset.
3. **Black-box baselines (GPT-4o / Gemini)**: prompt-based detectors
   (Prompt / ChainPoll / RAGAS-style) implemented as API calls, gated on
   `OPENAI_API_KEY` / `GEMINI_API_KEY` being set — used only as
   external-judge comparisons, not for mechanistic scoring (closed
   models don't expose internals).
4. **Extension**: robustness sweep (`redeep/robustness.py`) — accurate
   vs. irrelevant vs. numerically-corrupted retrieved context, same
   response, tracking how ECS/PKS shift. This is the novel contribution
   for the paper/presentation beyond reproducing ReDeEP itself.

## Running

```
cd redeep
../deepLenv/bin/python scripts/run_pipeline.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --n_per_class 30 --max_len 768
```

Results cache to `data/scores_<tag>.npz` so repeated runs (e.g. tuning
`--n_copying_heads`) don't require rescoring.
