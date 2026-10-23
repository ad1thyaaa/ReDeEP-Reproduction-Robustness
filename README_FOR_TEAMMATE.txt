ReDeEP project — final package
Gourilakshmi S (23BCS21) & Adithya K (23BCS51)

START HERE
  ReDeEP_Final_Report_IEEE.pdf   final report, IEEE format (10 pages)
  ReDeEP_presentation.pptx       slides with speaker notes (15 slides)

WHAT WE DID
  1. Reproduced ReDeEP (Sun et al., ICLR 2025) on RAGTruth on a 6 GB laptop GPU.
     Llama-2-7B: AUC 0.739 vs 0.7325 in the paper.
  2. Six LLaMA-family models (1.1B to 13B). On RAGTruth ReDeEP beats perplexity
     and LN-Entropy on all six; strongest on Llama-2-13B (AUC 0.801).
     Llama-2-13B and Llama-3-8B ran on a free Kaggle GPU.
  3. Replicated the paper's two mechanistic findings (copying heads use less
     context in hallucinations; later-layer FFNs inject more knowledge).
  4. With the authors' released code and data:
     - chunk-level ReDeEP reproduced exactly (0.731, same as the authors' own scores)
     - Dolly (AC) evaluated (token 0.690, chunk 0.713 on Llama-2-7B)
     - audit: the official code selects heads/layers on the test set; this inflates
       AUC by 0.016 on RAGTruth and about 0.08 on Dolly (AC) for all three models
     - AARF mitigation: the gate is always on (Llama-2) or almost never on
       (Llama-3-8B); its win over the baseline mostly disappears once the
       judge's position bias is controlled (Llama-2-7B: 33:24 -> 19:15)
  5. Our extension (robustness): ReDeEP flags an irrelevant passage (+0.25 to +0.32)
     but misses a passage whose numbers contradict the answer (about 0 change).
  6. Black-box judges on the same 124 responses: Gemini 0.827, GPT (gpt-oss-120b)
     0.790, ReDeEP 0.721, small Llama-3.2-3B judge 0.499 (chance).

FOLDERS
  redeep/redeep/      core code (mechanistic scores, detector, chunk level, AARF, judges)
  redeep/scripts/     experiment scripts (run_*.py, eval_chunk.py, leakage_check.py)
  redeep/paper/       report source (build_paper.py builds the PDF from the results)
  redeep/presentation/build_deck.js   slide builder
  redeep/kaggle/      the Kaggle GPU job used for Llama-2-13B / Llama-3-8B
  redeep/data/        all results (results_*.json, eval_chunk_*.json, aarf_*, logs)

DATA NOT INCLUDED (download if you want to re-run)
  RAGTruth:          https://github.com/ParticleMedia/RAGTruth
  Official ReDeEP code + dataset (Dolly(AC), Llama-3 responses, chunk spans):
                     https://github.com/Jeryi-Sun/ReDEeP-ICLR
  Put them next to the redeep/ folder as RAGTruth/ and ReDeEP_official_dataset/.
