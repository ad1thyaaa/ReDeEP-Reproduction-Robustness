# Results

## ReDeEP(token) vs. baselines — LLaMA family

RAGTruth test split, llama-2-7b-chat responses. 5-fold stratified CV, mean ± std.

| Backbone | n | ReDeEP AUC | ReDeEP PCC | ReDeEP F1 | Perplexity AUC | LN-Entropy AUC |
|---|---|---|---|---|---|---|
| NousResearch/Llama-2-7b-chat-hf (4-bit) | 415 | **0.739 ± 0.034** | 0.427 ± 0.060 | 0.687 ± 0.015 | 0.627 ± 0.066 | 0.670 ± 0.063 |
| unsloth/Llama-3.2-1B-Instruct | 415 | **0.744 ± 0.045** | 0.426 ± 0.079 | 0.692 ± 0.040 | 0.649 ± 0.076 | 0.646 ± 0.061 |
| unsloth/Llama-3.2-3B-Instruct (4-bit) | 415 | **0.727 ± 0.032** | 0.399 ± 0.058 | 0.695 ± 0.026 | 0.654 ± 0.063 | 0.663 ± 0.058 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 415 | **0.771 ± 0.034** | 0.475 ± 0.057 | 0.718 ± 0.022 | 0.656 ± 0.077 | 0.686 ± 0.068 |
| *Paper: Llama-2-7B-chat (Sun et al. 2025, Table 1)* | | *0.7325* | *0.3979* | | | |

## Black-box LLM judges (GPT / Gemini / LLaMA)

Same examples, no training. Score = judge's stated hallucination probability.

| Judge | n | AUC | PCC | Acc | Rec | F1 |
|---|---|---|---|---|---|---|
| llama: unsloth/Llama-3.2-3B-Instruct | 415 | 0.523 | 0.100 | 0.513 | 0.085 | 0.151 |

## Extension: robustness under corrupted retrieval

Same response, different context. Detector refit excluding the perturbed examples.

| Backbone | Condition | Mean ReDeEP score | ECS (E_A) | PKS (P_F) | AUC vs. orig. labels | Score shift vs. accurate (Wilcoxon p) |
|---|---|---|---|---|---|---|
| llama2_7b_4bit | accurate | 0.521 | 0.282 | 0.091 | 0.710 | — |
| llama2_7b_4bit | irrelevant | 0.770 | 0.288 | 0.114 | 0.564 | +0.250 (p=9.7e-25) |
| llama2_7b_4bit | contradictory | 0.510 | 0.310 | 0.092 | 0.715 | -0.010 (p=5.7e-06) |
| tinyllama_1.1b | accurate | 0.522 | 0.326 | 0.158 | 0.755 | — |
| tinyllama_1.1b | irrelevant | 0.840 | 0.244 | 0.187 | 0.545 | +0.317 (p=4.0e-26) |
| tinyllama_1.1b | contradictory | 0.528 | 0.330 | 0.160 | 0.747 | +0.005 (p=4.0e-01) |

- **llama2_7b_4bit**, restricted to the 122 examples whose response states a number the corrupted context now contradicts: score shift -0.009 (p=2.0e-04), score increased in only 31% of cases, ECS shift +0.032.
- **tinyllama_1.1b**, restricted to the 122 examples whose response states a number the corrupted context now contradicts: score shift +0.006 (p=2.8e-01), score increased in only 52% of cases, ECS shift +0.004.
