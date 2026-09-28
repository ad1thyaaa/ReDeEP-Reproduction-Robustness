"""Core mechanistic-interpretability primitives from the ReDeEP paper
(Sun et al., ICLR 2025), implemented generically against any HF
LLaMA-architecture causal LM (LlamaForCausalLM / TinyLlama / etc.):

  - Copying Head Score  C^{l,h}   (Appendix B: trace + Gershgorin/IQR)
  - External Context Score  E^{l,h}_n  (Eq. 2-4)
  - Parametric Knowledge Score  P^l_n  (Eq. 5-6, via LogitLens JSD)

All of this assumes a "LlamaModel"-style decoder: pre-norm residual
stream, `model.model.layers[i]` each with `.self_attn`, `.mlp`,
`.input_layernorm`, `.post_attention_layernorm`, and a tied/untied
`lm_head` for unembedding. TinyLlama, Llama-2, and Llama-3 chat models
all share this structure, so the same code runs against any of them --
only the model name changes.
"""
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class LlamaMechInterp:
    def __init__(self, model_name, device="cpu", dtype=torch.float32, cache_dir=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, cache_dir=cache_dir, attn_implementation="eager"
        ).to(device)
        self.model.eval()

        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size

        self._mid_hidden = {}
        self._register_mid_hooks()

    # ------------------------------------------------------------------
    # Hook plumbing: capture x^{mid,l} (residual stream right before the
    # FFN block, i.e. the input to post_attention_layernorm).
    # ------------------------------------------------------------------
    def _register_mid_hooks(self):
        layers = self.model.model.layers

        def make_hook(layer_idx):
            def hook(module, inputs):
                self._mid_hidden[layer_idx] = inputs[0].detach()
            return hook

        for i, layer in enumerate(layers):
            layer.post_attention_layernorm.register_forward_pre_hook(make_hook(i))

    # ------------------------------------------------------------------
    # Static, input-independent: Copying Head Score C^{l,h} (Appendix B)
    # ------------------------------------------------------------------
    def copying_head_scores(self, cache_path=None):
        if cache_path and Path(cache_path).exists():
            return np.load(cache_path)

        WE = self.model.model.embed_tokens.weight.detach()          # [V, d]
        WU = self.model.lm_head.weight.detach()                      # [V, d] (LLaMA ties shapes this way)
        vocab_sample = min(500, WE.shape[0])
        # Full [V,V] eigen-analysis is infeasible on CPU for large vocabularies;
        # subsample a fixed random set of vocab rows (fixed seed) as in
        # standard OV-circuit probing practice.
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(WE.shape[0], generator=g)[:vocab_sample].to(WE.device)
        WE_s = WE[idx].float()   # [v, d]  (upcast: eigen/trace math wants fp32 precision)
        WU_s = WU[idx].float()   # [v, d]

        layers = self.model.model.layers
        traces = np.zeros((self.num_layers, self.num_heads))
        outlier_counts = np.zeros((self.num_layers, self.num_heads))

        cfg = self.model.config
        num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        n_rep = self.num_heads // num_kv_heads   # query heads sharing each KV head (GQA)

        with torch.no_grad():
            for l, layer in enumerate(layers):
                attn = layer.self_attn
                Wv = attn.v_proj.weight.detach().float()   # [num_kv_heads*dh, d] (out,in) Linear convention
                Wo = attn.o_proj.weight.detach().float()   # [d, num_heads*dh]
                for h in range(self.num_heads):
                    kv_h = h // n_rep
                    kv_sl = slice(kv_h * self.head_dim, (kv_h + 1) * self.head_dim)
                    q_sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                    Wv_h = Wv[kv_sl, :]               # [dh, d]  (this head's shared KV projection)
                    Wo_h = Wo[:, q_sl]                # [d, dh]  (this query head's output slice)
                    # OV circuit for this head: x -> x @ Wv_h^T @ Wo_h^T
                    W_ov = Wv_h.T @ Wo_h.T             # [d, d]
                    M = WE_s @ W_ov @ WU_s.T            # [v, v]
                    tr = torch.trace(M).item()
                    traces[l, h] = tr

                    diag = torch.diagonal(M)
                    off_diag_sum = M.abs().sum(dim=1) - diag.abs()
                    upper = (diag + off_diag_sum)
                    lower = (diag - off_diag_sum)
                    pts = torch.cat([upper, lower]).cpu().numpy()
                    q1, q3 = np.percentile(pts, [25, 75])
                    iqr = q3 - q1
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outlier_counts[l, h] = int(((pts < lo) | (pts > hi)).sum())
                del M
                gc.collect()
                print(f"    copying-head layer {l+1}/{self.num_layers} done", flush=True)

        # rank: fewer outliers -> better; larger |trace| -> better
        rank_outliers = outlier_counts.flatten().argsort().argsort()          # ascending
        rank_trace = (-np.abs(traces).flatten()).argsort().argsort()          # descending abs(trace)
        combined = (rank_outliers + rank_trace).reshape(self.num_layers, self.num_heads)
        # lower combined rank = stronger copying head; convert to a score
        # where higher = more copying-head-like.
        score = combined.max() - combined
        score = score / max(score.max(), 1)

        if cache_path:
            np.save(cache_path, score)
        return score

    # ------------------------------------------------------------------
    # One forward pass producing everything needed for ECS + PKS for a
    # single (prompt, response) example.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_example(self, prompt_ids, response_ids, top_k_pct=10, max_len=1024):
        """prompt_ids, response_ids: 1D LongTensor of token ids (no batch dim).
        Returns dict with:
          E: np.array [num_layers, num_heads]   response-level ECS per head
          P: np.array [num_layers]              response-level PKS per layer
        Sequences longer than max_len are truncated from the left on the
        prompt side (keep all of the response).
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=0)
        if full_ids.shape[0] > max_len:
            overflow = full_ids.shape[0] - max_len
            prompt_ids = prompt_ids[overflow:]
            full_ids = torch.cat([prompt_ids, response_ids], dim=0)
        n_prompt = prompt_ids.shape[0]
        n_total = full_ids.shape[0]

        self._mid_hidden.clear()
        input_ids = full_ids.unsqueeze(0).to(self.device)
        out = self.model(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states: tuple len L+1, each [1, n, d]; hidden_states[l+1] == x^l
        hidden_states = [h[0] for h in out.hidden_states]     # list len L+1
        attentions = [a[0] for a in out.attentions]           # list len L, each [H, n, n]
        final_hidden = hidden_states[-1]                       # x^L  [n, d]

        k = max(1, int(np.ceil(top_k_pct / 100.0 * n_prompt)))

        E = np.zeros((self.num_layers, self.num_heads))
        response_positions = list(range(n_prompt, n_total))
        if len(response_positions) == 0:
            response_positions = [n_total - 1]

        norm_final = F.normalize(final_hidden, dim=-1)   # [n, d]
        resp_idx = torch.tensor(response_positions, device=final_hidden.device)
        k_eff = min(k, n_prompt)
        norm_final_resp = norm_final[resp_idx]           # [R, d]

        for l in range(self.num_layers):
            attn_l = attentions[l]                        # [H, n, n]
            ctx_scores = attn_l[:, resp_idx, :n_prompt]     # [H, R, n_prompt]
            topk_idx = torch.topk(ctx_scores, k_eff, dim=-1).indices   # [H, R, k]
            gathered = norm_final[topk_idx]                 # [H, R, k, d]
            e = gathered.mean(dim=2)                         # [H, R, d]
            e = F.normalize(e, dim=-1)
            cos = (e * norm_final_resp.unsqueeze(0)).sum(dim=-1)   # [H, R]
            E[l, :] = cos.mean(dim=1).cpu().numpy()

        P = np.zeros(self.num_layers)
        final_norm_module = self.model.model.norm
        lm_head = self.model.lm_head
        for l in range(self.num_layers):
            x_mid = self._mid_hidden[l][0]          # [n, d]
            x_post = hidden_states[l + 1]            # [n, d]  (x^l, after full layer)
            resp = torch.tensor(response_positions, device=x_mid.device)
            logits_mid = lm_head(final_norm_module(x_mid[resp])).float()
            logits_post = lm_head(final_norm_module(x_post[resp])).float()
            p_mid = F.log_softmax(logits_mid, dim=-1)
            p_post = F.log_softmax(logits_post, dim=-1)
            m = 0.5 * (p_mid.exp() + p_post.exp())
            jsd = 0.5 * F.kl_div(p_mid, m, reduction="none").sum(-1) + \
                  0.5 * F.kl_div(p_post, m, reduction="none").sum(-1)
            P[l] = jsd.mean().item()

        del out, hidden_states, attentions, final_hidden
        self._mid_hidden.clear()
        gc.collect()

        return {"E": E, "P": P, "n_prompt": n_prompt, "n_response": len(response_positions)}
