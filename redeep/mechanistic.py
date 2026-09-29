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


def _dense_weight(linear):
    """Return a Linear layer's weight as a dense fp32 tensor, dequantizing
    bitsandbytes 4-bit weights (used when the model is loaded with
    --load_in_4bit so a 7B model fits in 6GB of VRAM)."""
    w = linear.weight
    if getattr(w, "quant_state", None) is not None:
        import bitsandbytes.functional as bnbF
        return bnbF.dequantize_4bit(w.data, w.quant_state).float()
    return w.detach().float()


class LlamaMechInterp:
    def __init__(self, model_name, device="cpu", dtype=torch.float32, cache_dir=None,
                 load_in_4bit=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, cache_dir=cache_dir, attn_implementation="eager",
                quantization_config=qcfg, device_map={"": device},
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, cache_dir=cache_dir, attn_implementation="eager"
            ).to(device)
        self.model.eval()

        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size

        self._mid_hidden = {}
        self._register_mid_hooks()
        # Per-layer top-k attended context positions, filled by attention
        # hooks during score_example. Keeping only these (instead of
        # requesting every full [H, n, n] map via output_attentions) is
        # what lets 7B-scale models fit in 6GB of VRAM.
        self._attn_topk = {}
        self._topk_req = None   # (resp_idx, n_prompt, k) while scoring, else None
        self._register_attn_hooks()

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

    def _register_attn_hooks(self):
        def make_hook(layer_idx):
            def hook(module, inputs, output):
                if self._topk_req is None:
                    return
                attn = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if attn is None:
                    raise RuntimeError("attention weights unavailable; model must use attn_implementation='eager'")
                resp_idx, n_prompt, k = self._topk_req
                ctx_scores = attn[0][:, resp_idx, :n_prompt]              # [H, R, n_prompt]
                self._attn_topk[layer_idx] = torch.topk(ctx_scores, k, dim=-1).indices   # [H, R, k]
            return hook

        for i, layer in enumerate(self.model.model.layers):
            layer.self_attn.register_forward_hook(make_hook(i))

    # ------------------------------------------------------------------
    # Static, input-independent: Copying Head Score C^{l,h} (Appendix B)
    # ------------------------------------------------------------------
    def copying_head_scores(self, cache_path=None):
        if cache_path and Path(cache_path).exists():
            return np.load(cache_path)

        WE = self.model.model.embed_tokens.weight.detach()          # [V, d]
        WU = _dense_weight(self.model.lm_head)                       # [V, d] (LLaMA ties shapes this way)
        vocab_sample = min(500, WE.shape[0])
        # Full [V,V] eigen-analysis is infeasible on CPU for large vocabularies;
        # subsample a fixed random set of vocab rows (fixed seed) as in
        # standard OV-circuit probing practice.
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(WE.shape[0], generator=g)[:vocab_sample].to(WE.device)
        WE_s = WE[idx].float()   # [v, d]  (upcast: eigen/trace math wants fp32 precision)
        WU_s = WU[idx].float()   # [v, d]
        del WU

        layers = self.model.model.layers
        traces = np.zeros((self.num_layers, self.num_heads))
        outlier_counts = np.zeros((self.num_layers, self.num_heads))

        cfg = self.model.config
        num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        n_rep = self.num_heads // num_kv_heads   # query heads sharing each KV head (GQA)

        with torch.no_grad():
            for l, layer in enumerate(layers):
                attn = layer.self_attn
                Wv = _dense_weight(attn.v_proj)   # [num_kv_heads*dh, d] (out,in) Linear convention
                Wo = _dense_weight(attn.o_proj)   # [d, num_heads*dh]
                for h in range(self.num_heads):
                    kv_h = h // n_rep
                    kv_sl = slice(kv_h * self.head_dim, (kv_h + 1) * self.head_dim)
                    q_sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                    Wv_h = Wv[kv_sl, :]               # [dh, d]  (this head's shared KV projection)
                    Wo_h = Wo[:, q_sl]                # [d, dh]  (this query head's output slice)
                    # OV circuit for this head: x -> x @ Wv_h^T @ Wo_h^T
                    # (associativity: compute through the dh bottleneck
                    # instead of materialising the [d, d] matrix)
                    M = ((WE_s @ Wv_h.T) @ Wo_h.T) @ WU_s.T   # [v, v]
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
                del M, Wv, Wo
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
    # One forward pass producing everything needed for ECS + PKS (and the
    # perplexity / LN-entropy baselines) for a single (prompt, response).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_example(self, prompt_ids, response_ids, top_k_pct=10, max_len=1024):
        """prompt_ids, response_ids: 1D LongTensor of token ids (no batch dim).
        Returns dict with:
          E: np.array [num_layers, num_heads]   response-level ECS per head
          P: np.array [num_layers]              response-level PKS per layer
          ppl, lnent: perplexity / length-normalised entropy of the response
        Sequences longer than max_len are truncated from the left on the
        prompt side (keep all of the response).
        """
        response_ids = response_ids[: max_len // 2]   # guarantee some prompt survives truncation
        full_ids = torch.cat([prompt_ids, response_ids], dim=0)
        if full_ids.shape[0] > max_len:
            overflow = full_ids.shape[0] - max_len
            prompt_ids = prompt_ids[overflow:]
            full_ids = torch.cat([prompt_ids, response_ids], dim=0)
        n_prompt = prompt_ids.shape[0]
        n_total = full_ids.shape[0]

        response_positions = list(range(n_prompt, n_total))
        if len(response_positions) == 0:
            response_positions = [n_total - 1]
        resp_idx = torch.tensor(response_positions, device=self.device)
        k = max(1, int(np.ceil(top_k_pct / 100.0 * n_prompt)))
        k_eff = min(k, n_prompt)

        self._mid_hidden.clear()
        self._attn_topk.clear()
        self._topk_req = (resp_idx, n_prompt, k_eff)
        try:
            input_ids = full_ids.unsqueeze(0).to(self.device)
            out = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        finally:
            self._topk_req = None
        # hidden_states: tuple len L+1, each [1, n, d]; hidden_states[l+1] == x^l
        hidden_states = [h[0] for h in out.hidden_states]     # list len L+1
        final_hidden = hidden_states[-1]                       # x^L  [n, d]

        E = np.zeros((self.num_layers, self.num_heads))
        norm_final = F.normalize(final_hidden.float(), dim=-1)   # [n, d]
        norm_final_resp = norm_final[resp_idx]                    # [R, d]
        norm_final_ctx = norm_final[:n_prompt]                    # [n_prompt, d]

        for l in range(self.num_layers):
            topk_idx = self._attn_topk[l]                           # [H, R, k]
            # mean of the top-k attended context rows, done as a (sparse)
            # averaging matrix product rather than gathering [H, R, k, d]
            H, R, _ = topk_idx.shape
            avg = torch.zeros(H, R, n_prompt, device=norm_final.device)
            avg.scatter_(-1, topk_idx, 1.0 / k_eff)
            e = avg @ norm_final_ctx                                # [H, R, d]
            e = F.normalize(e, dim=-1)
            cos = (e * norm_final_resp.unsqueeze(0)).sum(dim=-1)   # [H, R]
            E[l, :] = cos.mean(dim=1).cpu().numpy()

        P = np.zeros(self.num_layers)
        final_norm_module = self.model.model.norm
        lm_head = self.model.lm_head
        for l in range(self.num_layers):
            x_mid = self._mid_hidden[l][0]          # [n, d]
            x_post = hidden_states[l + 1]            # [n, d]  (x^l, after full layer)
            logits_mid = lm_head(final_norm_module(x_mid[resp_idx])).float()
            logits_post = lm_head(final_norm_module(x_post[resp_idx])).float()
            p_mid = F.log_softmax(logits_mid, dim=-1)
            p_post = F.log_softmax(logits_post, dim=-1)
            m = 0.5 * (p_mid.exp() + p_post.exp())
            jsd = 0.5 * F.kl_div(p_mid, m, reduction="none").sum(-1) + \
                  0.5 * F.kl_div(p_post, m, reduction="none").sum(-1)
            P[l] = jsd.mean().item()
            del logits_mid, logits_post, p_mid, p_post, m

        # Baselines from the same forward pass: logits[i-1] predicts token i.
        pred_pos = [i - 1 for i in range(n_prompt, n_total) if i > 0]
        if pred_pos:
            pred_pos_t = torch.tensor(pred_pos, device=self.device)
            logp = F.log_softmax(out.logits[0, pred_pos_t].float(), dim=-1)   # [R, V]
            targets = input_ids[0, pred_pos_t + 1]
            nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            ent = -(logp.exp() * logp).sum(-1)
            ppl = float(torch.exp(nll.mean()).item())
            lnent = float(ent.mean().item())
        else:
            ppl = lnent = float("nan")

        del out, hidden_states, final_hidden
        self._mid_hidden.clear()
        self._attn_topk.clear()
        gc.collect()

        return {"E": E, "P": P, "ppl": ppl, "lnent": lnent,
                "n_prompt": n_prompt, "n_response": len(response_positions)}
