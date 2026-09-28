"""AARF -- Add Attention, Reduce FFN (Section 4.3 of the ReDeEP paper).

A training-free decoding-time intervention: for the selected Copying
Heads (set A) we amplify their attention-output contribution to the
residual stream by alpha2 > 1; for the selected Knowledge FFN layers
(set F) we damp their contribution by beta2 in (0, 1). This nudges
generation toward relying more on retrieved context and less on
parametric knowledge, without touching any model weights.

Implemented via forward hooks on `self_attn` (o_proj output, i.e. the
per-layer attention contribution before it's added to the residual) and
`mlp` (the FFN contribution before it's added to the residual).
"""
import torch


class AARFIntervention:
    def __init__(self, model, heads_A, layers_F, alpha2=5.0, beta2=0.2):
        self.model = model
        self.heads_A_by_layer = {}
        for (l, h) in heads_A:
            self.heads_A_by_layer.setdefault(l, []).append(h)
        self.layers_F = set(layers_F)
        self.alpha2 = alpha2
        self.beta2 = beta2
        self._handles = []

        cfg = model.config
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

    def _attn_prehook(self, layer_idx):
        heads = self.heads_A_by_layer.get(layer_idx, [])
        head_dim = self.head_dim
        alpha2 = self.alpha2

        def hook(module, args, kwargs):
            # o_proj input z is the concatenation over heads of each head's
            # pre-output-projection value: y = sum_h z[..., slice_h] @ Wo_h^T.
            # Scaling z's slice for head h by alpha2 is therefore exactly
            # equivalent to scaling that head's Attn^{l,h} contribution
            # (Eq. 7) by alpha2, matching the paper's intervention exactly.
            if not heads:
                return args, kwargs
            z = args[0].clone()
            for h in heads:
                sl = slice(h * head_dim, (h + 1) * head_dim)
                z[..., sl] = z[..., sl] * alpha2
            return (z,) + args[1:], kwargs
        return hook

    def _mlp_hook(self, layer_idx):
        def hook(module, inputs, output):
            if layer_idx not in self.layers_F:
                return output
            return output * self.beta2
        return hook

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            h1 = layer.self_attn.o_proj.register_forward_pre_hook(
                self._attn_prehook(i), with_kwargs=True
            )
            h2 = layer.mlp.register_forward_hook(self._mlp_hook(i))
            self._handles.extend([h1, h2])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles = []


def generate_with_and_without_aarf(mech, prompt_text, heads_A, layers_F,
                                    alpha2=5.0, beta2=0.2, max_new_tokens=128):
    tok = mech.tokenizer
    inputs = tok(prompt_text, return_tensors="pt")
    with torch.no_grad():
        baseline_ids = mech.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    baseline_text = tok.decode(baseline_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    with AARFIntervention(mech.model, heads_A, layers_F, alpha2, beta2):
        with torch.no_grad():
            aarf_ids = mech.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
    aarf_text = tok.decode(aarf_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return {"baseline": baseline_text, "aarf": aarf_text}
