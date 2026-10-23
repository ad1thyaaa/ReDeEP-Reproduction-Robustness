"""Gated AARF decoding, following the authors' modified LlamaForCausalLM
(ReDEeP-ICLR/transformers/.../modeling_llama.py):

at every decoding step the last token gets a token-level ReDeEP score
    pks  = mean over selected layers of JSD(logit-lens before FFN, after layer)
           (official calculate_dist: mean over vocab * 10e5)
    ecs  = mean over selected heads of cos(current hidden state,
           mean prefix hidden state of the top-10% attended prefix tokens)
    s    = norm(norm(pks; layers_max_min) - weight * norm(ecs; head_max_min); final_max_min)
and if s > threshold the step is recomputed with the selected copying heads'
outputs scaled by add_attention_weight and the selected FFNs' outputs scaled
by reduce_ffn_weight (no fine-tuning).
"""
import torch
import torch.nn.functional as F

from .aarf import AARFIntervention


def _norm(mm, v):
    mx, mn = mm
    return (v - mn) / (mx - mn) if mx != mn else 0.0


def _jsd_mean(a, b):
    la, lb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
    m = 0.5 * (la.exp() + lb.exp())
    return float(0.5 * (F.kl_div(la, m, reduction="none").mean(-1) + F.kl_div(lb, m, reduction="none").mean(-1))) * 10e5


class GatedAARF:
    def __init__(self, model, hp, add_attention_weight=1.2, reduce_ffn_weight=0.8, threshold=0.6):
        self.model, self.hp = model, hp
        self.heads = [tuple(h) for h in hp["select_heads"]]
        self.layers = list(hp["select_layers"])
        self.a2, self.b2, self.tau = add_attention_weight, reduce_ffn_weight, threshold
        self.n_layers = model.config.num_hidden_layers
        self._capture = False
        self._attn, self._mid = {}, {}
        hl = {}
        for l, h in self.heads:
            hl.setdefault(l, []).append(h)
        for l, layer in enumerate(model.model.layers):
            if l in hl:
                layer.self_attn.register_forward_hook(self._attn_hook(l, hl[l]))
            if l in self.layers:
                layer.post_attention_layernorm.register_forward_pre_hook(self._mid_hook(l))

    def _attn_hook(self, l, heads):
        def hook(m, i, out):
            if self._capture:
                for h in heads:
                    self._attn[(l, h)] = out[1][0, h, -1].float()     # last query row
        return hook

    def _mid_hook(self, l):
        def hook(m, inp):
            if self._capture:
                self._mid[l] = inp[0][0, -1]
        return hook

    def _step_score(self, out, prefix_hidden, prefix_len):
        norm, head = self.model.model.norm, self.model.lm_head
        hs = out.hidden_states
        pks = []
        for l in self.layers:
            post = hs[l + 1][0, -1] if l == self.n_layers - 1 else norm(hs[l + 1][0, -1])
            pks.append(_jsd_mean(head(post), head(norm(self._mid[l]))))
        pks = sum(pks) / len(pks)
        cur = hs[-1][0, -1].float()
        k = max(1, int(prefix_len * 0.1))
        cos = []
        for lh in self.heads:
            idx = torch.topk(self._attn[lh][:prefix_len], k).indices
            cos.append(float(F.cosine_similarity(prefix_hidden[idx].mean(0), cur, dim=0)))
        ecs = sum(cos) / len(cos)
        s = _norm(self.hp["layers_max_min"], pks) - self.hp["weight"] * _norm(self.hp["head_max_min"], ecs)
        g = _norm(self.hp["final_max_min"], s)
        self.trace.append((pks, ecs, g))
        return g

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=512, aarf=True, eos_ids=()):
        from transformers import DynamicCache
        cache = DynamicCache()
        self.trace = []           # (raw PKS, raw ECS, normalised gate score) per decoding step
        self._capture = True
        out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
        self._capture = False
        prefix_len = input_ids.shape[1]
        prefix_hidden = out.hidden_states[-1][0].float()
        tokens, n_interv = [], 0
        nxt = out.logits[0, -1].argmax().view(1, 1)
        for _ in range(max_new_tokens):
            t = int(nxt)
            if t in eos_ids:
                break
            tokens.append(t)
            before = cache.get_seq_length()
            self._capture = aarf
            out = self.model(input_ids=nxt, past_key_values=cache, use_cache=True, output_hidden_states=aarf)
            self._capture = False
            if aarf and self._step_score(out, prefix_hidden, prefix_len) > self.tau:
                # discard this step's keys/values, recompute with AARF. Newer
                # transformers only accept a negative crop (tokens to remove).
                try:
                    cache.crop(before)
                except ValueError:
                    cache.crop(before - cache.get_seq_length())
                with AARFIntervention(self.model, self.heads, self.layers, self.a2, self.b2):
                    out = self.model(input_ids=nxt, past_key_values=cache, use_cache=True)
                n_interv += 1
            nxt = out.logits[0, -1].argmax().view(1, 1)
        return tokens, n_interv
