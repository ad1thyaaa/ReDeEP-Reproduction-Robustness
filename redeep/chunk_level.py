"""ReDeEP(chunk): chunk-level External Context Score / Parametric
Knowledge Score, following the official implementation
(Jeryi-Sun/ReDEeP-ICLR, ReDeEP/chunk_level_detect.py):

  * prompt and response are split into the chunks given by the official
    `prompt_spans` / `response_spans` (character offsets);
  * ECS(head, response chunk): the prompt chunk receiving the most
    attention mass from that response chunk under that copying head is
    found, and the BGE-embedding cosine similarity between the two chunk
    texts is the score;
  * PKS(layer, response chunk): sum over the chunk's tokens of the
    Jensen-Shannon divergence between the logit-lens distributions just
    before the FFN and after the layer (official `calculate_dist_2d`).

Attention is never materialised for all heads: a forward hook on each
layer keeps only the selected copying heads' response->prompt-chunk
attention sums, so long RAGTruth prompts fit in 6 GB with a 4-bit 7B model.
"""
import torch
import torch.nn.functional as F

LLAMA2_SYS = "You are a helpful assistant."


def chat_prefix(tokenizer, prompt):
    """The official code wraps the prompt in the chat template with a
    system message. Llama-2 tokenizers on the Hub often lack a template,
    so the standard Llama-2 chat format is written out in that case."""
    msgs = [{"role": "system", "content": LLAMA2_SYS}, {"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"[INST] <<SYS>>\n{LLAMA2_SYS}\n<</SYS>>\n\n{prompt} [/INST]"


class BGE:
    def __init__(self, name="BAAI/bge-base-en-v1.5", device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts):
        x = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        return F.normalize(self.model(**x).last_hidden_state[:, 0].float(), dim=-1)


def _char_to_token_spans(offsets, spans, base):
    """Map character spans (relative to `base`) to [start, end) token index ranges."""
    out = []
    for a, b in spans:
        a, b = a + base, b + base
        toks = [i for i, (s, e) in enumerate(offsets) if e > a and s < b and e > s]
        out.append((toks[0], toks[-1] + 1) if toks else None)
    return out


def _jsd_sum(logits_a, logits_b):
    la, lb = F.log_softmax(logits_a.float(), -1), F.log_softmax(logits_b.float(), -1)
    m = 0.5 * (la.exp() + lb.exp())
    kl1 = F.kl_div(la, m, reduction="none").sum(-1)
    kl2 = F.kl_div(lb, m, reduction="none").sum(-1)
    return float((0.5 * (kl1 + kl2)).sum())


class ChunkScorer:
    def __init__(self, mech, copy_heads, bge, max_tokens=2048, prompt_chars=12000):
        self.mech, self.bge = mech, bge
        self.copy_heads = [tuple(h) for h in copy_heads]
        self.by_layer = {}
        for l, h in self.copy_heads:
            self.by_layer.setdefault(l, []).append(h)
        self.max_tokens, self.prompt_chars = max_tokens, prompt_chars
        self._req = None        # (r_spans, p_spans) while scoring
        self._span_attn = {}    # (l, h) -> [n_r, n_p] attention mass
        for l, layer in enumerate(mech.model.model.layers):
            if l in self.by_layer:
                layer.self_attn.register_forward_hook(self._hook(l))

    def _hook(self, l):
        heads = self.by_layer[l]

        def hook(module, inputs, output):
            if self._req is None:
                return
            attn = output[1][0]                                    # [H, n, n]
            r_spans, p_spans = self._req
            for h in heads:
                a = attn[h].float()
                cum = torch.cumsum(a, dim=1)                       # prefix sums over key positions
                pad = torch.cat([torch.zeros(a.shape[0], 1, device=a.device), cum], 1)
                mass = torch.zeros(len(r_spans), len(p_spans))
                for i, (r0, r1) in enumerate(r_spans):
                    rows = pad[r0:r1]                              # [len_r, n+1]
                    for j, (p0, p1) in enumerate(p_spans):
                        mass[i, j] = float((rows[:, p1] - rows[:, p0]).sum())
                self._span_attn[(l, h)] = mass
        return hook

    @torch.no_grad()
    def score(self, prompt, prompt_spans, response, response_spans, label_spans=()):
        tok, mech = self.mech.tokenizer, self.mech
        prompt = prompt[: self.prompt_chars]
        prefix = chat_prefix(tok, prompt)
        base_p = prefix.find(prompt[:200])
        text = prefix + response
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        p_spans = [(max(0, a), min(len(prompt), b)) for a, b in prompt_spans if a < len(prompt)]
        p_tok = _char_to_token_spans(offs, p_spans, base_p)
        r_tok = _char_to_token_spans(offs, response_spans, len(prefix))

        # left-truncate the prompt if the sequence is too long; drop prompt
        # chunks that fall (partly) outside the kept window
        cut = max(0, len(ids) - self.max_tokens)
        keep_p = [(i, (a - cut, b - cut)) for i, s in enumerate(p_tok) if s is not None for a, b in [s] if a >= cut]
        keep_r = [(i, (a - cut, b - cut)) for i, s in enumerate(r_tok) if s is not None for a, b in [s]]
        if not keep_p or not keep_r:
            return None
        ids = ids[cut:]

        self._span_attn.clear()
        self._req = ([s for _, s in keep_r], [s for _, s in keep_p])
        mech._mid_hidden.clear()
        try:
            out = mech.model(input_ids=torch.tensor([ids], device=mech.device), output_hidden_states=True, use_cache=False)
        finally:
            self._req = None

        # ECS: BGE similarity of each response chunk with its most-attended prompt chunk
        r_text = [response[response_spans[i][0]:response_spans[i][1]] for i, _ in keep_r]
        p_text = [prompt[p_spans[i][0]:p_spans[i][1]] for i, _ in keep_p]
        sim = (self.bge.embed(r_text) @ self.bge.embed(p_text).T).cpu()          # [n_r, n_p]
        E = torch.zeros(len(keep_r), len(self.copy_heads))
        for c, lh in enumerate(self.copy_heads):
            best = self._span_attn[lh].argmax(dim=1)
            E[:, c] = sim[torch.arange(len(keep_r)), best]

        # PKS: summed JSD over each response chunk's tokens, per layer
        hs = out.hidden_states
        norm, head = mech.model.model.norm, mech.model.lm_head
        L = mech.num_layers
        P = torch.zeros(len(keep_r), L)
        for l in range(L):
            post_all, mid_all = hs[l + 1][0], mech._mid_hidden[l][0]
            for i, (_, (r0, r1)) in enumerate(keep_r):
                post = post_all[r0:r1] if l == L - 1 else norm(post_all[r0:r1])   # last hidden state is already normed
                P[i, l] = _jsd_sum(head(post), head(norm(mid_all[r0:r1])))
        mech._mid_hidden.clear()
        del out, hs

        # chunk labels: overlap with any annotated hallucination span (characters)
        lab = []
        for i, _ in keep_r:
            a, b = response_spans[i]
            lab.append(int(any(s["start"] < b and s["end"] > a for s in label_spans)))
        return {"E": E.numpy(), "P": P.numpy(), "chunk_labels": lab,
                "n_tokens": len(ids), "truncated": cut > 0}
