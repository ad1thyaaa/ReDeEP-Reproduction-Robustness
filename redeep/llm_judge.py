"""Black-box LLM-as-judge hallucination detectors (README plan step 3).

Closed models (GPT, Gemini) don't expose attention / FFN internals, so
they can't run ReDeEP itself; instead they're used as external judges in
the style of the paper's "Prompt" baseline: given the RAG prompt (task +
retrieved context) and the response, the judge returns a 0-100 score for
how likely the response contains a hallucination. A local LLaMA judge is
included too so the black-box comparison has an open-weights point.

Providers (all but `openai` have a free tier, no card required):
  openai  -- needs OPENAI_API_KEY   (e.g. --judge_model gpt-4o-mini)
  github  -- GitHub Models, free GPT access; needs GITHUB_TOKEN (a PAT with
             the "models" permission)   (e.g. --judge_model openai/gpt-4o-mini)
  groq    -- Groq, free; needs GROQ_API_KEY   (e.g. --judge_model openai/gpt-oss-120b)
  gemini  -- Google AI Studio, free; needs GEMINI_API_KEY   (e.g. --judge_model gemini-2.5-flash)
  llama   -- local HF model         (e.g. --judge_model unsloth/Llama-3.2-3B-Instruct)
"""
import json
import os
import re
import time

JUDGE_SYSTEM = (
    "You are an expert fact-checker for retrieval-augmented generation. "
    "You will be given the input that was shown to an AI assistant (a task "
    "instruction plus retrieved reference content) and the assistant's response. "
    "Decide whether the response contains any hallucination: a claim that "
    "contradicts the reference content, or information that is not supported "
    "by the reference content. Judge only against the reference content, not "
    "against your own world knowledge."
)

JUDGE_TEMPLATE = """### Input shown to the assistant
{prompt}

### Assistant response
{response}

### Your task
Give the probability (0-100) that the response contains at least one hallucination.
Reply with JSON only, in the form {{"hallucination_probability": <integer 0-100>, "reason": "<one sentence>"}}"""


def parse_score(text):
    """Extract a 0-1 score from the judge's reply; None if unparsable."""
    if not text:
        return None
    m = re.search(r'"hallucination_probability"\s*:\s*"?(\d+(?:\.\d+)?)', text)
    if not m:
        m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", text)
    if not m:
        return None
    return min(max(float(m.group(1)), 0.0), 100.0) / 100.0


OPENAI_COMPATIBLE = {
    "openai": (None, "OPENAI_API_KEY"),
    "github": ("https://models.github.ai/inference", "GITHUB_TOKEN"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}


def _gh_cli_token():
    """Fall back to the GitHub CLI login (`gh auth login`) when GITHUB_TOKEN
    isn't set -- GitHub Models accepts the CLI's OAuth token."""
    import shutil
    import subprocess
    gh = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"
    try:
        return subprocess.run([gh, "auth", "token"], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


class OpenAIJudge:
    def __init__(self, model, provider="openai"):
        from openai import OpenAI
        base_url, key_env = OPENAI_COMPATIBLE[provider]
        key = os.environ.get(key_env)
        if not key and provider == "github":
            key = _gh_cli_token()
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.model = model

    def __call__(self, prompt, response):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": JUDGE_TEMPLATE.format(prompt=prompt, response=response)}],
            response_format={"type": "json_object"},
        )
        return r.choices[0].message.content


class GeminiJudge:
    def __init__(self, model):
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.config = types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM, response_mime_type="application/json")

    def __call__(self, prompt, response):
        r = self.client.models.generate_content(
            model=self.model, contents=JUDGE_TEMPLATE.format(prompt=prompt, response=response),
            config=self.config)
        return r.text


class LlamaJudge:
    def __init__(self, model, load_in_4bit=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(model)
        kw = {"dtype": torch.float16, "device_map": {"": "cuda"}}
        if load_in_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()

    def __call__(self, prompt, response):
        import torch
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_TEMPLATE.format(prompt=prompt, response=response)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to("cuda")
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=80, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def make_judge(provider, model):
    if provider in OPENAI_COMPATIBLE:
        return OpenAIJudge(model, provider)
    return {"gemini": GeminiJudge, "llama": LlamaJudge}[provider](model)


def run_judge(judge, examples, cache_path, max_retries=5, min_interval=0.0, max_consecutive_fail=3):
    """Judge every example, appending replies to a JSONL cache so
    interrupted runs resume and API calls are never repeated. Failed calls
    are not cached; after `max_consecutive_fail` failures in a row (e.g. a
    free-tier daily quota is exhausted) it stops, and re-running later
    picks up where it left off."""
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[row["id"]] = row
    n_fail = 0
    last_call = 0.0
    with open(cache_path, "a", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            if ex["id"] in done:
                continue
            reply, err = None, None
            for attempt in range(max_retries):
                time.sleep(max(0.0, min_interval - (time.time() - last_call)))
                last_call = time.time()
                try:
                    reply = judge(ex["prompt"], ex["response"])
                    break
                except Exception as e:   # rate limits / transient API errors
                    err = repr(e)[:300]
                    print(f"    retry {attempt+1}/{max_retries}: {err}", flush=True)
                    time.sleep(min(60, 2 ** attempt * 5))
            if reply is None:
                n_fail += 1
                print(f"  [{i+1}/{len(examples)}] FAILED: {err}", flush=True)
                if n_fail >= max_consecutive_fail:
                    print(f"  {n_fail} consecutive failures (quota exhausted?) -- stopping; "
                          f"re-run later to resume.", flush=True)
                    break
                continue
            n_fail = 0
            row = {"id": ex["id"], "label": ex["label"], "reply": reply, "score": parse_score(reply)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[ex["id"]] = row
            print(f"  [{i+1}/{len(examples)}] label={ex['label']} score={row['score']}", flush=True)
    return [done[ex["id"]] for ex in examples if ex["id"] in done]
