"""Extension beyond the paper: robustness of ReDeEP's ECS/PKS signals
under deliberately corrupted retrieval.

The paper explicitly separates "RAG Hallucination" (model contradicts
*accurate* retrieved context) from "Knowledge Conflict" (retrieved
context itself is wrong/contradictory) and scopes their method to the
former. This module tests what happens to the ECS/PKS/ReDeEP signals
when we cross that line on purpose:

  1. accurate     -- the real, correct retrieved passage (control)
  2. irrelevant   -- swap in a passage from a *different* source_id
                     (tests: does ECS collapse when there's nothing
                     correct to attend to?)
  3. contradictory -- programmatically negate/alter a key fact in the
                     passage (e.g. swap a number/entity) so it directly
                     conflicts with the question (tests: does the model
                     lean on Copying Heads to faithfully reproduce the
                     *wrong* context, and does PKS/ECS still predict
                     the RAGTruth-style hallucination label, or does it
                     now reflect "faithfully repeating wrong info"?)

This lets us report, as a novel contribution, whether ReDeEP's decoupled
covariates are actually measuring "unsupported by context" (robust
under 2 and 3) versus secretly measuring "matches world knowledge"
(which would break under 3).
"""
import random
import re
import zlib

import numpy as np


def swap_source_info(example, all_examples, rng=None):
    """Condition 2: irrelevant context -- replace this example's source
    passage with another (unrelated) example's, keep the same question."""
    rng = rng or random.Random(0)
    same_task = [e for e in all_examples if e["task_type"] == example["task_type"]
                 and e["source_id"] != example["source_id"]]
    if not same_task:
        return None
    donor = rng.choice(same_task)
    return donor["prompt"]


_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def perturb_numbers(text, rng=None, scale_range=(1.3, 3.0)):
    """Condition 3 (cheap, deterministic corruption for numeric/data tasks):
    multiply every standalone number in the passage by a random factor,
    so any number the model copies from context is now factually wrong."""
    rng = rng or random.Random(0)

    def _replace(m):
        try:
            val = float(m.group(0))
        except ValueError:
            return m.group(0)
        factor = rng.uniform(*scale_range)
        new_val = val * factor
        if val.is_integer():
            return str(int(round(new_val)))
        return f"{new_val:.2f}"

    return _NUMBER_RE.sub(_replace, text)


def build_robustness_conditions(example, all_examples, rng=None):
    # crc32, not hash(): str hashing is randomised per process, which made
    # the corrupted contexts differ from run to run
    rng = rng or random.Random(zlib.crc32(str(example["id"]).encode()))
    conditions = {"accurate": example["prompt"]}
    swapped = swap_source_info(example, all_examples, rng)
    if swapped is not None:
        conditions["irrelevant"] = swapped
    conditions["contradictory"] = perturb_numbers(example["prompt"], rng)
    return conditions


def run_robustness_sweep(mech, examples, all_examples, n=10, max_len=768,
                         heads_A=None, layers_F=None):
    """For a handful of examples, compute response-level ECS/PKS under each
    condition and return a list of per-example dicts for later aggregation
    / plotting. Uses the example's *original* response for scoring, i.e.
    tests how the detector's internal signals shift purely due to a
    corrupted context, holding the (possibly hallucinated) response fixed.
    If heads_A / layers_F (the detector's selected Copying Heads and
    Knowledge FFNs) are given, the detector-relevant aggregates E_A / P_F
    are recorded too."""
    results = []
    for ex in examples[:n]:
        conditions = build_robustness_conditions(ex, all_examples)
        row = {"id": ex["id"], "label": ex["label"], "task_type": ex["task_type"]}
        for cond_name, prompt_text in conditions.items():
            prompt_ids = mech.tokenizer(prompt_text, return_tensors="pt").input_ids[0]
            response_ids = mech.tokenizer(ex["response"], return_tensors="pt",
                                           add_special_tokens=False).input_ids[0]
            if response_ids.shape[0] == 0:
                continue
            scored = mech.score_example(prompt_ids, response_ids, max_len=max_len)
            row[f"E_mean_{cond_name}"] = float(scored["E"].mean())
            row[f"P_mean_{cond_name}"] = float(scored["P"].mean())
            if heads_A:
                row[f"E_A_{cond_name}"] = float(np.mean([scored["E"][l, h] for l, h in heads_A]))
            if layers_F:
                row[f"P_F_{cond_name}"] = float(np.mean(scored["P"][list(layers_F)]))
        results.append(row)
    return results
