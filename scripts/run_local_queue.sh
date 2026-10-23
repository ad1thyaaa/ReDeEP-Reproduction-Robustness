#!/usr/bin/env bash
# Local GPU queue after the chunk runs: token-level Dolly, AARF generation, AARF judge.
cd "$(dirname "$0")/.."
PY="${PY:-/c/Users/Adithya K/venvs/deepLenv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export GEMINI_API_KEY="$(powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')" | tr -d '\r')"
until grep -q CHUNKDONE data/logs/chunk_local.log; do sleep 30; done
M=NousResearch/Llama-2-7b-chat-hf
echo "===== token dolly $(date)"; "$PY" scripts/run_token_official.py --model $M --load_in_4bit --tag tok_llama2_7b_dolly --dataset dolly --response_model llama-2-7b-chat > data/logs/tok_llama2_7b_dolly.log 2>&1; echo "  exit=$?"
echo "===== aarf gen $(date)";    "$PY" scripts/run_aarf.py --model $M --load_in_4bit --hp official:test_llama2_7B --tag llama2_7b > data/logs/aarf_gen_llama2_7b.log 2>&1; echo "  exit=$?"
echo "===== aarf judge $(date)";  "$PY" scripts/judge_aarf.py --tag llama2_7b --judge_model gemini-3.5-flash-lite > data/logs/aarf_judge_llama2_7b.log 2>&1; echo "  exit=$?"
echo QUEUEDONE
