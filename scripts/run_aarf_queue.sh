#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY="${PY:-/c/Users/Adithya K/venvs/deepLenv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export GEMINI_API_KEY="$(powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')" | tr -d '\r')"
echo "===== aarf gen $(date)";   "$PY" scripts/run_aarf.py --model NousResearch/Llama-2-7b-chat-hf --load_in_4bit --hp official:test_llama2_7B --tag llama2_7b > data/logs/aarf_gen_llama2_7b.log 2>&1; echo "  exit=$?"
echo "===== aarf judge $(date)"; "$PY" scripts/judge_aarf.py --tag llama2_7b --judge_model gemini-3.5-flash-lite > data/logs/aarf_judge_llama2_7b.log 2>&1; echo "  exit=$?"
echo QUEUEDONE
