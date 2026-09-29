#!/usr/bin/env bash
# Robustness sweep + local LLaMA judge, after run_llama_family.sh.
cd "$(dirname "$0")/.."
PY="${PY:-/c/Users/Adithya K/venvs/deepLenv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
step() { name=$1; shift; echo "===== $name $(date)"; "$PY" "$@" > "data/logs/$name.log" 2>&1; echo "  exit=$? $(date)"; }
step robust_tinyllama  scripts/run_robustness.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tag tinyllama_1.1b --n 150
step robust_llama2_7b  scripts/run_robustness.py --model NousResearch/Llama-2-7b-chat-hf --tag llama2_7b_4bit --n 150 --load_in_4bit
step judge_llama3.2_3b scripts/run_judges.py --provider llama --judge_model unsloth/Llama-3.2-3B-Instruct
echo ALLDONE
