#!/usr/bin/env bash
# Chunk-level ReDeEP runs that fit the local 6 GB GPU (Llama-2-7B in 4-bit).
cd "$(dirname "$0")/.."
PY="${PY:-/c/Users/Adithya K/venvs/deepLenv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
M=NousResearch/Llama-2-7b-chat-hf
echo "===== ragtruth $(date)"; "$PY" scripts/run_chunk.py --model $M --load_in_4bit --tag llama2_7b_ragtruth --dataset ragtruth --response_model llama-2-7b-chat --copy_heads official:test_llama2_7B > data/logs/chunk_llama2_7b_ragtruth.log 2>&1; echo "  exit=$?"
echo "===== dolly $(date)";    "$PY" scripts/run_chunk.py --model $M --load_in_4bit --tag llama2_7b_dolly --dataset dolly --response_model llama-2-7b-chat --copy_heads official:test_llama2_7B > data/logs/chunk_llama2_7b_dolly.log 2>&1; echo "  exit=$?"
echo CHUNKDONE
