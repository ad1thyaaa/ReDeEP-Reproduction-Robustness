#!/usr/bin/env bash
# GPT (gpt-oss-120b via Groq free tier) and Gemini (AI Studio free tier)
# judges: held-out test split first, then all 415. Resumable.
cd "$(dirname "$0")/.."
PY="${PY:-/c/Users/Adithya K/venvs/deepLenv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
for v in GEMINI_API_KEY GROQ_API_KEY; do
  [ -z "${!v}" ] && export $v="$(powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('$v','User')" | tr -d '\r')"
done
judge() {  # name provider model interval
  for scope in --only_test_split ""; do
    "$PY" scripts/run_judges.py --provider $2 --judge_model $3 --min_interval $4 $scope >> data/logs/$1.log 2>&1
  done
  echo "$1 DONE $(date)"
}
judge judge_gpt    groq   openai/gpt-oss-120b  12 &
judge judge_gemini gemini gemini-3.6-flash     7  &
wait; echo ALLDONE
