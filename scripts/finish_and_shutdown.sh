#!/usr/bin/env bash
# Wait for the GPT judge (run_api_judges.sh) and the Gemini Flash-Lite judge to
# finish (test split, then all 415), summarise results, then shut down Windows.
# Safety cap: shuts down after 4 hours even if a judge hangs.
cd "$(dirname "$0")/.."
PY="/c/Users/Adithya K/venvs/deepLenv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export GEMINI_API_KEY="$(powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')" | tr -d '\r')"
LOG=data/logs/finish_and_shutdown.log
start=$(date +%s); cap=$((4*3600))
alive() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count" | tr -d '\r'; }
echo "started $(date)" >> $LOG
# 1) Gemini lite: wait for the test-split run, then run the full set (resumable, stops on quota)
while [ "$(alive 'gemini-3.5-flash-lite')" != "0" ] && [ $(( $(date +%s) - start )) -lt $cap ]; do sleep 30; done
echo "gemini test split finished $(date)" >> $LOG
timeout $(( cap - ($(date +%s) - start) )) "$PY" scripts/run_judges.py --provider gemini --judge_model gemini-3.5-flash-lite --min_interval 5 >> data/logs/judge_gemini_lite.log 2>&1
echo "gemini full finished $(date)" >> $LOG
# 2) GPT: wait for run_api_judges.sh to print ALLDONE
until grep -q ALLDONE data/logs/api_judges.log || [ $(( $(date +%s) - start )) -ge $cap ]; do sleep 30; done
echo "gpt finished $(date)" >> $LOG
"$PY" scripts/summarize.py > /dev/null 2>&1
git add -A >/dev/null 2>&1; git -c user.email=aditgouriii@gmail.com -c user.name="Adithya K" commit -qm "GPT and Gemini judge results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>&1
echo "shutting down $(date)" >> $LOG
shutdown //s //f //t 120 //c "Deep learning runs finished - shutting down in 2 minutes"
