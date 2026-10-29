#!/usr/bin/env bash
# Live loop: new news -> both production models -> SSI board (trading hours only). Meant for cron every 15 min.
# Each step is incremental (collect skips stored URLs, process skips scored versions), so a rerun is cheap.
set -u
cd "$(dirname "$0")/.."
exec 9>data/live.lock
flock -n 9 || { echo "$(date -Is) previous run still going, skip"; exit 0; }
export TZ=Asia/Ho_Chi_Minh
CLI=".venv/bin/python -m nlp_module.cli -v"
echo "=== $(date -Is) live update"
for s in cafef vietstock hose; do
    $CLI collect "$s" "$(date -d yesterday +%F)" "$(date +%F)" || echo "collect $s failed, continuing"
done
# thresholds from README (abstain on dev validation, ~73% coverage at 0.85 accuracy)
$CLI process --model baseline:data/models/baseline.joblib --abstain-entropy 0.888
$CLI process --model phobert:data/models/phobert-v1 --abstain-entropy 0.571
hm=$((10#$(date +%H%M)))
if [ "$(date +%u)" -le 5 ] && [ "$hm" -ge 900 ] && [ "$hm" -le 1515 ]; then $CLI quotes; fi
# own window, not nested in the quotes one: predict reads the stored post-close bar (live_bars only accepts fetched_at >= 08:00 UTC), so an early run harmlessly re-predicts the previous session
if [ "$(date +%u)" -le 5 ] && [ "$hm" -ge 1500 ] && [ "$hm" -le 1600 ]; then .venv/bin/python -m nlp_module.volcast predict || echo "volcast predict failed, continuing"; fi
