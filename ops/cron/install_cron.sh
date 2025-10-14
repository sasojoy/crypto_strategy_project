#!/usr/bin/env bash
# ops/cron/install_cron.sh
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/strategy}"
# 以台北時間執行每天 08:05
( crontab -l 2>/dev/null; echo 'CRON_TZ=Asia/Taipei'; \
  echo "5 8 * * * cd ${APP_DIR} && /bin/bash scripts/run_daily_pipeline.sh >> logs/cron.log 2>&1" \
) | crontab -
echo "Cron installed. Check with: crontab -l"
