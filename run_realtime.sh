#!/usr/bin/env bash
set -euo pipefail

cd /opt/crypto_strategy_project

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

echo "$(date) Start realtime loop..."
exec python scripts/realtime_loop.py --cfg csp/configs/strategy.yaml --delay-sec 15
