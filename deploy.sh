#!/usr/bin/env bash
set -euo pipefail

cd /opt/crypto_strategy_project

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

if [ -f requirements.txt ]; then
  python -m pip install --upgrade pip
  pip install -r requirements.txt
fi

# Run migrations or other initialization commands as needed
# Example: python manage.py migrate

systemctl restart trader

echo "[deploy] done at $(date '+%Y-%m-%d %H:%M:%S')"
