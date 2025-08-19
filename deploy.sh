#!/usr/bin/env bash
set -euo pipefail
APP_DIR=/opt/crypto_strategy_project
cd "$APP_DIR"

# 確保所有權
chown -R deploy:deploy "$APP_DIR"

# venv
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
PY=.venv/bin/python
PIP=.venv/bin/pip

$PY -m pip install --upgrade pip setuptools wheel

# 先試 editable 安裝；失敗就退回 requirements.txt
if [ -f pyproject.toml ] || [ -f setup.cfg ] || [ -f setup.py ]; then
  $PIP install -e . || { echo "[deploy] editable install failed, fallback..."; true; }
fi
[ -f requirements.txt ] && $PIP install -r requirements.txt || true

# 重啟/觸發
sudo -n /usr/bin/systemctl restart trader
# 或 timer 方案：sudo -n /usr/bin/systemctl start trader-once.service

echo "[deploy] done at $(date '+%Y-%m-%d %H:%M:%S')"
