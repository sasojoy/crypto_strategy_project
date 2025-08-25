#!/usr/bin/env bash
set -euo pipefail
APP_DIR=/opt/crypto_strategy_project
cd "$APP_DIR"

# 確保所有權
chown -R deploy:deploy "$APP_DIR"

# 清除可能殘留的日期限制環境變數
unset START_DATE END_DATE

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

# 訓練最新模型（過去360天）
echo "[deploy] training multi-symbol models (360 days)"
DAYS=360 $PY scripts/train_multi.py --cfg csp/configs/strategy.yaml

# 60天資料回測優化
echo "[deploy] backtest optimization (60 days)"
$PY scripts/feature_optimize.py --cfg csp/configs/strategy.yaml --days 60

# 安裝 systemd service/timer
if [ -d systemd ]; then
  sudo cp systemd/trader.service /etc/systemd/system/trader.service
  sudo cp systemd/trader.timer /etc/systemd/system/trader.timer
  sudo systemctl daemon-reload
  sudo systemctl enable trader.timer
  sudo systemctl start trader.timer
fi

# 重啟/觸發一次性服務
#sudo -n /usr/bin/systemctl restart trader
sudo -n /usr/bin/systemctl start trader-once.service

echo "[deploy] done at $(date '+%Y-%m-%d %H:%M:%S')"
