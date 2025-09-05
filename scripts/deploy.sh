#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(pwd)"
TARGET_DIR="/opt/crypto_strategy_project"

echo "[DEPLOY] Sync code to ${TARGET_DIR}"
sudo mkdir -p "${TARGET_DIR}"
# 同步整個專案（可視需要調整排除清單）
sudo rsync -a --delete \
  --exclude='.git' \
  --exclude='.github' \
  --exclude='.venv' \
  "${REPO_DIR}/" "${TARGET_DIR}/"

cd "${TARGET_DIR}"

echo "[DEPLOY] Sanity check: systemd files"
if [ ! -d systemd ]; then
  echo "[ERR] systemd/ 資料夾不存在於 ${TARGET_DIR}"
  ls -la || true
  exit 1
fi
test -f systemd/trader-once.service || { echo "[ERR] 缺 systemd/trader-once.service"; ls -l systemd; exit 1; }
test -f systemd/trader-once.timer   || { echo "[ERR] 缺 systemd/trader-once.timer";   ls -l systemd; exit 1; }

echo "[DEPLOY] Install/overwrite systemd units"
sudo install -D -m 0644 systemd/trader-once.service /etc/systemd/system/trader-once.service
sudo install -D -m 0644 systemd/trader-once.timer   /etc/systemd/system/trader-once.timer

echo "[DEPLOY] Reload systemd and (re)enable timer"
sudo systemctl daemon-reload
sudo systemctl stop trader-once.service || true
sudo systemctl enable --now trader-once.timer
# 立刻跑一次一次性服務（可選）
sudo systemctl start trader-once.service || true

echo "[DEPLOY] Status"
sudo systemctl status trader-once.timer --no-pager || true
sudo systemctl status trader-once.service --no-pager || true

