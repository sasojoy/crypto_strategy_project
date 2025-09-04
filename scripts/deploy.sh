#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(pwd)"
TARGET_DIR="/opt/crypto_strategy_project"

echo "[DEPLOY] Sync code to ${TARGET_DIR}"
sudo mkdir -p "${TARGET_DIR}"
sudo rsync -a --delete "${REPO_DIR}/" "${TARGET_DIR}/"

if [ -d "${REPO_DIR}/systemd" ]; then
  echo "[DEPLOY] Install systemd units"
  for unit in ${REPO_DIR}/systemd/*.service ${REPO_DIR}/systemd/*.timer; do
    [ -e "$unit" ] || continue
    sudo cp "$unit" "/etc/systemd/system/$(basename "$unit")"
  done
  sudo systemctl daemon-reload
  sudo systemctl enable trader-once.timer >/dev/null 2>&1 || true
  sudo systemctl enable trader.timer >/dev/null 2>&1 || true
fi

sudo systemctl start trader-once.service || true

echo "[DEPLOY] done"
