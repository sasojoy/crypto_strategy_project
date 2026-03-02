#!/usr/bin/env bash
# scripts/run_daily_pipeline.sh
set -euo pipefail

# ---- settings ----
ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

# 載入環境變數（可在 /opt/strategy/.env 放 TELEGRAM_*、CSV_PATH、FEATURE_* 等）
if [[ -f ".env" ]]; then
  set -o allexport
  source .env
  set +o allexport
fi

mkdir -p logs
LOG_FILE="logs/daily_$(date -u +%Y%m%dT%H%M%SZ).log"

# 防重複（避免重疊執行）
LOCK="/tmp/run_daily_pipeline.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[SKIP] another run in progress" | tee -a "$LOG_FILE"
  exit 0
fi

# 預設參數（可用環境變數覆蓋）
CSV_PATH="${CSV_PATH:-resources/btc_15m.csv}"
MODELS_DIR="${MODELS_DIR:-models/BTCUSDT}"
FEATURE_FUNC="${FEATURE_FUNC:-csp.features.build_features}"
FEATURE_KWARGS="${FEATURE_KWARGS:-}"

HORIZON="${HORIZON:-16}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-30}"
TARGET_WIN="${TARGET_WIN:-0.70}"
TARGET_RET="${TARGET_RET:-0.10}"
GRID_MIN="${GRID_MIN:-0.50}"
GRID_MAX="${GRID_MAX:-0.75}"
GRID_STEP="${GRID_STEP:-0.01}"
FEES_BPS="${FEES_BPS:-6}"
SLIPPAGE_BPS="${SLIPPAGE_BPS:-1}"

echo "[START] $(date -u +%FT%TZ)" | tee -a "$LOG_FILE"

set -x
python -m scripts.ci_orchestrator \
  --csv "$CSV_PATH" \
  --models-dir "$MODELS_DIR" \
  --feature-func "$FEATURE_FUNC" \
  ${FEATURE_KWARGS:+ --feature-kwargs "$FEATURE_KWARGS"} \
  --horizon "$HORIZON" \
  --lookback-days "$LOOKBACK_DAYS" \
  --target-win "$TARGET_WIN" \
  --target-ret "$TARGET_RET" \
  --grid "$GRID_MIN" "$GRID_MAX" "$GRID_STEP" \
  --fees-bps "$FEES_BPS" \
  --slippage-bps "$SLIPPAGE_BPS" \
  2>&1 | tee -a "$LOG_FILE"
set +x

# 無論成功或失敗都送出最新回測結果
python -m scripts.notify_latest_backtest 2>&1 | tee -a "$LOG_FILE" || true

echo "[DONE]  $(date -u +%FT%TZ)" | tee -a "$LOG_FILE"
