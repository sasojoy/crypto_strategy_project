# Crypto Strategy Project

以 15 分鐘 K 線為主的多幣別量化策略。專案涵蓋模型訓練、回測、即時訊號與 CI/CD 自動化部署，預設透過 Telegram 通知即時結果。

## Quick Start

```bash
# 訓練（多幣）
python scripts/train_multi.py --cfg csp/configs/strategy.yaml

# 回測（多幣，使用最新資料）
python scripts/backtest_multi.py \
  --cfg csp/configs/strategy.yaml \
  --days 30 --fetch inc \
  --save-summary --out-dir reports --format both
```

## 實盤

- `scripts/realtime_loop.py` 在每次輪詢時會先等待約 5 秒，再以 `align_15m` 將當前 UTC 時間對齊至最近的 15 分鐘開盤時間，以確保資料源已生成最新一根 K 線的開盤價。
- 推播前會檢查 `/tmp/realtime_state.json`，該狀態檔會記錄最後一次已推送的 `bar_open` 時間；若同一根 K 線已推播過會直接跳過，避免重複通知。
- 只要狀態檔仍存在，即便系統重啟或服務重新啟動，也不會對同一根 K 線再次發送訊息。

## 設定說明

- `execution.*`：進出場門檻與持有時間；`execution.atr.enabled=false` 可關閉以 ATR 為基礎的停利/停損（pt/sl）。
- `fetch.interval` 建議維持 `15m`，程式會正規化成 `15T`，避免 pandas 對非固定頻率的例外。
- `backtest.*`：定義初始資金、單邊手續費與預估滑點等參數。

## CI/CD（A 方案）

- CI 的模型訓練在 GitHub Runner 上執行，回測 /（可選）優化 / 部署則在 VM 上完成，回測一律使用 `--fetch inc` 以確保最新資料。
- 回測報告會打包成 Artifact；若績效門檻未達成則停止部署並透過 Telegram 通知失敗。
- 部署成功後，`trader-once.timer` 會每 15 分鐘觸發一次；系統服務使用 `/etc/crypto_strategy_project.env` 提供 Telegram 憑證。

## 模型自動化 CI（B 方案）

- **流程概覽**：GitHub Actions 觸發 `model-ci` workflow 後，`scripts/ci_orchestrator.py` 會先以 `scripts/train_h16_wf.py` 進行時間序交叉驗證訓練，再呼叫 `scripts/threshold_report.py` 回測門檻表現；若主要指標（預設 `roc_auc`）未達標，會自動展開小型參數搜尋（歷程紀錄於 `logs/ci_run.json`），最後依狀態透過 Telegram 通知。
- **觸發方式**：支援 push（`main`、`work`、`ci/**`）、Pull Request、排程（週一 03:00 UTC）以及 workflow_dispatch 手動觸發。
- **GitHub Secrets**：必須在專案設定中加入 `TELEGRAM_BOT_TOKEN` 以及 `TELEGRAM_CHAT_ID` 才能接收通知；若未設定將僅在 CI console 顯示結果。若 workflow 由 fork 對 upstream 發起 PR 觸發，GitHub 會阻擋 secrets，因此請改在本倉庫分支上執行或改用其他事件（例如 `pull_request_target`，須自行評估權限風險）。
- **環境需求**：請確保保留 `scripts/__init__.py`（供 `python -m scripts.ci_orchestrator` 匯入）並安裝 `requests` 套件；CI 成功或失敗都會透過 Telegram 推播結果。`scripts/ci_notify_from_log.py` 會在 workflow 結束時讀取 `logs/ci_run.json`，若有 `[SUMMARY ALL]` 開頭的行會原樣推送到 Telegram；否則會用預設格式（含 win_rate / total_return 百分比）通知結果。額外的 `scripts/notify_latest_backtest.py` 會原樣轉發 `[SUMMARY ALL] {...}`；若不存在則轉發最佳指標（`win_rate` / `total_return` / `threshold`）。
- **Secrets 設定**：可在 GitHub Repository → Settings → Secrets and variables → Actions 中新增 `TELEGRAM_BOT_TOKEN`（BotFather 發送）與 `TELEGRAM_CHAT_ID`（`@getidsbot` 或 `curl` 查詢）兩組 secrets。
- **產出物**：
  - `logs/ci_run.json`：詳細記錄各次訓練、回測、門檻掃描與參數搜尋狀態，供失敗時貼給 ChatGPT 進行問題排查。
  - `logs/threshold_report.json`：最佳門檻的覆蓋率、Precision/F1 與平均報酬。
  - `models/ci/**`：保存校正後的模型與 metadata（`model.joblib`、`metadata.json`、`thresholds.json`）。
  - `artifacts/ci/best_run.json`：最佳嘗試的摘要，方便下載檢視。
- **失敗排查**：若門檻未達標，workflow 仍會完成並將 `logs/ci_run.json` 上傳成 Artifact，請將該檔案貼給 ChatGPT 或研發群組討論後續調整策略/資料品質。

## Server 排程（每日 08:05 Asia/Taipei）

- **每日流程腳本**：`scripts/run_daily_pipeline.sh` 會處理防重複鎖、統一 log（輸出在 `logs/daily_*.log`）與 `.env` 的環境變數載入，並呼叫 `python -m scripts.ci_orchestrator` 完成訓練/回測、最後一律透過 `scripts.notify_latest_backtest` 發送最新結果。
- **systemd timer**（推薦）：`ops/systemd/model-ci.service` 與 `ops/systemd/model-ci.timer` 可部署至伺服器 `/etc/systemd/system/`，透過 `OnCalendar` 每日 08:05（Asia/Taipei）觸發。部署步驟：
  ```bash
  sudo cp ops/systemd/model-ci.* /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now model-ci.timer
  # 手動立即跑一次（可選）
  sudo systemctl start model-ci.service
  # 看狀態/下一次時間
  systemctl status model-ci.timer
  systemctl list-timers | grep model-ci
  ```
- **cron 替代方案**：若環境不便使用 systemd，可執行 `ops/cron/install_cron.sh` 自動將排程寫入 `crontab`（同樣為每日 08:05 Asia/Taipei），log 會輸出到 `logs/cron.log`。
- **環境變數設定**：可在 `/opt/strategy/.env`（或你自訂的 `ROOT_DIR`）配置 `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`、`CSV_PATH`、`MODELS_DIR`、`FEATURE_FUNC`、`FEATURE_KWARGS` 等設定，腳本會自動載入。

## CI/CD Deploy 後執行一次

- GitHub Actions 的 `model-ci` workflow 在訓練與回測成功後，會以 SSH（`appleboy/ssh-action`）連線到遠端伺服器（預設 `/opt/strategy`），執行 `git pull --rebase` 並立即觸發 `scripts/run_daily_pipeline.sh` 或相同流程的 systemd 服務。
- Secrets 需在 GitHub Repository 設定 `SSH_HOST`、`SSH_USER`、`SSH_KEY` 才能成功連線；若是 fork PR，GitHub 不會注入 secrets，請改用本倉庫分支或手動觸發（`workflow_dispatch` / `schedule`）。

## 通知

- `scripts.ci_orchestrator` 與 `scripts.notify_latest_backtest` 會在 pipeline 結束時讀取 `logs/ci_run.json`，若存在 `[SUMMARY ALL] {...}` 會優先推播；若沒有則會退回推送最佳指標（win_rate / total_return / threshold）至 Telegram。

## 故障排除

- **HTTP 451**：已將回測移至 VM；若仍遇到請檢查 VM 網路與 `fetch.base_url` 設定。
- **Telegram 沒通知**：確認 `/etc/crypto_strategy_project.env` 權限為 600，且 systemd 服務使用 `EnvironmentFile=` 載入。
- **KeyError: 'execution' / 'atr'**：確保 `csp/configs/strategy.yaml` 已含 `execution` 區塊，或依預設值執行。
- **pandas 非固定頻率**：程式已將 `15m` 正規化為 `15T`，若自訂頻率請確認為 pandas 支援的固定頻率。
