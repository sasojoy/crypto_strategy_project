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

## 設定說明

- `execution.*`：進出場門檻與持有時間；`execution.atr.enabled=false` 可關閉以 ATR 為基礎的停利/停損（pt/sl）。
- `fetch.interval` 建議維持 `15m`，程式會正規化成 `15T`，避免 pandas 對非固定頻率的例外。
- `backtest.*`：定義初始資金、單邊手續費與預估滑點等參數。

## CI/CD（A 方案）

- CI 的模型訓練在 GitHub Runner 上執行，回測 /（可選）優化 / 部署則在 VM 上完成，回測一律使用 `--fetch inc` 以確保最新資料。
- 回測報告會打包成 Artifact；若績效門檻未達成則停止部署並透過 Telegram 通知失敗。
- 部署成功後，`trader-once.timer` 會每 15 分鐘觸發一次；系統服務使用 `/etc/crypto_strategy_project.env` 提供 Telegram 憑證。

## 故障排除

- **HTTP 451**：已將回測移至 VM；若仍遇到請檢查 VM 網路與 `fetch.base_url` 設定。
- **Telegram 沒通知**：確認 `/etc/crypto_strategy_project.env` 權限為 600，且 systemd 服務使用 `EnvironmentFile=` 載入。
- **KeyError: 'execution' / 'atr'**：確保 `csp/configs/strategy.yaml` 已含 `execution` 區塊，或依預設值執行。
- **pandas 非固定頻率**：程式已將 `15m` 正規化為 `15T`，若自訂頻率請確認為 pandas 支援的固定頻率。
