# Crypto Strategy Project — 使用說明（Windows / CMD 版）
> 版本日期：2025-08-24

本專案提供「**訓練** → **回測** → **即時訊號**」的一條龍流程，使用 15 分鐘 K 線資料建立模型，並在即時模式推送多幣別整合訊息。
**重點更新**：
- 支援「**日期區間**」功能（以本地 UTC+8 自然日解析），可精準指定訓練/初始化所用的歷史範圍。
- 回測可輸出完整績效摘要與交易明細，支援 CSV / JSON 報表。
- 新增 **訊號匯總器**（aggregator），整合多 horizon 機率並輸出統一的多/空/無決策。
- 擴充 **即時通知**（Telegram）模組，涵蓋最新訊號、進出場與風控事件。

---

## 目錄
- [環境需求](#環境需求)
- [安裝](#安裝)
- [資料與設定](#資料與設定)
- [快速開始](#快速開始)
- [指令與功能總覽](#指令與功能總覽)
  - [訓練（scripts\train_multi.py）](#訓練scriptstrain_multipy)
  - [回測（scripts\backtest_multi.py）](#回測scriptsbacktest_multipy)
  - [即時（scripts\realtime_multi.py / scripts\realtime_loop.py）](#即時scriptsrealtime_multipy--scriptsrealtime_looppy)
  - [特徵優化（scripts\feature_optimize.py）](#特徵優化scriptsfeature_optimizepy)
  - [即時出場監控（scripts\exit_watchdog.py）](#即時出場監控scriptsexit_watchdogpy)
- [日期區間功能說明](#日期區間功能說明)
- [輸出與檔案位置](#輸出與檔案位置)
- [常見問題 FAQ](#常見問題-faq)
- [驗證清單（建議）](#驗證清單建議)

---

## 環境需求
- Windows 10/11（CMD 或 PowerShell 皆可，以下以 **CMD** 為主）
- Python 3.10+（建議使用虛擬環境）
- 必要套件：`pandas`, `numpy`, `xgboost`（或你專案實際用到的 ML 套件）, `matplotlib`（若要輸出資金曲線 PNG）

---

## 安裝
```cmd
REM 建議先建立並啟用虛擬環境（可略）
python -m venv ml-env
ml-env\Scripts\activate

REM 安裝
pip install --upgrade pip setuptools wheel
pip install -e .
```

> 若遇到 `Multiple top-level packages`，請確認 `pyproject.toml` 僅包含你的套件前綴（例如 `csp*`）。

---

## 資料與設定

### 1) 設定檔 `csp\configs\strategy.yaml`
你可以在這裡定義幣種、CSV 路徑、模型/輸出目錄、是否即時抓幣安、紀錄檔等。範例節選：
```yaml
symbols: [BTCUSDT, ETHUSDT, BCHUSDT]

io:
  csv_paths:
    BTCUSDT: resources/btc_15m.csv
    ETHUSDT: resources/eth_15m.csv
    BCHUSDT: resources/bch_15m.csv
  models_dir: models
  logs_dir: logs
  position_file: resources/current_position.yaml

strategy:
  enter_threshold: 0.75
  aggregator_method: "max_weighted"    # or "majority"
  weight_fn: "sqrt"                    # or "log", "linear"

risk:
  take_profit_ratio: 0.05
  stop_loss_ratio: 0.02
  max_holding_minutes: 240
  flip_threshold: 0.6
position_sizing:
  mode: "hybrid"            # "atr" | "kelly" | "hybrid"
  risk_per_trade: 0.01      # 每筆風險占淨值比例
  atr_k: 1.5                # 估算風險距離所用 ATR 倍數
  kelly_coef: 0.5           # Kelly 權重（0~1）
  kelly_floor: -0.5         # 最低倍率（-0.5 -> 最多縮至 0.5x）
  kelly_cap: 1.0            # 最高加成（+100%）
  default_win_rate: 0.6
  exchange_rule:
    min_qty: 0.0001
    qty_step: 0.0001
    min_notional: 10
    max_leverage: 10

realtime:
  notify:
    telegram:
      enabled: true
      bot_token: "<YOUR_BOT_TOKEN>"
      chat_id: "<YOUR_CHAT_ID>"
```

### 2) 資料檔案（CSV）
- 以 15 分鐘 K 線為主，欄位需含 `timestamp`（可被 `pandas.to_datetime(..., utc=True)` 解析）、`open/high/low/close/volume` 等。
- 建議資料時間軸無重複且已排序。

### Timezones
- 整個流程（訓練 / 回測 / 即時）內部一律使用 **UTC**。
- 讀取 CSV 時會以 `utc=True` 解析時間戳並透過 `ensure_utc_index` 設成 UTC index。
- bar 對齊與 `now` 取得皆以 UTC 計算；如需顯示本地時間僅在最後轉換（例如 `Asia/Taipei`）。

---

## 快速開始

### 訓練（多幣）
```cmd
python scripts\train_multi.py --cfg csp\configs\strategy.yaml
```
- 預設會從設定檔載入每個幣種的 CSV，並將模型輸出到 `io.models_dir`（每幣一個子資料夾）。
- 所有腳本會透過 `csp.utils.io.load_cfg` 讀取設定，無論傳入的是檔案路徑或已解析的 `dict`。

### 回測（多幣）
```cmd
python scripts\backtest_multi.py --cfg csp\configs\strategy.yaml --days 30 --fetch inc --save-summary --out-dir reports --format both
```
- `--days 30`：取最近 30 天資料。
- `--fetch inc`：只補缺口；用 `--fetch full` 可完整覆蓋重抓。
- `--out-dir reports`：報表輸出目錄（預設 `reports`）。
- `--format both`：報表格式，可選 `csv`、`json` 或 `both`。

## 訓練（多幣）＋可選超參優化（A 方案）
```bash
# 基本訓練（不啟用優化）
python scripts/train_multi.py --cfg csp/configs/strategy.yaml \
  --symbols BTCUSDT,ETHUSDT,BCHUSDT --out-dir models

# 啟用優化（開關不改演算法；內部重用現有 Optuna 流程）
CSP_OPTIMIZE=true CSP_OPT_N_TRIALS=60 CSP_OPT_TIMEOUT_MIN=25 \
python scripts/train_multi.py --cfg csp/configs/strategy.yaml \
  --symbols BTCUSDT,ETHUSDT,BCHUSDT --out-dir models --optimize
```

回測（多幣）— 預設會補到最新收盤K

```bash
# 預設 --fetch inc（補缺口），確保回測用到最新資料（若遇 451 會跳過抓取但仍完成回測）
python scripts/backtest_multi.py --cfg csp/configs/strategy.yaml \
  --days 30 --fetch inc --save-summary --out-dir reports --format both
```

- 回測視窗上限會對齊最後一根收盤K（例如 15m 週期 → 對齊 :00, :15, :30, :45）。
- 若 CSV 已是最新，增量抓取不會多做事；若 runner 被交易所封鎖 (HTTP 451)，會警告但不中斷。

CI 流程（Train → Backtest → Deploy）

- Train：可用 `CSP_OPTIMIZE=true` 開啟優化，否則只做基本訓練。
- Backtest：固定 `--fetch inc`，永遠嘗試補到最新收盤K。
- Deploy：沿用現行 SSH/systemd 流程；Telegram SMOKE 確認成功。

### 即時訊號
```cmd
python scripts\realtime_multi.py --cfg csp\configs\strategy.yaml
```
或長駐迴圈（每 15 秒）
```cmd
python scripts\realtime_loop.py --cfg csp\configs\strategy.yaml --delay-sec 15
```

---

## 指令與功能總覽

### 訓練（`scripts\train_multi.py`）
- **用途**：根據 `strategy.yaml` 的幣種列表，逐幣讀取 CSV 並呼叫模型訓練（如 `csp\models\train_h16_dynamic.py`）。
- **常用參數**：
  - `--cfg <path>`：指定設定檔（必填）。
- **日期區間**（不改 CLI 也能用）  
  使用 **環境變數** 即可（見下方「日期區間功能說明」）。

**範例（指定 2025/07 全月）**
```cmd
set START_DATE=2025-07-01
set END_DATE=2025-07-31
python scripts\train_multi.py --cfg csp\configs\strategy.yaml
```

**範例（指定最近 180 天；只在未設 start/end 時生效）**
```cmd
set DAYS=180
python scripts\train_multi.py --cfg csp\configs\strategy.yaml
```

---

### 回測（`scripts\backtest_multi.py`）
- **用途**：多幣同時回測，支援自動補檔與資金曲線輸出。
- **常用參數**：
  - `--cfg <path>`：指定設定檔。
  - `--days <N>`：從最新往回 N 天（當未指定 `--start/--end` 時）。
  - `--fetch inc|full`：`inc` 只補缺口；`full` 重新抓取覆蓋。
  - `--export-equity-bars <path.csv>`：輸出等時間柱狀的資金曲線。
  - `--plot-equity <path.png>`：輸出資金曲線圖（需要 `matplotlib`）。
  - `--save-summary`：輸出回測摘要（含 win_rate、profit_factor、signal_count、avg_holding_minutes…）。
  - `--out-dir <dir>`：指定報表輸出目錄（預設 `reports`）。
  - `--format csv|json|both`：報表輸出格式（預設 `both`）。
- **日期區間**：支援 `--start` / `--end`（如果你的 `ver14` 已套用該功能）或用環境變數（推薦）。

**範例（回測 90 天 + 出資金曲線）**
```cmd
python scripts\backtest_multi.py --cfg csp\configs\strategy.yaml --days 90 --fetch full ^
  --export-equity-bars outputs\equity_90d.csv --plot-equity outputs\equity_90d.png --save-summary --out-dir reports --format both
```

**範例（指定 2025/07 全月，以環境變數方式）**
```cmd
set START_DATE=2025-07-01
set END_DATE=2025-07-31
python scripts\backtest_multi.py --cfg csp\configs\strategy.yaml --save-summary --out-dir reports --format both
```

---

### 即時（`scripts\realtime_multi.py` & `scripts\realtime_loop.py`）
- **用途**：讀取最新資料並推送**一則整合的多幣別訊息**（時間格式 `YYYY-MM-DD HH:MM`，不帶 `+08:00`）。
- **常用參數**：
  - `--cfg <path>`：指定設定檔。
  - `--delay-sec <int>`（限 `realtime_loop.py`）：輪詢間隔秒數（預設 15 秒）。
- 內建訊號匯總器：將多 horizon 機率整合為單一 LONG/SHORT/NONE 訊號，可透過 `strategy.enter_threshold`、`strategy.aggregator_method`、`strategy.weight_fn` 調整。
- 支援 Telegram 即時通知：會針對「最新訊號」、「進場/出場」與「風控事件」發送標準欄位訊息。
- **日期區間（初始化 warmup）**：
  - 在初始化歷史（做特徵/狀態建立）時，可用 **環境變數** 限縮歷史區間，不影響之後的即時抓取。

#### 故障排除
- `NONE | score=0.00 (reason=no_models_loaded)`：找不到模型檔或檔案不完整。
- `NONE | score=0.00 (reason=feature_nan)`：最新一根特徵含 NaN，修補失敗。

### systemd 範例
本專案提供 `systemd/trader.service` 與 `systemd/trader.timer` 兩個範例檔案。
- `trader.service` 在啟動前會 `ExecStartPre=/bin/sleep 15`，避免過早讀取未收盤的資料，實際執行腳本 `run_realtime.sh`。
- `trader.timer` 以 `OnCalendar=*:0/15` 每 15 分鐘觸發，並設定 `Persistent=true` 與 `AccuracySec=1s`。

安裝方式：
```bash
sudo cp systemd/trader.service /etc/systemd/system/
sudo cp systemd/trader.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trader.timer
```
- `NONE | score=0.00 (reason=stale_data)`：最新 K 線落後目前時間超過 15 分鐘。
- `NONE | score=0.00 (reason=empty_or_invalid_inputs)`：匯總器收到空或全無效的輸入。

**範例（warmup 僅用 8/1~8/10 的歷史）**
```cmd
set START_DATE=2025-08-01
set END_DATE=2025-08-10
python scripts\realtime_multi.py --cfg csp\configs\strategy.yaml
```

**例：長駐模式**
```cmd
set START_DATE=2025-08-01
set END_DATE=2025-08-10
python scripts\realtime_loop.py --cfg csp\configs\strategy.yaml --delay-sec 15
```

---

### 即時出場監控（`scripts\exit_watchdog.py`）
- **用途**：循環檢查持倉，達到 TP/SL/時間/翻向 即平倉。
- **常用參數**：
  - `--cfg <path>`：指定設定檔。
  - `--interval-sec <int>`：輪詢間隔秒數。
  - `--dry-run`：僅模擬，不寫檔或通知。
- 觸發平倉時會推送 Telegram 通知，內容含理由、PnL% 與持倉時間。
- **範例**：
```cmd
python scripts\exit_watchdog.py --cfg csp\configs\strategy.yaml --interval-sec 1 --dry-run
```

---

### 特徵優化（`scripts\feature_optimize.py`）
- **用途**：利用 Optuna 搜索特徵參數，逐幣回測評估。
- **常用參數**：
  - `--cfg <path>`：指定設定檔。
  - `--symbols sym1,sym2`：以逗號分隔的幣別清單（預設讀 cfg.symbols）。
  - `--days <N>`：回測天數（或配合環境變數 START_DATE/END_DATE）。
  - `--trials <N>`：Optuna 試驗次數。
  - `--apply-to-cfg`：將最佳參數寫回 `strategy.yaml`。
- **範例**：
```cmd
python scripts\feature_optimize.py --cfg csp\configs\strategy.yaml --symbols BTCUSDT,ETHUSDT --days 30 --trials 10 --apply-to-cfg
```

---

## 日期區間功能說明

- **輸入方式**（三選一，互斥優先順序如下）：  
  1) **環境變數**（建議；最無侵入）：`START_DATE` / `END_DATE` / `DAYS`  
  2) **程式呼叫參數**：`train(..., start='YYYY-MM-DD', end='YYYY-MM-DD', days=...)` 或 `date_args={...}`  
  3) （可選）CLI 旗標（若你已整合）

- **優先順序**：`start/end` ＞ `days` ＞ 預設（全量或 既有 `--days` 預設值）。
- **時區規則**：`start`/`end` 以 **UTC+8 自然日** 解析，內部統一轉為 **UTC** 切片；`end` 為包含式上界（含當日 23:59:59.999999）。
- **訓練 warmup 緩衝**：會依最大指標視窗與最大 horizon **自動向前回推** 若干分鐘，以避免在區間起頭的技術指標與標籤失真。你可在 `csp\models\train_h16_dynamic.py` 調整：
  ```python
  INDICATOR_MAX_WINDOW = 200
  HORIZON_MAX = 192
  BAR_MINUTES = 15
  SAFETY_BARS = 10
  ```

---

## 輸出與檔案位置
- **模型**：`io.models_dir`（按幣種分目錄，如 `models\BTCUSDT`）
- **回測摘要**：啟用 `--save-summary` 後輸出到 `--out-dir`（預設 `reports`），結構如下：
  - `reports/{run_id}/summary_{symbol}.json`
  - `reports/{run_id}/summary_{symbol}.csv`
  - `reports/{run_id}/trades_{symbol}.csv`
  - `reports/{run_id}/summary_all.json`
- **資金曲線**：
  - `--export-equity-bars <csv>`：輸出欄位通常為 `timestamp,equity`
  - `--plot-equity <png>`：輸出資金曲線圖
- **日誌**：`io.logs_dir`（例如 `logs\`）
- **持倉狀態**：`io.position_file`（例如 `resources\current_position.yaml`）

## 模型輸出結構與載入方式
- 多 horizon × 多門檻訓練會在輸出目錄產生 `model_{h}_{t}.pkl` 與可選的 `cal_{h}_{t}.pkl`（校準器），以及 `meta.json`。
- `meta.json` 包含訓練時使用的 `horizons`、`thresholds`、`feature_columns` 等資訊。
- 載入範例：

```python
from csp.models.classifier_multi import MultiThresholdClassifier
m = MultiThresholdClassifier.load("models/BTCUSDT/cls_multi")
probs = m.predict_proba(df_features.tail(1))
```

## 🔍 CI Debugging with Diagnostic Logs

To investigate why signals may show `NONE | score=nan`, the CI workflow now includes a step that runs:

```bash
python scripts/predict_and_notify.py --cfg csp/configs/strategy.yaml
```

During CI runs, you can check the GitHub Actions logs for lines starting with `[DIAG]`, which provide:

- Feature NaN counts and values of the last row
- Model type and availability of `predict_proba`
- Probability outputs from the model
- Detailed reason if `score=nan` (e.g., missing features)

These logs are for debugging only and do not affect trading logic.


## CI Train & Deploy

GitHub Actions workflow [`deploy.yml`](.github/workflows/deploy.yml) automates model
training and deployment:

1. **train** job installs dependencies and runs
   `python scripts/train_multi.py --cfg csp/configs/strategy.yaml` to generate
   models under `models/<SYMBOL>/` for BTCUSDT, ETHUSDT and BCHUSDT. The folder
   is uploaded as an artifact.
2. **deploy** job downloads the artifact, rsyncs the repository and models to the
   VM at `/opt/crypto_strategy_project`, installs systemd unit files, reloads
   systemd and triggers one run of `trader-once.service`.

Required GitHub secrets:

- `SSH_USER`
- `SSH_HOST`
- `SSH_KEY`

## 遠端 VM 健康檢查

若要在 GitHub Actions 或其他 CI 流程中檢查部署於遠端 VM 的 `trader-once` 服務，可使用以下 SSH 指令。請先於環境變數或 secrets 中設定 `SSH_USER` 與 `SSH_HOST`：

```bash
ssh -o StrictHostKeyChecking=yes $SSH_USER@$SSH_HOST 'bash -lc "
  set -euo pipefail
  cd /opt/crypto_strategy_project

  sudo systemctl status trader-once.timer --no-pager
  sudo systemctl status trader-once.service --no-pager
  sudo journalctl -u trader-once.service -n 200 -o cat
"'
```

上述指令會顯示 `trader-once` 的 systemd 服務狀態以及最近 200 行日誌。僅能在使用 systemd 的 Linux 主機上執行；若於未啟動 systemd 的環境（如純容器）執行，將會出現 `System has not been booted with systemd` 的訊息。


---

## 常見問題 FAQ

**Q1：設了日期但沒資料？**  
A：請確認日期是否落在 CSV 的資料期間內；也確認 `timestamp` 欄位能被 `pandas.to_datetime(..., utc=True)` 正確解析。

**Q2：看到錯誤 `Cannot pass a datetime or Timestamp with tzinfo with the tz parameter`？**  
A：避免同時使用 `utc=True` 與 `.tz_localize()`；若 index 已帶時區請改用 `.tz_convert("UTC")`。

**Q3：即時訊息沒有合併？**  
A：請使用 `scripts\realtime_multi.py` 或 `realtime_loop.py`（該版本會合併成單則多幣別訊息；時間格式 `YYYY-MM-DD HH:MM`）。

**Q4：如何比對被動持有？**  
A：回測可另外加入基準輸出（未內建時可自行擴充 `--equity-benchmark`，或於分析階段用 CSV 合併）。

---

## 驗證清單（建議）
- [ ] 訓練與回測可在 **未設定任何日期** 時正常跑完（回溯到預設天數或全量）。
- [ ] 設定 `START_DATE`/`END_DATE` 後，訓練/回測/即時初始化可跑完，並且結果範圍正確。
- [ ] `--fetch inc`/`--fetch full` 的補檔流程可正確更新資料，再進行日期切片。
- [ ] 回測可輸出資金曲線 CSV 與 PNG，最後一筆與回測最終 equity 相符。
- [ ] 即時訊息合併、中文摘要與其他原有功能不受影響。

## CI：Train & Deploy（含非阻塞 Backtest 報表）

本專案的 GitHub Actions workflow：**每次 push 到 `main`**（或手動觸發）會進行 **訓練 →（並行）回測 / 部署**。

流程拓撲：

```
push → train ──→ deploy
        │
        └────→ backtest（不阻擋 deploy）
```

### jobs.train（訓練）
- 多幣訓練指令：
  ```bash
  python scripts/train_multi.py \
    --symbols "BTCUSDT,ETHUSDT,BCHUSDT" \
    --out-dir models \
    --cfg csp/configs/strategy.yaml
  ```
- 產物：`models/`（每幣一個子資料夾 + `manifest.json`），上傳 artifact：**`trained-models`**。
- 備註：也可只帶 `--cfg`，由設定檔的 `io.csv_paths` 與 `io.models_dir` 驅動。

### jobs.backtest（回測，**非阻塞**）
- 依賴 `train`，下載 `trained-models` 到 `models/`，以**確保與部署同一版模型**。
- 指令：
  ```bash
  python scripts/backtest_multi.py \
    --cfg csp/configs/strategy.yaml \
    --days 30 --fetch inc \
    --save-summary --out-dir reports --format both
  ```
- 設定 `continue-on-error: true` 且上傳報表 `if: always()`，**即使回測失敗也不會卡住部署**。
- 產物：`reports/` 上傳 artifact：**`backtest-reports`**（預設保存 7 天）。

#### 下載回測報表
1. GitHub → **Actions** → 找到對應的 **Train & Deploy** 執行紀錄  
2. 在 **Artifacts** 區塊下載 **`backtest-reports`**（zip），內含 CSV/JSON 總結與各幣別回測結果。

### jobs.deploy（部署）
- 依賴 `train`（**不依賴 backtest**）
- 步驟摘要：
  - 下載 `trained-models` → 與程式碼一起 `rsync` 至 VM（`/opt/crypto_strategy_project`）
  - 建立/覆寫 `/etc/crypto_strategy_project.env`（寫入 TELEGRAM secrets）
  - 安裝 systemd 服務與排程：
    - `trader-once.service`：單次執行驗證
    - `trader-once.timer`：**每 15 分鐘 + 15 秒** 觸發
  - 觸發一次 `service`（驗證本次 build 可跑）
  - 送一則 Telegram **smoke** 訊息（只檢查 HTTP 狀態；不含敏感資訊）

部署後檢查（在 VM 上）：
```bash
sudo systemctl status trader-once.timer --no-pager
sudo systemctl status trader-once.service --no-pager
sudo journalctl -u trader-once.service -n 200 -o cat
sudo systemctl list-timers --all | grep -i trader-once
```

---

## 本機等效指令

### 訓練（多幣）
```bash
# 方式 A：完全走設定檔
python scripts/train_multi.py --cfg csp/configs/strategy.yaml

# 方式 B：顯式指定幣種與輸出
python scripts/train_multi.py \
  --symbols "BTCUSDT,ETHUSDT,BCHUSDT" \
  --out-dir models \
  --cfg csp/configs/strategy.yaml
```

### 回測（多幣）
```bash
python scripts/backtest_multi.py \
  --cfg csp/configs/strategy.yaml \
  --days 30 --fetch inc \
  --save-summary --out-dir reports --format both
```

---

## Secrets 與通知

部署需要以下 GitHub Secrets（由 CI 傳到 VM 並寫入 `/etc/crypto_strategy_project.env`）：
- `SSH_USER`、`SSH_HOST`、`SSH_KEY`：連線與同步程式碼
- `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`：用於訊息通知

在 `strategy.yaml` 中控制是否啟用 Telegram 通知：
```yaml
runtime:
  notify:
    telegram: true
```

若部署後 Telegram 沒收到通知，請檢查：
1. `/etc/crypto_strategy_project.env` 權限與內容（root:root, 0600；應包含 `TELEGRAM_BOT_TOKEN=` 與 `TELEGRAM_CHAT_ID=`）。
2. 服務日誌是否出現 `notify: telegram disabled or no config`（代表環境變數/設定未生效）。

---

## 訊號訊息格式（範例與欄位）

```
⏱️ 多幣別即時訊號 (build=<git_sha8>, host=<hostname>)
BTCUSDT: LONG | score=0.83 | h=+4h | pt=+2.1% | ↑=0.61 ↓=0.23 | price=115,881.98 | reason=cross/ema
ETHUSDT: NONE | score=0.49 | h=-    | pt=-     | ↑=-    ↓=-    | price=4,660.00   | reason=-
BCHUSDT: SHORT| score=0.21 | h=+2h  | pt=-1.0% | ↑=0.18 ↓=0.55 | price=598.10     | reason=rsi-div
```

欄位說明：
- `side`：建議方向（`LONG` / `SHORT` / `NONE`）
- `score`：模型信心分數（0–1）
- `h`：模型挑選的觀察/持有時間視窗（若無則 `-`）
- `pt`：目標幅度估計（若無則 `-`）
- `↑ / ↓`：上/下行機率估計（若無則 `-`）
- `price`：訊號當下現價
- `reason`：簡短解釋（資料不足則 `-`）

> **進出場規則**以策略程式邏輯為準；`side=NONE` 表示不建議動作。  
> 部署的 `trade.mode=signal_only` 僅發通知、不會自動下單。

---

## 常見問題（FAQ）

**Q：每次上版都會重算模型與回測嗎？**  
**A：會。** `train` job 每次 push 到 `main` 都會重訓並產出 `trained-models`。`backtest` 用同一套模型做 30 天回測，但為**非阻塞**，不影響部署。

**Q：想調整回測天數或關閉回測？**  
A：修改 `.github/workflows/train-deploy.yml` 的 `backtest` job 參數（如 `--days 30`），或暫時註解整個 `backtest` job。部署只依賴 `train`。

**Q：部署後要看下一次排程什麼時候跑？**  
A：`systemctl list-timers --all | grep -i trader-once` 可看到下一次觸發時間。

**Q：看到 `notify: telegram disabled or no config`？**  
A：表示程式沒拿到 `TELEGRAM_BOT_TOKEN/CHAT_ID` 或 `strategy.yaml` 關閉了 telegram。請依「Secrets 與通知」一節檢查。

---

## 工作流狀態徽章（可選）

將 `<your_org_or_user>` 與 `<your_repo>` 換成實際路徑：

```md
![Train & Deploy](https://github.com/<your_org_or_user>/<your_repo>/actions/workflows/train-deploy.yml/badge.svg)
```

