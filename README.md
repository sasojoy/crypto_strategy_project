# Crypto Strategy Project — 使用說明（Windows / CMD 版）
> 版本日期：2025-08-15

本專案提供「**訓練** → **回測** → **即時訊號**」的一條龍流程，使用 15 分鐘 K 線資料建立模型，並在即時模式推送多幣別整合訊息。  
**重點更新**：支援「**日期區間**」功能（以本地 UTC+8 自然日解析），可精準指定訓練/初始化所用的歷史範圍。

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

risk:
  take_profit_ratio: 0.02
  stop_loss_ratio: 0.01
  max_holding_minutes: 240
  flip_threshold: 0.6

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

---

## 快速開始

### 訓練（多幣）
```cmd
python scripts\train_multi.py --cfg csp\configs\strategy.yaml
```
- 預設會從設定檔載入每個幣種的 CSV，並將模型輸出到 `io.models_dir`（每幣一個子資料夾）。

### 回測（多幣）
```cmd
python scripts\backtest_multi.py --cfg csp\configs\strategy.yaml --days 30 --fetch inc --save-summary
```
- `--days 30`：取最近 30 天資料。
- `--fetch inc`：只補缺口；用 `--fetch full` 可完整覆蓋重抓。

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
- **日期區間**：支援 `--start` / `--end`（如果你的 `ver14` 已套用該功能）或用環境變數（推薦）。

**範例（回測 90 天 + 出資金曲線）**
```cmd
python scripts\backtest_multi.py --cfg csp\configs\strategy.yaml --days 90 --fetch full ^
  --export-equity-bars outputs\equity_90d.csv --plot-equity outputs\equity_90d.png --save-summary
```

**範例（指定 2025/07 全月，以環境變數方式）**
```cmd
set START_DATE=2025-07-01
set END_DATE=2025-07-31
python scripts\backtest_multi.py --cfg csp\configs\strategy.yaml --save-summary
```

---

### 即時（`scripts\realtime_multi.py` & `scripts\realtime_loop.py`）
- **用途**：讀取最新資料並推送**一則整合的多幣別訊息**（時間格式 `YYYY-MM-DD HH:MM`，不帶 `+08:00`）。
- **常用參數**：
  - `--cfg <path>`：指定設定檔。
  - `--delay-sec <int>`（限 `realtime_loop.py`）：輪詢間隔秒數（預設 15 秒）。
- **日期區間（初始化 warmup）**：
  - 在初始化歷史（做特徵/狀態建立）時，可用 **環境變數** 限縮歷史區間，不影響之後的即時抓取。

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
- **回測摘要**：啟用 `--save-summary` 後輸出到預設的輸出目錄（依專案實作而定）。
- **資金曲線**：
  - `--export-equity-bars <csv>`：輸出欄位通常為 `timestamp,equity`
  - `--plot-equity <png>`：輸出資金曲線圖
- **日誌**：`io.logs_dir`（例如 `logs\`）
- **持倉狀態**：`io.position_file`（例如 `resources\current_position.yaml`）

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
