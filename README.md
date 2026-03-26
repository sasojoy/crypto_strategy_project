
# Crypto Strategy Project (v10 - Structured)

這是精簡並重構過的加密貨幣交易策略專案，使用 15 分鐘 K 線資料進行機器學習模型訓練、即時預測與回測分析。目標是開發分類模型預測未來短期漲跌機率並觸發交易訊號。

---

## 📁 資料夾說明

### `/pipeline/`
- `realtime_cls.py`：主即時預測與交易策略腳本，會根據模型結果發出訊號。
- `sync_to_sheets.py`：將交易紀錄同步至 Google Sheets。

### `/model/`
- `train_model_cls.py`：訓練分類模型（多 horizon + 多閾值），並儲存模型與 scaler。
- `xgb_cls_model.json`：訓練完成的 XGBoost 模型。
- `xgb_cls_scaler.joblib`：特徵標準化用的 Scikit-learn scaler。

### `/strategy/`
- `strategy_manager.py`：分類策略邏輯的進出場條件計算與交易決策核心。
- `regression.py`：過去回歸策略邏輯的保留模組。

### `/data/`
- `data_module.py`：資料讀取與時間補齊。
- `feature_engineering.py`：特徵處理與轉換。
- `btc_15m_data_360days_b.csv`：歷史 K 線資料。
- `position.json`：當前持倉記錄。

### `/core/`
- `feature.py`：共用特徵處理工具與 helper 函數。

### `/config/`
- `strategy_config.yaml`：策略參數設定。
- `joystrategy-btc-11536b5d09b1.json`：Google Sheets 授權憑證。

### `/util/`
- `notifier.py`：負責推播 Telegram 訊息與交易紀錄格式化。

### `/backtest/`
- `backtest_module.py`：執行策略回測與績效統計。

---

## 🚀 使用方式（常用指令）

### 訓練模型
```bash
python model/train_model_cls.py
```

### 即時預測與交易推播
```bash
python pipeline/realtime_cls.py
```

### 回測策略績效
```bash
python backtest/backtest_module.py
```

---

## ✅ 目標策略成果
- 使用分類模型預測短期報酬區間
- 動態多閾值決策
- 回測支援停利/停損、持倉時間控制
- 與 Google Sheets / Telegram 整合即時通知

---

請根據實際需求調整 `strategy_config.yaml` 參數，並確保模型已先訓練完成。

