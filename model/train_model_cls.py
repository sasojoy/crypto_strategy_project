import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

# 若環境有 xgboost 就用，沒有就 fallback HGB
try:
    import xgboost as xgb
    _XGB_OK = True
except Exception:
    _XGB_OK = False

from data.data_module import load_data_from_file
from data.feature_engineering import add_indicators  # 你提供的版本（含 core.add_features）

# ===================== 路徑與常數 =====================
DATA_PATH  = "data/btc_15m_data_360days.csv"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_cls_model.json" if _XGB_OK else "hgb_cls_model.pkl")
SCALER_PATH= os.path.join(MODEL_DIR, "xgb_cls_scaler.joblib")
META_PATH  = os.path.join(MODEL_DIR, "xgb_cls_meta.yaml")
os.makedirs(MODEL_DIR, exist_ok=True)

RESAMPLE_4H = "4h"          # 4 小時（16 根 15m）
BARS_AHEAD  = 16            # 4 小時 horizon
LABEL_MODE  = "atr_event"       # ← 可選: "tplus" 或 "atr_event"

# 門檻掃描與目標
THRESH_GRID = np.linspace(0.50, 0.90, 41)  # 0.50 ~ 0.90 每 0.01
TARGET_PRECISION  = 0.70
TARGET_TRADES_30D = 60

# ====== 與線上相同的動態倍數參數（供 atr_event 標籤用） ======
BASE_SL_LONG  = 1.75
BASE_TP_LONG  = 3.50
BASE_SL_SHORT = 1.50
BASE_TP_SHORT = 3.00

ATR_N             = 14
ATR_LOOKBACK_PCTL = 96
ATR_PCTL_LOW      = 0.30
ATR_PCTL_HIGH     = 0.70
ADX_N             = 14
ADX_TREND         = 20
RANGING_ADX       = 12

EMA_GAP_STRONG = 0.0025
EMA_GAP_RANGE  = 0.0015
EMA_SAME_SIDE_K = 4

# ===================== 4h 重採樣 & 多週期特徵 =====================
def _resample_ohlcv(df_15m: pd.DataFrame, rule=RESAMPLE_4H) -> pd.DataFrame:
    df = df_15m.copy().sort_values("timestamp").set_index("timestamp")
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum(min_count=1)
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    return out.dropna().reset_index()

def _add_derived_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    d = df.copy()
    for col in ("open","high","low","close","volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()

    # 需要的欄位（add_indicators + core.add_features 會產）
    for k in ["close","ema_fast","ema_slow","rsi","bb_high","bb_low","macd_diff","EMA50"]:
        if k not in d.columns:
            d[k] = np.nan

    d["ema_gap_pct"] = (d["ema_fast"] - d["ema_slow"]) / d["close"]
    width = (d["bb_high"] - d["bb_low"]).replace(0, np.nan)
    d["band_pct"] = (d["close"] - d["bb_low"]) / width
    d["rsi_dev"] = (d["rsi"] - 50.0) / 50.0
    d["macd_norm"] = d["macd_diff"] / d["close"]
    d["price_above_ema"] = (d["close"] - d["EMA50"]) / d["close"]

    if prefix:
        rename = {c: f"{prefix}{c}" for c in ["ema_gap_pct","band_pct","rsi_dev","macd_norm","price_above_ema"] if c in d.columns}
        d = d.rename(columns=rename)
    return d

def _select_feature_columns(df: pd.DataFrame) -> list:
    drop = set(["timestamp","open","high","low","close","volume","return_next","label"])
    cols = [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]
    return cols

def build_multiframe_features(df_15m: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    # 15m
    f15 = add_indicators(df_15m.copy())
    f15 = _add_derived_features(f15).dropna().reset_index(drop=True)
    # 4h
    f4h_raw = _resample_ohlcv(df_15m)
    f4h = add_indicators(f4h_raw)
    f4h = _add_derived_features(f4h, prefix="h4_").dropna().reset_index(drop=True)
    # 對齊 4h → 15m
    h4_cols = [c for c in f4h.columns if c not in ["timestamp","open","high","low","close","volume"]]
    f4h_pref = f4h[["timestamp"] + h4_cols].copy()
    merged = pd.merge_asof(
        f15.sort_values("timestamp"),
        f4h_pref.sort_values("timestamp"),
        on="timestamp", direction="backward"
    ).dropna().reset_index(drop=True)

    feature_cols = _select_feature_columns(merged)
    return merged, feature_cols

# ===================== 市況與 ATR（供 atr_event 標籤用） =====================
def calc_atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    df = df.copy()
    for col in ("high","low","close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["high","low","close"])
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    return atr.bfill().ffill()

def _rma(series_like, n):
    x = pd.Series(series_like).fillna(0.0)
    alpha = 1.0 / n
    y = np.zeros(len(x))
    y[0] = x.iloc[:n].mean() if len(x) >= n else x.iloc[0]
    for i in range(1, len(x)):
        y[i] = y[i-1] + alpha * (x.iloc[i] - y[i-1])
    return pd.Series(y, index=x.index)

def compute_adx(df: pd.DataFrame, n: int = ADX_N) -> pd.Series:
    h = pd.to_numeric(df["high"], errors="coerce").ffill()
    l = pd.to_numeric(df["low"], errors="coerce").ffill()
    c = pd.to_numeric(df["close"], errors="coerce").ffill()
    up_move   = np.r_[np.nan, np.diff(h)]
    down_move = -np.r_[np.nan, np.diff(l)]
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),  up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum.reduce([
        h.values - l.values,
        np.abs(h.values - np.r_[np.nan, c.values[:-1]]),
        np.abs(l.values - np.r_[np.nan, c.values[:-1]])
    ])
    tr_rma      = _rma(tr, n).replace(0, np.nan)
    plus_dm_rma = _rma(plus_dm, n)
    minus_dm_rma= _rma(minus_dm, n)
    plus_di  = 100.0 * (plus_dm_rma / tr_rma)
    minus_di = 100.0 * (minus_dm_rma / tr_rma)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    adx = _rma(dx, n).replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0.0)
    return adx

def market_state_from_past(dslice: pd.DataFrame) -> dict:
    ds = dslice.copy()
    for col in ("high","low","close"):
        ds[col] = pd.to_numeric(ds[col], errors="coerce")
    ds = ds.dropna(subset=["high","low","close"])
    if len(ds) < max(ATR_N, 5):
        return dict(atr_now=np.nan, adx_now=0.0, high_vol=False, low_vol=False,
                    strong_trend=False, ranging=True)
    ds["ATR"] = calc_atr(ds, ATR_N)
    ds["EMA20"] = ds["close"].ewm(span=20, adjust=False).mean()
    ds["EMA50"] = ds["close"].ewm(span=50, adjust=False).mean()
    ds["ADX"] = compute_adx(ds, ADX_N)

    atr_now = float(ds["ATR"].iloc[-1])
    adx_now = float(ds["ADX"].iloc[-1]) if np.isfinite(ds["ADX"].iloc[-1]) else 0.0

    # EMA gap 混合規則
    price_now = float(ds["close"].iloc[-1])
    ema20_now = float(ds["EMA20"].iloc[-1]); ema50_now = float(ds["EMA50"].iloc[-1])
    ema_gap_pct = abs(ema20_now - ema50_now) / price_now if price_now > 0 else 0.0
    ema_side = (ds["EMA20"] > ds["EMA50"]).astype(int) - (ds["EMA20"] < ds["EMA50"]).astype(int)
    same_side_lastk = int(abs(ema_side.tail(EMA_SAME_SIDE_K).sum())) == EMA_SAME_SIDE_K

    adx_strong = adx_now >= ADX_TREND
    adx_range  = adx_now <= RANGING_ADX
    strong_trend = bool(adx_strong or ((ema_gap_pct >= EMA_GAP_STRONG) and same_side_lastk))
    ranging      = bool(adx_range  or  (ema_gap_pct <  EMA_GAP_RANGE))

    return dict(atr_now=atr_now, adx_now=adx_now,
                strong_trend=strong_trend, ranging=ranging,
                high_vol=False, low_vol=False)

def choose_multipliers(side: str, state: dict):
    if side == "LONG":
        sl_mult = BASE_SL_LONG; tp_mult = BASE_TP_LONG
    else:
        sl_mult = BASE_SL_SHORT; tp_mult = BASE_TP_SHORT
    if state.get("strong_trend", False):
        tp_mult *= 1.30
    elif state.get("ranging", False):
        sl_mult *= 0.90; tp_mult *= 0.70
    sl_mult = float(np.clip(sl_mult, 0.8, 3.0))
    tp_mult = float(np.clip(tp_mult, 1.2, 5.0))
    return sl_mult, tp_mult

# ===================== 標籤（兩種模式） =====================
def make_label_tplus(df: pd.DataFrame, bars_ahead: int = BARS_AHEAD) -> pd.DataFrame:
    df = df.copy()
    df["return_next"] = df["close"].shift(-bars_ahead) / df["close"] - 1
    df["label"] = (df["return_next"] > 0).astype(int)
    return df

def make_label_atr_event(df: pd.DataFrame, bars_ahead: int = BARS_AHEAD) -> pd.DataFrame:
    """
    快速版：預先計算所有狀態(ATR/EMA/ADX/強趨勢/盤整)，然後用 vectorized 方式判定未來16根內誰先觸發。
    同根同時觸發 → 視為不決（丟棄）。
    """
    out = df.copy().reset_index(drop=True)

    # ===== 1) 預先計算狀態（一次算完）=====
    close = pd.to_numeric(out["close"], errors="coerce")
    high  = pd.to_numeric(out["high"],  errors="coerce")
    low   = pd.to_numeric(out["low"],   errors="coerce")

    # ATR (RMA/簡化) —— 用 rolling 近似：與線上同參數 ATR_N
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N, min_periods=1).mean().bfill().ffill()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema_gap_pct = (ema20 - ema50) / close.replace(0, np.nan)

    # ADX 簡化近似：用過去 ADX_N 計算一次，不在迴圈裡重算
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),  up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_roll   = tr.rolling(ADX_N, min_periods=1).mean().replace(0, np.nan)
    plus_di   = 100.0 * (pd.Series(plus_dm).rolling(ADX_N, min_periods=1).mean() / tr_roll)
    minus_di  = 100.0 * (pd.Series(minus_dm).rolling(ADX_N, min_periods=1).mean() / tr_roll)
    dx        = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    adx       = dx.ewm(alpha=1/ADX_N, adjust=False).mean().fillna(0.0)

    # 強趨勢/盤整旗標（與線上規則一致）
    ema_side = (ema20 > ema50).astype(int) - (ema20 < ema50).astype(int)
    same_side_k = ema_side.rolling(EMA_SAME_SIDE_K).sum().abs().eq(EMA_SAME_SIDE_K)
    strong_trend = (adx >= ADX_TREND) | ((ema_gap_pct.abs() >= EMA_GAP_STRONG) & same_side_k.fillna(False))
    ranging      = (adx <= RANGING_ADX) | (ema_gap_pct.abs() < EMA_GAP_RANGE)

    # 倍數（每根一組，用向量運算）
    sl_mult_long  = np.clip(np.where(ranging, 0.90, 1.0) * BASE_SL_LONG, 0.8, 3.0)
    tp_mult_long  = np.clip(np.where(ranging, 0.70, 1.0) * np.where(strong_trend, 1.30, 1.0) * BASE_TP_LONG, 1.2, 5.0)
    sl_mult_short = np.clip(np.where(ranging, 0.90, 1.0) * BASE_SL_SHORT, 0.8, 3.0)
    tp_mult_short = np.clip(np.where(ranging, 0.70, 1.0) * np.where(strong_trend, 1.30, 1.0) * BASE_TP_SHORT, 1.2, 5.0)

    # ===== 2) 計算每根的 TP/SL 價位（兩個方向）=====
    slL = close - sl_mult_long  * atr
    tpL = close + tp_mult_long  * atr
    slS = close + sl_mult_short * atr
    tpS = close - tp_mult_short * atr

    # ===== 3) 用 window=16 的滑動窗口找「第一個命中」=====
    n = len(out)
    label = np.full(n, np.nan)

    H = high.to_numpy()
    L = low.to_numpy()
    slL_np = slL.to_numpy(); tpL_np = tpL.to_numpy()
    slS_np = slS.to_numpy(); tpS_np = tpS.to_numpy()

    W = bars_ahead
    for i in range(n - W):
        # 取未來視窗
        Hwin = H[i+1:i+1+W]
        Lwin = L[i+1:i+1+W]

        # 找最早命中的 index（沒有命中則為 None）
        idx_L_tp = np.argmax(Hwin >= tpL_np[i]) if np.any(Hwin >= tpL_np[i]) else -1
        idx_L_sl = np.argmax(Lwin <= slL_np[i]) if np.any(Lwin <= slL_np[i]) else -1
        idx_S_tp = np.argmax(Lwin <= tpS_np[i]) if np.any(Lwin <= tpS_np[i]) else -1
        idx_S_sl = np.argmax(Hwin >= slS_np[i]) if np.any(Hwin >= slS_np[i]) else -1

        # 轉為「事件發生的步數」（-1 → None）
        cand = []
        if idx_L_tp != -1: cand.append(("LONG","TP", idx_L_tp))
        if idx_L_sl != -1: cand.append(("LONG","SL", idx_L_sl))
        if idx_S_tp != -1: cand.append(("SHORT","TP", idx_S_tp))
        if idx_S_sl != -1: cand.append(("SHORT","SL", idx_S_sl))

        if not cand:
            continue  # 無標籤（丟棄）

        # 取最早的事件；同根若多事件 → 丟棄（避免歧義）
        steps = [c[2] for c in cand]
        m = np.min(steps)
        firsts = [c for c in cand if c[2] == m]
        if len(firsts) != 1:
            continue  # 同根多事件，丟棄
        label[i] = 1.0 if firsts[0][0] == "LONG" else 0.0

    out["label"] = label
    return out


def make_label(df: pd.DataFrame, mode: str = LABEL_MODE, bars_ahead: int = BARS_AHEAD) -> pd.DataFrame:
    if mode == "tplus":
        return make_label_tplus(df, bars_ahead)
    elif mode == "atr_event":
        return make_label_atr_event(df, bars_ahead)
    else:
        raise ValueError(f"Unknown LABEL_MODE: {mode}")

# ===================== 訓練與門檻選擇 =====================
def _fit_classifier(X_train: pd.DataFrame, y_train: np.ndarray):
    if _XGB_OK:
        pos = float((y_train == 1).sum()); neg = float((y_train == 0).sum())
        spw = max(1.0, neg / max(1.0, pos))
        model = xgb.XGBClassifier(
            n_estimators=700, max_depth=4, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.0, reg_lambda=1.0, min_child_weight=1.0,
            eval_metric='logloss', random_state=42, tree_method="hist",
            scale_pos_weight=spw, n_jobs=0
        )
        model.fit(X_train, y_train)
        return model, "xgboost"
    else:
        model = HistGradientBoostingClassifier(random_state=42, max_depth=3)
        model.fit(X_train, y_train)
        return model, "hgb"

def _pick_threshold(df_eval: pd.DataFrame, thr_grid=THRESH_GRID,
                    target_precision=TARGET_PRECISION, min_trades_30d=TARGET_TRADES_30D):
    last_ts = df_eval["timestamp"].max()
    win_start = last_ts - timedelta(days=30)
    recent = df_eval[df_eval["timestamp"] >= win_start].copy()
    if recent.empty:
        recent = df_eval.copy()

    best = {"thr": None, "precision": -1, "trades_30d": -1, "recall": -1, "coverage": -1}
    fallback = best.copy()

    for thr in thr_grid:
        preds = (recent["proba_up"] >= thr).astype(int)
        trades = int(preds.sum())
        precision = 0.0 if trades == 0 else precision_score(recent["label"], preds)
        recall = float(((preds==1) & (recent["label"]==1)).sum()) / max(1, (recent["label"]==1).sum())
        coverage = trades / max(1, len(recent))

        # 折衷解優先：precision → trades
        if (precision > fallback["precision"]) or (precision == fallback["precision"] and trades > fallback["trades_30d"]):
            fallback = {"thr":thr, "precision":precision, "trades_30d":trades, "recall":recall, "coverage":coverage}

        # 同時達標 → 優先 trades 較多
        if precision >= target_precision and trades >= min_trades_30d:
            if (best["thr"] is None) or (trades > best["trades_30d"]):
                best = {"thr":thr, "precision":precision, "trades_30d":trades, "recall":recall, "coverage":coverage}

    return best if best["thr"] is not None else fallback

# ===================== 主流程 =====================
def main():
    # 1) 載入 15m OHLCV
    df_raw = load_data_from_file(DATA_PATH)
    df_raw = df_raw.dropna().sort_values("timestamp").reset_index(drop=True)

    # 2) 多週期特徵
    df_feat, feature_cols = build_multiframe_features(df_raw)

    # 3) 打標（與策略一致的 4h horizon）
    df_lab = make_label(df_feat, mode=LABEL_MODE, bars_ahead=BARS_AHEAD)
    df_lab = df_lab.dropna(subset=["label"]).reset_index(drop=True)

    # 4) 時序切分
    n = len(df_lab); cut = int(n * 0.8)
    train_df, valid_df = df_lab.iloc[:cut].copy(), df_lab.iloc[cut:].copy()

    # 5) 標準化
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_df[feature_cols]), columns=feature_cols, index=train_df.index)
    X_valid = pd.DataFrame(scaler.transform(valid_df[feature_cols]), columns=feature_cols, index=valid_df.index)
    y_train = train_df["label"].values
    y_valid = valid_df["label"].values

    # 6) 訓練
    model, model_name = _fit_classifier(X_train, y_train)

    # 7) 驗證集機率 & 門檻
    if hasattr(model, "predict_proba"):
        proba_valid = model.predict_proba(X_valid)[:, 1]
    else:
        preds = model.predict(X_valid)
        proba_valid = preds.astype(float)
    valid_eval = valid_df[["timestamp","label"]].copy()
    valid_eval["proba_up"] = proba_valid

    best = _pick_threshold(valid_eval)

    # 8) 存檔
    if model_name == "xgboost":
        model.save_model(MODEL_PATH)
    else:
        joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "feature_cols": feature_cols,
        "threshold": float(best["thr"]) if best["thr"] is not None else 0.60,
        "valid_precision": float(best["precision"]),
        "valid_trades_30d": int(best["trades_30d"]),
        "valid_recall": float(best["recall"]),
        "valid_coverage": float(best["coverage"]),
        "time_split_ratio": 0.2,
        "resample_4h": RESAMPLE_4H,
        "model_type": model_name,
        "label_mode": LABEL_MODE,
        "bars_ahead": int(BARS_AHEAD)
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True)

    # 9) 輸出摘要
    ok = (best["precision"] >= TARGET_PRECISION and best["trades_30d"] >= TARGET_TRADES_30D)
    tag = "（達成目標！）" if ok else "（未完全達成，為折衷門檻）"
    print("✅ 訓練完成與輸出：")
    print(f"   - 模型: {MODEL_PATH}")
    print(f"   - Scaler: {SCALER_PATH}")
    print(f"   - Meta: {META_PATH}")
    print(f"   - Label mode: {LABEL_MODE}（bars_ahead={BARS_AHEAD}）")
    print("\n🎯 建議門檻：")
    print(f"   threshold = {meta['threshold']:.2f} → precision={meta['valid_precision']:.2%}, 近30天觸發={meta['valid_trades_30d']} {tag}")

if __name__ == "__main__":
    main()
