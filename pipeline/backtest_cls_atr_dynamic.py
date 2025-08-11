# backtest_cls_atr_dynamic.py —— 15m + 4h 多週期特徵，使用訓練門檻與動態 ATR TP/SL 回測
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import yaml
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from datetime import datetime

from data.data_module import load_data_from_file
from data.feature_engineering import add_indicators  # 你提供的版本（含 core.add_features）

# ===== 路徑與常數 =====
DATA_PATH   = "data/btc_15m_data_3days.csv"
MODEL_PATH  = "models/xgb_cls_model.json"
SCALER_PATH = "models/xgb_cls_scaler.joblib"
META_PATH   = "models/xgb_cls_meta.yaml"

OUT_DIR = os.path.join(os.getcwd(), "backtests")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 動態 TP/SL 參數（與線上一致） =====
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
MIN_HOLD_BARS     = 2
MAX_HOLD_BARS     = 16

EMA_GAP_STRONG = 0.0025
EMA_GAP_RANGE  = 0.0015
EMA_SAME_SIDE_K = 4

FEE_BPS  = 8.0   # 單邊
SLIP_BPS = 0.0   # 單邊

# ===== 讀取訓練 meta（門檻/特徵欄位） =====
DEFAULT_THR = 0.60
DEFAULT_FEATURE_COLS = None
try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        _META = yaml.safe_load(f) or {}
    PROB_THR = float(_META.get("threshold", DEFAULT_THR))
    META_FEATURE_COLS = _META.get("feature_cols", DEFAULT_FEATURE_COLS)
except Exception:
    PROB_THR = DEFAULT_THR
    META_FEATURE_COLS = DEFAULT_FEATURE_COLS

# ===== 指標/市況/動態倍數 =====
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
    if not np.isfinite(atr.iloc[-1]):
        pct_std = df["close"].pct_change().rolling(n, min_periods=2).std()
        atr_est = (pct_std * df["close"]).rolling(n, min_periods=1).mean()
        atr = atr.fillna(atr_est)
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

def market_state_from_slice(df_slice: pd.DataFrame) -> dict:
    ds = df_slice.copy()
    for col in ("high","low","close"):
        ds[col] = pd.to_numeric(ds[col], errors="coerce")
    ds = ds.dropna(subset=["high","low","close"])

    if len(ds) < max(ATR_N, 5):
        return dict(
            atr_now=np.nan, adx_now=0.0, high_vol=False, low_vol=False,
            strong_trend=False, ranging=True, atr_p30=np.nan, atr_p70=np.nan
        )
    ds["ATR"] = calc_atr(ds, ATR_N)
    ds["EMA20"] = ds["close"].ewm(span=20, adjust=False).mean()
    ds["EMA50"] = ds["close"].ewm(span=50, adjust=False).mean()
    ds["ADX"] = compute_adx(ds, ADX_N)

    atr_now = float(ds["ATR"].iloc[-1])
    adx_now = float(ds["ADX"].iloc[-1]) if np.isfinite(ds["ADX"].iloc[-1]) else 0.0

    tail = ds["ATR"].tail(ATR_LOOKBACK_PCTL).dropna()
    if len(tail) < 10:
        tail = ds["ATR"].dropna().tail(ATR_LOOKBACK_PCTL)
    atr_p30 = float(np.percentile(tail, ATR_PCTL_LOW*100)) if len(tail) else atr_now
    atr_p70 = float(np.percentile(tail, ATR_PCTL_HIGH*100)) if len(tail) else atr_now
    high_vol = (np.isfinite(atr_now) and np.isfinite(atr_p70) and atr_now > atr_p70)
    low_vol  = (np.isfinite(atr_now) and np.isfinite(atr_p30) and atr_now < atr_p30)

    price_now = float(ds["close"].iloc[-1])
    ema20_now = float(ds["EMA20"].iloc[-1]); ema50_now = float(ds["EMA50"].iloc[-1])
    ema_gap_pct = abs(ema20_now - ema50_now) / price_now if price_now > 0 else 0.0
    ema_side = (ds["EMA20"] > ds["EMA50"]).astype(int) - (ds["EMA20"] < ds["EMA50"]).astype(int)
    same_side_lastk = int(abs(ema_side.tail(EMA_SAME_SIDE_K).sum())) == EMA_SAME_SIDE_K

    adx_strong = adx_now >= ADX_TREND
    adx_range  = adx_now <= RANGING_ADX
    strong_trend = bool(adx_strong or ((ema_gap_pct >= EMA_GAP_STRONG) and same_side_lastk))
    ranging      = bool(adx_range  or  (ema_gap_pct <  EMA_GAP_RANGE))

    return dict(
        atr_now=atr_now, adx_now=adx_now, high_vol=high_vol, low_vol=low_vol,
        strong_trend=strong_trend, ranging=ranging, atr_p30=atr_p30, atr_p70=atr_p70
    )

def choose_multipliers(side: str, state: dict):
    if side == "LONG":
        sl_mult = BASE_SL_LONG; tp_mult = BASE_TP_LONG
    else:
        sl_mult = BASE_SL_SHORT; tp_mult = BASE_TP_SHORT
    if state["high_vol"]:
        sl_mult *= 1.20; tp_mult *= 1.20
    elif state["low_vol"]:
        sl_mult *= 0.80; tp_mult *= 0.80
    if state["strong_trend"]:
        tp_mult *= 1.30
    elif state["ranging"]:
        sl_mult *= 0.90; tp_mult *= 0.70
    sl_mult = float(np.clip(sl_mult, 0.8, 3.0))
    tp_mult = float(np.clip(tp_mult, 1.2, 5.0))
    return sl_mult, tp_mult

def build_tp_sl(side: str, entry: float, atr: float, sl_mult: float, tp_mult: float):
    if side == "LONG":
        sl = round(entry - sl_mult * atr, 2); tp = round(entry + tp_mult * atr, 2)
    else:
        sl = round(entry + sl_mult * atr, 2); tp = round(entry - tp_mult * atr, 2)
    return sl, tp

def tighten_stop_only(side: str, current_sl: float, entry_price: float, atr_now: float):
    if side == "LONG":
        proposed = round(entry_price - 1.2 * atr_now, 2); return max(current_sl, proposed)
    else:
        proposed = round(entry_price + 1.2 * atr_now, 2); return min(current_sl, proposed)

# ===== 多週期特徵（與訓練、線上一致） =====
def _resample_4h(df15):
    d = df15.set_index("timestamp")
    o = d["open"].resample("4H").first()
    h = d["high"].resample("4H").max()
    l = d["low"].resample("4H").min()
    c = d["close"].resample("4H").last()
    v = d["volume"].resample("4H").sum(min_count=1)
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ["open","high","low","close","volume"]
    return out.dropna().reset_index()

def _add_derived_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    d = df.copy()
    for col in ("open","high","low","close","volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    for k in ["close","ema_fast","ema_slow","rsi","bb_high","bb_low","macd_diff","EMA50"]:
        if k not in d.columns:
            d[k] = np.nan
    d["ema_gap_pct"]   = (d["ema_fast"] - d["ema_slow"]) / d["close"]
    width = (d["bb_high"] - d["bb_low"]).replace(0, np.nan)
    d["band_pct"]      = (d["close"] - d["bb_low"]) / width
    d["rsi_dev"]       = (d["rsi"] - 50.0) / 50.0
    d["macd_norm"]     = d["macd_diff"] / d["close"]
    d["price_above_ema"]= (d["close"] - d["EMA50"]) / d["close"]
    if prefix:
        rename = {c: f"{prefix}{c}" for c in ["ema_gap_pct","band_pct","rsi_dev","macd_norm","price_above_ema"] if c in d.columns}
        d = d.rename(columns=rename)
    return d

def _select_feature_columns(df: pd.DataFrame) -> list:
    drop = set(["timestamp","open","high","low","close","volume","return_next","label"])
    return [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]

def build_features(df15: pd.DataFrame) -> pd.DataFrame:
    f15 = add_indicators(df15.copy())
    f15 = _add_derived_features(f15).dropna().reset_index(drop=True)
    f4h = add_indicators(_resample_4h(df15))
    f4h = _add_derived_features(f4h, prefix="h4_").dropna().reset_index(drop=True)
    h4_cols = [c for c in f4h.columns if c not in ["timestamp","open","high","low","close","volume"]]
    f4h_pref = f4h[["timestamp"] + h4_cols].copy()
    merged = pd.merge_asof(
        f15.sort_values("timestamp"),
        f4h_pref.sort_values("timestamp"),
        on="timestamp", direction="backward"
    ).dropna().reset_index(drop=True)
    return merged

# ===== 模型載入與機率推論 =====
def load_model_and_scaler():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def infer_prob_row(model, scaler, feat_row: pd.Series):
    feat_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else META_FEATURE_COLS
    if not feat_cols:
        raise ValueError("無法取得特徵欄位（scaler.feature_names_in_ 與 meta.feature_cols 均不可用）")
    X = feat_row[feat_cols]
    if isinstance(X, pd.Series): X = X.to_frame().T  # 保持 DataFrame
    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[0]
    return float(p[0]), float(p[1])  # down, up

# ===== 回測主流程 =====
def main():
    df = load_data_from_file(DATA_PATH)
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    # 準備特徵（整批）
    df_feat_all = build_features(df)
    if df_feat_all.empty:
        raise RuntimeError("特徵為空，請檢查資料/特徵回看。")

    model, scaler = load_model_and_scaler()

    trades = []
    in_pos = False; pos = {}
    fee = FEE_BPS / 10000.0
    slip = SLIP_BPS / 10000.0
    thr  = PROB_THR

    # 對齊 timestamp
    if "timestamp" not in df_feat_all.columns:
        raise ValueError("特徵缺少 timestamp 欄位。")
    df_feat_all["timestamp"] = pd.to_datetime(df_feat_all["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for i in range(len(df_feat_all)):
        t = df_feat_all["timestamp"].iloc[i]
        row = df[df["timestamp"] == t]
        if row.empty:  # 對齊不到，跳過
            continue
        idx = row.index[0]
        dslice = df.iloc[:idx+1]

        # 機率
        down_p, up_p = infer_prob_row(model, scaler, df_feat_all.iloc[i])

        if in_pos:
            side = pos["side"]
            pos["bars_held"] += 1

            state = market_state_from_slice(dslice)
            atr_now = state["atr_now"]
            px = float(df.loc[idx, "close"])

            # 鎖利：只收緊
            if pos["bars_held"] >= MIN_HOLD_BARS:
                entry = pos["entry_price"]
                unreal = ((px - entry)/entry) if side=="LONG" else ((entry - px)/entry)
                if unreal > 0.005 or state["low_vol"]:
                    pos["sl"] = tighten_stop_only(side, pos.get("sl", entry), entry, atr_now)

            # 本根 TP/SL 判斷（同根同時命中→SL 先）
            hi = float(df.loc[idx, "high"]); lo = float(df.loc[idx, "low"])
            sl = pos["sl"]; tp = pos["tp"]
            if side=="LONG":
                hit_tp = hi >= tp; hit_sl = lo <= sl
            else:
                hit_tp = lo <= tp; hit_sl = hi >= sl

            exit_reason = None
            if hit_tp and hit_sl: exit_reason = "SL"
            elif hit_tp:          exit_reason = "TP"
            elif hit_sl:          exit_reason = "SL"
            if not exit_reason and pos["bars_held"] >= MAX_HOLD_BARS:
                exit_reason = "TIMEOUT"

            if exit_reason:
                if exit_reason in ("TP","SL"):
                    px_out = tp if exit_reason=="TP" else sl
                else:
                    px_out = float(df.loc[idx, "close"])

                if side=="LONG":
                    entry_eff = pos["entry_price"] * (1 + fee + slip)
                    exit_eff  = px_out * (1 - fee - slip)
                    ret = (exit_eff - entry_eff) / entry_eff
                else:
                    entry_eff = pos["entry_price"] * (1 - fee - slip)
                    exit_eff  = px_out * (1 + fee + slip)
                    ret = (entry_eff - exit_eff) / entry_eff

                trades.append(dict(
                    entry_time = pos["entry_time"],
                    exit_time  = t.strftime("%Y-%m-%d %H:%M:%S"),
                    side       = side,
                    reason     = exit_reason,
                    entry_price= round(pos["entry_price"], 2),
                    exit_price = round(px_out, 2),
                    return_pct = float(np.round(ret, 6)),
                    holding_minutes = int(pos["bars_held"]*15),
                    tp_sl_mode = "ATR_DYNAMIC",
                    sl=float(sl), tp=float(tp),
                    sl_mult=float(pos["sl_mult"]), tp_mult=float(pos["tp_mult"]),
                    atr_at_entry=float(pos["atr_at_entry"]),
                    atr_n=ATR_N,
                    bars_held_close=int(pos["bars_held"]),
                    max_hold_bars=MAX_HOLD_BARS,
                    high_vol=bool(state["high_vol"]),
                    low_vol=bool(state["low_vol"]),
                    strong_trend=bool(state["strong_trend"]),
                    ranging=bool(state["ranging"])
                ))
                in_pos = False; pos = {}
                continue

            continue  # 未出場 → 下一根

        # 無持倉：進場判斷（使用訓練最佳門檻）
        side = None
        if up_p > thr:   side = "LONG"
        elif down_p > thr: side = "SHORT"

        if side:
            state = market_state_from_slice(dslice)
            atr_now = state["atr_now"]
            if not np.isfinite(atr_now) or atr_now <= 0:
                continue
            sl_mult, tp_mult = choose_multipliers(side, state)
            entry = float(df.loc[idx, "close"])
            sl, tp = build_tp_sl(side, entry, atr_now, sl_mult, tp_mult)
            in_pos = True
            pos = dict(
                entry_time = t.strftime("%Y-%m-%d %H:%M:%S"),
                entry_price= float(entry),
                side=side,
                sl=float(sl), tp=float(tp),
                atr_at_entry=float(atr_now),
                sl_mult=float(sl_mult), tp_mult=float(tp_mult),
                bars_held=0
            )

    # ===== 輸出 =====
    trades_df = pd.DataFrame(trades)
    trades_path = os.path.join(OUT_DIR, "bt_trades.csv")
    trades_df.to_csv(trades_path, index=False)

    if trades_df.empty:
        summary = {"trades": 0, "note": "no trades", "threshold_used": float(thr)}
    else:
        wins = (trades_df["return_pct"] > 0).mean()
        avg_ret = trades_df["return_pct"].mean()
        gross = trades_df["return_pct"].sum()
        eq = (1 + trades_df["return_pct"]).cumprod()
        peak = eq.cummax()
        mdd = (eq/peak - 1.0).min()
        summary = {
            "trades": int(len(trades_df)),
            "win_rate": float(np.round(wins, 4)),
            "avg_return_per_trade": float(np.round(avg_ret, 6)),
            "total_return_simple": float(np.round(gross, 4)),
            "avg_holding_minutes": float(np.round(trades_df["holding_minutes"].mean(), 2)),
            "max_drawdown_compounded": float(np.round(mdd, 4)),
            "fee_bps": float(FEE_BPS),
            "slip_bps": float(SLIP_BPS),
            "threshold_used": float(thr)
        }

    summ_path = os.path.join(OUT_DIR, "bt_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Backtest Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved trades to: {trades_path}")
    print(f"Saved summary to: {summ_path}")

if __name__ == "__main__":
    main()
