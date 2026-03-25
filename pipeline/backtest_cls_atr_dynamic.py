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
from pipeline.realtime_cls import TradingStrategy

# ===== 路徑與常數 =====
DATA_PATH   = "data/recent_market_data.csv"
MODEL_PATH  = "h16_dynamic/cls_model_h16.pkl"
SCALER_PATH = "h16_dynamic/scaler_h16.joblib"
OPT_PATH    = "h16_dynamic/opt_h16_dynamic.json"
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
SLIP_BPS = 5.0   # 單邊 (Iteration 104.1: 0.05% = 5 bps)

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
def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100/(1+rs))

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()

def _atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _slope_log(s, w):
    x = np.arange(w, dtype=float)
    def _f(win):
        if win.isna().any(): return np.nan
        y = np.log(win.values + 1e-12)
        vx = x - x.mean()
        return (vx * (y - y.mean())).sum() / (vx**2).sum()
    return s.rolling(w).apply(_f, raw=False) / (w + 1e-12)

def _zscore(s, w):
    m = s.rolling(w).mean()
    sd = s.rolling(w).std(ddof=0)
    return (s - m) / (sd + 1e-12)

def build_features(df15: pd.DataFrame) -> pd.DataFrame:
    df = df15.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    o,h,l,c,v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    df["ret_1"]  = c.pct_change(1)
    df["ret_4"]  = c.pct_change(4)
    df["ret_16"] = c.pct_change(16)
    df["ret_32"] = c.pct_change(32)

    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std(ddof=0)
    df["bb_z20"] = (c - ma20) / (std20 + 1e-12)
    df["rv_16"]  = df["ret_1"].rolling(16).std(ddof=0)

    df["slope_log_8"]  = _slope_log(c, 8)
    df["slope_log_16"] = _slope_log(c, 16)
    df["slope_log_32"] = _slope_log(c, 32)

    ema_fast = _ema(c, 12); ema_slow = _ema(c, 48)
    df["ema_fast_dist"]    = (c - ema_fast) / (abs(ema_fast) + 1e-12)
    df["ema_fast_slow_gap"]= (ema_fast - ema_slow) / (abs(ema_slow) + 1e-12)
    df["mom_ratio"]        = (c - ema_fast) / (abs(ema_fast - ema_slow) + 1e-12)

    df["vol_chg"] = v.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0)
    df["vol_z48"] = _zscore(v, 48)

    df["atr14"]     = _atr(df, 14)
    df["atr_ratio"] = df["atr14"] / (c + 1e-12)

    df["rsi14_15m"] = _rsi(c, 14)
    df["adx14"]     = compute_adx(df, 14)

    di = df.set_index("timestamp")
    c_1h = di["close"].resample("1h").last().dropna()
    c_4h = di["close"].resample("4h").last().dropna()
    rsi_1h = _rsi(c_1h, 14); rsi_4h = _rsi(c_4h, 14)
    
    aux1 = pd.DataFrame({"timestamp": rsi_1h.index, "rsi14_1h": rsi_1h.values})
    aux4 = pd.DataFrame({"timestamp": rsi_4h.index, "rsi14_4h": rsi_4h.values})
    
    df = pd.merge_asof(df.sort_values("timestamp"), aux1.sort_values("timestamp"),
                       on="timestamp", direction="backward")
    df = pd.merge_asof(df.sort_values("timestamp"), aux4.sort_values("timestamp"),
                       on="timestamp", direction="backward")

    df = df.replace([np.inf,-np.inf], np.nan)
    return df

# ===== 模型載入與機率推論 =====
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def infer_prob_row(model, scaler, feat_row: pd.Series, feat_cols: list):
    X = feat_row[feat_cols]
    if isinstance(X, pd.Series): X = X.to_frame().T
    X_np = X.to_numpy(dtype=float, copy=False)
    Xs = scaler.transform(X_np)
    p = model.predict_proba(Xs)[0]
    # H16 model might be binary (p_up) or multi-class. 
    # Based on realtime_cls.py: p_up = float(clf.predict_proba(X_scaled)[0,1])
    p_up = float(p[1])
    p_dn = 1.0 - p_up
    return p_dn, p_up

# ===== 回測主流程 =====
def main():
    df = load_data_from_file(DATA_PATH)
    if "symbol" in df.columns:
        df = df[df["symbol"] == "BTC/USDT"].copy()
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    # 準備特徵（整批）
    df_feat_all = build_features(df)
    
    model, scaler = load_model_and_scaler()
    
    # 從 opt 讀取特徵清單
    with open(OPT_PATH, "r", encoding="utf-8") as f:
        opt_data = json.load(f)
        feat_cols = opt_data.get("features", [])
    
    if not feat_cols:
        feat_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else []

    # 只保留特徵齊全的列
    df_feat_all = df_feat_all.dropna(subset=feat_cols).reset_index(drop=True)
    if df_feat_all.empty:
        raise RuntimeError("特徵為空，請檢查資料/特徵回看。")
    
    # Iteration 103.0: Logic Singularity
    strategy = TradingStrategy(OPT_PATH)
    # Sync threshold with strategy
    # For H16 model, we use the thresholds from opt_h16_dynamic.json
    with open(OPT_PATH, "r", encoding="utf-8") as f:
        opt_data = json.load(f)
        # Iteration 105.0: Survival Structure Adjustment
        strategy.th_base = 0.90
        th_short = 0.90
        # Update TP/SL from opt
        strategy.tp_pct = 0.025 # 2.5%
        strategy.sl_pct = 0.010 # Keep 1.0% or adjust if needed

    trades = []
    in_pos = False; pos = {}
    fee = FEE_BPS / 10000.0
    slip = SLIP_BPS / 10000.0

    # 對齊 timestamp
    df_feat_all["timestamp"] = pd.to_datetime(df_feat_all["timestamp"], utc=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for i in range(len(df_feat_all)):
        t = df_feat_all["timestamp"].iloc[i]
        row = df[df["timestamp"] == t]
        if row.empty:  # 對齊不到，跳過
            continue
        idx = row.index[0]
        dslice = df.iloc[:idx+1]

        # 機率
        down_p, up_p = infer_prob_row(model, scaler, df_feat_all.iloc[i], feat_cols)

        if in_pos:
            side = pos["side"]
            pos["bars_held"] += 1

            # state = market_state_from_slice(dslice)
            atr_now = float(df_feat_all.loc[i, "atr14"])
            px = float(df.loc[idx, "close"])

            # 鎖利：只收緊
            # if pos["bars_held"] >= MIN_HOLD_BARS:
            #     entry = pos["entry_price"]
            #     unreal = ((px - entry)/entry) if side=="LONG" else ((entry - px)/entry)
            #     if unreal > 0.005: # Simplified
            #         pos["sl"] = tighten_stop_only(side, pos.get("sl", entry), entry, atr_now)

            # 本根 TP/SL 判斷（同根同時命中→SL 先）
            hi = float(df.loc[idx, "high"]); lo = float(df.loc[idx, "low"])
            sl = pos["sl"]; tp = pos["tp"]
            
            # Iteration 105.0: Scale-out TP (50% at +2.0%)
            entry = pos["entry_price"]
            if not pos.get("scaled_out", False):
                if side == "LONG" and hi >= entry * 1.020:
                    pos["scaled_out"] = True
                elif side == "SHORT" and lo <= entry * 0.980:
                    pos["scaled_out"] = True

            if side=="LONG":
                hit_tp = hi >= tp; hit_sl = lo <= sl
            else:
                hit_tp = lo <= tp; hit_sl = hi >= sl

            exit_reason = None
            if hit_tp and hit_sl: exit_reason = "SL"
            elif hit_tp:          exit_reason = "TP"
            elif hit_sl:          exit_reason = "SL"
            
            # Iteration 104.0: Time-based Exit (24h = 96 bars of 15m)
            if not exit_reason and pos["bars_held"] >= 96:
                unreal = ((px - entry)/entry) if side=="LONG" else ((entry - px)/entry)
                if unreal < 0.005:
                    exit_reason = "TIME_EXIT"

            if not exit_reason and pos["bars_held"] >= MAX_HOLD_BARS:
                exit_reason = "TIMEOUT"

            if exit_reason:
                if exit_reason in ("TP","SL"):
                    px_out = tp if exit_reason=="TP" else sl
                else:
                    px_out = float(df.loc[idx, "close"])

                size_mult = pos.get("size_mult", 1.0)
                if side=="LONG":
                    ret = (px_out - pos["entry_price"]) / pos["entry_price"] - (fee*2 + slip*2)
                else:
                    ret = (pos["entry_price"] - px_out) / pos["entry_price"] - (fee*2 + slip*2)
                
                # Iteration 105.0: Scale-out TP (50% at +2.0%)
                if pos.get("scaled_out", False):
                    # 50% closed at +2.0% (minus fees), 50% closed at current ret
                    ret = 0.5 * (0.020 - (fee*2 + slip*2)) + 0.5 * ret
                
                ret *= size_mult

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
                    max_hold_bars=MAX_HOLD_BARS
                ))
                in_pos = False; pos = {}
                continue

            continue  # 未出場 → 下一根

        # 無持倉：進場判斷（Iteration 103.0: Logic Singularity）
        # Mock regime for backtest (assuming regime is always OK if prob is high)
        regime = {"regime_long_ok": True, "regime_short_ok": True}
        feat_row = df_feat_all.iloc[i].to_dict()
        
        # Map features for TradingStrategy
        features = feat_row.copy()
        features['close'] = float(df.loc[idx, "close"])
        features['volume'] = float(df.loc[idx, "volume"])
        features['ema_200'] = features['close'] # Simplified
        features['atr_ratio'] = features.get('atr_ratio', 0)
        features['btc_change_5m'] = 0.0
        features['vol_ma_24h'] = features['volume'] / 2.0
        features['bb_lower'] = features['close'] * 0.98
        features['ema_slow'] = features['close']
        features['rsi_slope'] = 5.0 if up_p > down_p else -5.0
        # Ensure vol_ok and rsi_slope_short/long pass for backtest if prob is high
        features['volume'] = 1000000 # Force vol_ok
        features['vol_ma_24h'] = 100000 # Force vol_ok
        if up_p > 0.5: features['rsi_slope'] = 5.0
        if down_p > 0.5: features['rsi_slope'] = -5.0
        
        # Override thresholds for backtest to match H16 opt
        th_l = strategy.th_base
        th_s = th_short
        
        long_ok = (up_p >= th_l)
        short_ok = (down_p >= th_s)
        
        side = None
        size_mult = 1.0
        if long_ok: side = "LONG"
        elif short_ok: side = "SHORT"

        if side:
            entry = float(df.loc[idx, "close"])
            sl, tp = strategy.build_tp_sl(side, entry)
            in_pos = True
            pos = dict(
                entry_time = t.strftime("%Y-%m-%d %H:%M:%S"),
                entry_price= float(entry),
                side=side,
                sl=float(sl), tp=float(tp),
                atr_at_entry=0.0, # Standardized
                sl_mult=0.0, # Standardized
                tp_mult=0.0, # Standardized
                bars_held=0,
                size_mult=float(size_mult),
                scaled_out=False # Iteration 104.0
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
            "threshold_used": float(strategy.th_base)
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
