

import sys, os, json, joblib, pickle
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from strategy.logic import TradingStrategy, compute_adx

# ===== Config =====
DATA_PATH   = "data/btc_15m_data_360days.csv"
MODEL_PATH  = "h16_dynamic/cls_model_h16.pkl"
SCALER_PATH = "h16_dynamic/scaler_h16.joblib"
FEE_BPS     = 8.0
SLIP_BPS    = 2.0  # Iteration 105.1: Emergency Stop (0.02%)
TH_BASE     = 0.92 # Iteration 105.1: High Conviction
TP_PCT      = 0.025
SL_PCT      = 0.010

def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100/(1+rs))

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()

def _atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(), (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)
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

def build_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    c = df["close"]; v = df["volume"]
    df["ret_1"] = c.pct_change(1); df["ret_4"] = c.pct_change(4)
    df["ret_16"] = c.pct_change(16); df["ret_32"] = c.pct_change(32)
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std(ddof=0)
    df["bb_z20"] = (c - ma20) / (std20 + 1e-12)
    df["rv_16"] = df["ret_1"].rolling(16).std(ddof=0)
    df["slope_log_8"] = _slope_log(c, 8); df["slope_log_16"] = _slope_log(c, 16); df["slope_log_32"] = _slope_log(c, 32)
    ema_fast = _ema(c, 12); ema_slow = _ema(c, 48)
    df["ema_fast_dist"] = (c - ema_fast) / (abs(ema_fast) + 1e-12)
    df["ema_fast_slow_gap"] = (ema_fast - ema_slow) / (abs(ema_slow) + 1e-12)
    df["mom_ratio"] = (c - ema_fast) / (abs(ema_fast - ema_slow) + 1e-12)
    df["vol_chg"] = v.pct_change().fillna(0); df["vol_z48"] = _zscore(v, 48)
    df["atr14"] = _atr(df, 14); df["atr_ratio"] = df["atr14"] / (c + 1e-12)
    df["rsi14_15m"] = _rsi(c, 14); df["adx14"] = compute_adx(df, 14)
    df["ema_slow"] = ema_slow
    
    di = df.set_index("timestamp")
    c_1h = di["close"].resample("1h").last().ffill()
    c_4h = di["close"].resample("4h").last().ffill()
    rsi_1h = _rsi(c_1h, 14); rsi_4h = _rsi(c_4h, 14)
    df = pd.merge_asof(df, pd.DataFrame({"timestamp": rsi_1h.index, "rsi14_1h": rsi_1h.values}), on="timestamp", direction="backward")
    df = pd.merge_asof(df, pd.DataFrame({"timestamp": rsi_4h.index, "rsi14_4h": rsi_4h.values}), on="timestamp", direction="backward")
    return df

def main():
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    
    with open(MODEL_PATH, "rb") as f: clf = pickle.load(f)
    scaler = joblib.load(SCALER_PATH)
    features_list = [
        "ret_1", "ret_4", "ret_16", "ret_32",
        "bb_z20", "rv_16", "slope_log_8", "slope_log_16", "slope_log_32",
        "ema_fast_dist", "ema_fast_slow_gap", "mom_ratio",
        "vol_chg", "vol_z48", "atr14", "atr_ratio",
        "rsi14_15m", "rsi14_1h", "rsi14_4h"
    ]
    
    strategy = TradingStrategy(th_base=TH_BASE, tp_pct=TP_PCT, sl_pct=SL_PCT)
    
    trades = []
    in_pos = False; pos = {}
    fee = FEE_BPS / 10000.0; slip = SLIP_BPS / 10000.0
    
    df = df.dropna(subset=features_list).reset_index(drop=True)
    max_up = 0; max_dn = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        X = row[features_list].to_numpy().reshape(1, -1)
        p_up = float(clf.predict_proba(scaler.transform(X))[0, 1])
        p_dn = 1.0 - p_up
        max_up = max(max_up, p_up); max_dn = max(max_dn, p_dn)
        
        if in_pos:
            side = pos["side"]
            pos["bars_held"] += 1
            px = row["close"]; hi = row["high"]; lo = row["low"]
            
            # Scale-out TP (50% at +2.0%)
            if not pos["scaled_out"]:
                if side == "LONG" and hi >= pos["entry_price"] * 1.020: pos["scaled_out"] = True
                elif side == "SHORT" and lo <= pos["entry_price"] * 0.980: pos["scaled_out"] = True
            
            hit_tp = (hi >= pos["tp"]) if side == "LONG" else (lo <= pos["tp"])
            hit_sl = (lo <= pos["sl"]) if side == "LONG" else (hi >= pos["sl"])
            
            if hit_tp or hit_sl or pos["bars_held"] >= 16:
                px_out = pos["tp"] if hit_tp else (pos["sl"] if hit_sl else px)
                if side == "LONG": ret = (px_out - pos["entry_price"]) / pos["entry_price"] - (fee*2 + slip*2)
                else: ret = (pos["entry_price"] - px_out) / pos["entry_price"] - (fee*2 + slip*2)
                
                if pos["scaled_out"]: ret = 0.5 * (0.020 - (fee*2 + slip*2)) + 0.5 * ret
                
                trades.append({"ret": ret, "side": side, "bars": pos["bars_held"]})
                in_pos = False; pos = {}
        else:
            side, size = strategy.check_entry(p_up, p_dn, row.to_dict(), {"regime_long_ok": True, "regime_short_ok": True})
            if side:
                sl, tp = strategy.build_tp_sl(side, row["close"])
                in_pos = True
                pos = {"side": side, "entry_price": row["close"], "sl": sl, "tp": tp, "bars_held": 0, "scaled_out": False}
                
    print(f"Max Prob Up: {max_up:.4f}, Max Prob Down: {max_dn:.4f}")
    if trades:
        tdf = pd.DataFrame(trades)
        print(f"Trades: {len(tdf)}")
        print(f"Win Rate: {(tdf['ret'] > 0).mean():.2%}")
        print(f"Total PnL: {tdf['ret'].sum():.2%}")
    else:
        print("No trades.")

if __name__ == "__main__":
    main()

