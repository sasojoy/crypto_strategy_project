
import sys, os, json, pickle, joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ===== Paths =====
DATA_PATH = "/workspace/crypto_strategy_project/data/recent_market_data.csv"
MODEL_PATH = "/workspace/crypto_strategy_project/h16_dynamic/cls_model_h16.pkl"
SCALER_PATH = "/workspace/crypto_strategy_project/h16_dynamic/scaler_h16.joblib"
OPT_PATH = "/workspace/crypto_strategy_project/h16_dynamic/opt_h16_dynamic.json"

# ===== Load Production Parameters =====
with open(MODEL_PATH, "rb") as f: clf = pickle.load(f)
scaler = joblib.load(SCALER_PATH)
with open(OPT_PATH, "r") as f:
    _opt_full = json.load(f)
    _opt = _opt_full["best"]
    FEATURES = _opt_full["features"]
    TH_LONG = 0.85
    TH_SHORT = 0.85
    TP_L, SL_L, TP_S, SL_S = float(_opt["tpL"]), float(_opt["slL"]), float(_opt["tpS"]), float(_opt["slS"])

# --- [COPY-PASTE FROM realtime_cls.py & feature_engineering_h16.py] ---
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

def build_regime_masks(px: pd.DataFrame):
    c = px["close"]
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std(ddof=0)
    bb_z = (c - ma20) / (std20 + 1e-12)
    rsi14 = _rsi(c, 14)
    bb_z = bb_z.fillna(0); rsi14 = rsi14.fillna(50)
    BBZ_ZMIN = 0.25; RSI_LONG_MAX = 35; RSI_SHORT_MIN = 65
    mask_long  = (bb_z >=  BBZ_ZMIN) | (rsi14 <= RSI_LONG_MAX)
    mask_short = (bb_z <= -BBZ_ZMIN) | (rsi14 >= RSI_SHORT_MIN)
    atr_ratio = (px["atr14"] / (px["close"] + 1e-12)).fillna(0)
    ret1_abs  = px["close"].pct_change(1).abs().fillna(0)
    liq_mask  = (atr_ratio >= 0.003) & (ret1_abs <= 0.01)
    return mask_long, mask_short, liq_mask, bb_z, rsi14, atr_ratio, ret1_abs

def build_features_h16(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    c = df['close']
    v = df['volume']
    df['ret_1'] = c.pct_change(1)
    df['ret_4'] = c.pct_change(4)
    df['ret_16'] = c.pct_change(16)
    df['ret_32'] = c.pct_change(32)
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std(ddof=0)
    df['bb_z20'] = (c - ma20) / (std20 + 1e-12)
    df['rv_16'] = df['ret_1'].rolling(16).std(ddof=0)
    df['ema_fast'] = _ema(c, 12); df['ema_slow'] = _ema(c, 48)
    df['ema_fast_dist'] = (c - df['ema_fast']) / (abs(df['ema_fast']) + 1e-12)
    df['ema_fast_slow_gap'] = (df['ema_fast'] - df['ema_slow']) / (abs(df['ema_slow']) + 1e-12)
    df['mom_ratio'] = (c - df['ema_fast']) / (abs(df['ema_fast'] - df['ema_slow']) + 1e-12)
    df['vol_chg'] = v.pct_change().fillna(0)
    df['vol_z48'] = (v - v.rolling(48).mean()) / (v.rolling(48).std(ddof=0) + 1e-12)
    df['atr14'] = _atr(df, 14)
    df['atr_ratio'] = df['atr14'] / (c + 1e-12)
    df['rsi14_15m'] = _rsi(c, 14)
    df['rsi_slope'] = df['rsi14_15m'].diff().rolling(4).mean()
    df['vol_ma_24h'] = v.rolling(96).mean()
    di = df.set_index("timestamp")
    c_1h = di["close"].resample("1h").last().dropna()
    c_4h = di["close"].resample("4h").last().dropna()
    rsi_1h = _rsi(c_1h, 14); rsi_4h = _rsi(c_4h, 14)
    df = pd.merge_asof(df, pd.DataFrame({"timestamp": rsi_1h.index, "rsi14_1h": rsi_1h.values}), on="timestamp", direction="backward")
    df = pd.merge_asof(df, pd.DataFrame({"timestamp": rsi_4h.index, "rsi14_4h": rsi_4h.values}), on="timestamp", direction="backward")
    def _slope_log(s, w):
        x = np.arange(w, dtype=float)
        def _f(win):
            if win.isna().any(): return np.nan
            y = np.log(win.values + 1e-12)
            vx = x - x.mean()
            return (vx * (y - y.mean())).sum() / (vx**2).sum()
        return s.rolling(w).apply(_f, raw=False) / (w + 1e-12)
    df['slope_log_8'] = _slope_log(c, 8)
    df['slope_log_16'] = _slope_log(c, 16)
    df['slope_log_32'] = _slope_log(c, 32)
    return df

def run_audit():
    df_all = pd.read_csv(DATA_PATH)
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], utc=True)
    start, end = pd.to_datetime("2026-02-21", utc=True), pd.to_datetime("2026-03-22", utc=True)

    symbol_dfs = {}
    for s in df_all['symbol'].unique():
        df = build_features_h16(df_all[df_all['symbol']==s])
        symbol_dfs[s] = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].reset_index(drop=True)

    all_timestamps = sorted(df_all[(df_all['timestamp'] >= start) & (df_all['timestamp'] <= end)]['timestamp'].unique())
    trades = []
    active_trade = None

    dead_zone_logs = []

    for t in all_timestamps:
        if active_trade:
            s = active_trade['symbol']
            row = symbol_dfs[s][symbol_dfs[s]['timestamp'] == t]
            if row.empty: continue
            row = row.iloc[0]
            active_trade['held'] += 1
            exit_p, reason = None, None
            if active_trade['side'] == 'LONG':
                if row['low'] <= active_trade['sl']: exit_p, reason = active_trade['sl'], "SL"
                elif row['high'] >= active_trade['tp']: exit_p, reason = active_trade['tp'], "TP"
            else:
                if row['high'] >= active_trade['sl']: exit_p, reason = active_trade['sl'], "SL"
                elif row['low'] <= active_trade['tp']: exit_p, reason = active_trade['tp'], "TP"
            if not reason and active_trade['held'] >= 16: exit_p, reason = row['close'], "TIMEOUT"
            if reason:
                ret = (exit_p - active_trade['entry_p'])/active_trade['entry_p'] if active_trade['side']=='LONG' else (active_trade['entry_p'] - exit_p)/active_trade['entry_p']
                active_trade['exit_t'] = t
                active_trade['pnl'] = ret - 0.0016
                active_trade['reason'] = reason
                trades.append(active_trade)
                active_trade = None
            continue

        candidates = []
        for s, df in symbol_dfs.items():
            row_idx = df.index[df['timestamp'] == t]
            if row_idx.empty: continue
            idx = row_idx[0]
            row = df.iloc[idx]

            X = scaler.transform(row[FEATURES].to_frame().T)
            prob = clf.predict_proba(X)[0]

            is_long = prob[1] >= TH_LONG
            is_short = prob[0] >= TH_SHORT

            mL, mS, mLiq, bbz, rsi, atr_r, ret1 = build_regime_masks(df)

            ema_ok = (row['close'] > row['ema_slow']) if is_long else (row['close'] < row['ema_slow'])
            vol_ok = row['volume'] > (row['vol_ma_24h'] * 2.0)
            rsi_ok = (row['rsi_slope'] > 2.0) if is_long else (row['rsi_slope'] < -2.0)

            # Dead Zone Logging (e.g., 3/21)
            if t.strftime('%Y-%m-%d') == '2026-03-21' and s == 'BTC/USDT':
                reasons = []
                if not is_long and not is_short: reasons.append(f"AI_Score_Low({max(prob):.3f})")
                if not vol_ok: reasons.append(f"Vol_Filter(Ratio={row['volume']/row['vol_ma_24h']:.2f})")
                if not rsi_ok: reasons.append(f"RSI_Slope_Filter({row['rsi_slope']:.2f})")
                if reasons:
                    dead_zone_logs.append(f"[{t}] BTC/USDT Blocked: {' & '.join(reasons)}")

            if not (is_long or is_short): continue

            regime_ok = mL[idx] if is_long else mS[idx]
            if regime_ok and mLiq[idx] and ema_ok and vol_ok and rsi_ok:
                candidates.append({
                    "symbol": s, "side": "LONG" if is_long else "SHORT", "entry_p": row['close'],
                    "sl": row['close'] - SL_L*row['atr14'] if is_long else row['close'] + SL_S*row['atr14'],
                    "tp": row['close'] + TP_L*row['atr14'] if is_long else row['close'] - TP_S*row['atr14'],
                    "held": 0, "entry_t": t, "score": prob[1] if is_long else prob[0],
                    "features": row.to_dict()
                })

        if candidates:
            active_trade = max(candidates, key=lambda x: x['score'])

    # Output Audit Trail
    print("# Iteration 99.1 Audit Trail")
    print("| Symbol | Entry Time | Exit Time | AI Score | PnL % | Reason |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for tr in trades:
        print(f"| {tr['symbol']} | {tr['entry_t']} | {tr['exit_t']} | {tr['score']:.3f} | {tr['pnl']:.2%} | {tr['reason']} |")

    # Logic Backtracking (3 trades)
    print("\n## Logic Backtracking (3 Selected Trades)")
    import random
    random.seed(42) # For reproducibility
    selected = random.sample(trades, min(3, len(trades)))
    for i, tr in enumerate(selected):
        f = tr['features']
        print(f"\n### Trade {i+1}: {tr['symbol']} @ {tr['entry_t']}")
        print(f"- AI Score: {tr['score']:.3f}")
        print(f"- RSI Slope: {f['rsi_slope']:.3f}")
        print(f"- EMA Slow: {f['ema_slow']:.2f} (Price: {tr['entry_p']:.2f})")
        print(f"- Vol MA 24H: {f['vol_ma_24h']:.2f} (Vol: {f['volume']:.2f})")
        print(f"- ATR Ratio: {f['atr_ratio']:.4f}")
        print(f"- BB Z-Score: {f['bb_z20']:.3f}")
        print(f"- RSI 15m: {f['rsi14_15m']:.1f}")

    # Dead Zone Analysis
    print("\n## Dead Zone Analysis (2026-03-21)")
    for log in dead_zone_logs[:10]:
        print(f"- {log}")

run_audit()
