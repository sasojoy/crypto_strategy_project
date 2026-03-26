import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime

# ===== 超參數（一致於線上 realtime 版本） =====
BASE_SL_LONG  = 1.75
BASE_TP_LONG  = 3.50
BASE_SL_SHORT = 1.50
BASE_TP_SHORT = 3.00

ATR_N             = 14
ATR_LOOKBACK_PCTL = 96
ATR_PCTL_LOW      = 0.30
ATR_PCTL_HIGH     = 0.70
ADX_N             = 14
ADX_TREND         = 20      # 強趨勢（ADX）
RANGING_ADX       = 12      # 震盪（ADX）
MIN_HOLD_BARS     = 2
MAX_HOLD_BARS     = 16      # 4 小時

EMA_GAP_STRONG = 0.0025     # 0.25%
EMA_GAP_RANGE  = 0.0015     # 0.15%
EMA_SAME_SIDE_K = 4

# ============== 指標（與線上版等價） ==============
def calc_atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    ds = df.copy()
    for col in ("high","low","close"):
        ds[col] = pd.to_numeric(ds[col], errors="coerce")
    ds = ds.dropna(subset=["high","low","close"])
    tr1 = (ds["high"] - ds["low"]).abs()
    tr2 = (ds["high"] - ds["close"].shift(1)).abs()
    tr3 = (ds["low"]  - ds["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    if not np.isfinite(atr.iloc[-1]):
        pct_std = ds["close"].pct_change().rolling(n, min_periods=2).std()
        atr_est = (pct_std * ds["close"]).rolling(n, min_periods=1).mean()
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

def market_state_from_slice(df_slice: pd.DataFrame):
    ds = df_slice.copy()
    for col in ("high","low","close"):
        ds[col] = pd.to_numeric(ds[col], errors="coerce")
    ds = ds.dropna(subset=["high","low","close"])

    if len(ds) < max(ATR_N, 5):
        return dict(
            atr_now=np.nan, adx_now=0.0,
            high_vol=False, low_vol=False,
            strong_trend=False, ranging=True,
            atr_p30=np.nan, atr_p70=np.nan,
            ema_gap_pct=0.0
        )

    ds["ATR"] = calc_atr(ds, ATR_N)
    ds["EMA20"] = ds["close"].ewm(span=20, adjust=False).mean()
    ds["EMA50"] = ds["close"].ewm(span=50, adjust=False).mean()
    ds["ADX"] = compute_adx(ds, ADX_N)

    atr_now = float(ds["ATR"].iloc[-1])
    adx_now = float(ds["ADX"].iloc[-1])
    if not np.isfinite(adx_now): adx_now = 0.0

    tail = ds["ATR"].tail(ATR_LOOKBACK_PCTL).dropna()
    if len(tail) < 10:
        tail = ds["ATR"].dropna().tail(ATR_LOOKBACK_PCTL)
    atr_p30 = float(np.percentile(tail, ATR_PCTL_LOW*100)) if len(tail) else atr_now
    atr_p70 = float(np.percentile(tail, ATR_PCTL_HIGH*100)) if len(tail) else atr_now

    high_vol = (np.isfinite(atr_now) and np.isfinite(atr_p70) and atr_now > atr_p70)
    low_vol  = (np.isfinite(atr_now) and np.isfinite(atr_p30) and atr_now < atr_p30)

    price_now = float(ds["close"].iloc[-1])
    ema20_now = float(ds["EMA20"].iloc[-1])
    ema50_now = float(ds["EMA50"].iloc[-1])
    ema_gap_pct = abs(ema20_now - ema50_now) / price_now if price_now > 0 else 0.0
    ema_side = (ds["EMA20"] > ds["EMA50"]).astype(int) - (ds["EMA20"] < ds["EMA50"]).astype(int)
    same_side_lastk = int(abs(ema_side.tail(EMA_SAME_SIDE_K).sum())) == EMA_SAME_SIDE_K

    adx_strong = adx_now >= ADX_TREND
    adx_range  = adx_now <= RANGING_ADX
    strong_trend = bool(adx_strong or ((ema_gap_pct >= EMA_GAP_STRONG) and same_side_lastk))
    ranging      = bool(adx_range  or  (ema_gap_pct <  EMA_GAP_RANGE))

    return dict(
        atr_now=atr_now, adx_now=adx_now,
        high_vol=high_vol, low_vol=low_vol,
        strong_trend=strong_trend, ranging=ranging,
        atr_p30=atr_p30, atr_p70=atr_p70,
        ema_gap_pct=ema_gap_pct
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
        sl = round(entry - sl_mult * atr, 2)
        tp = round(entry + tp_mult * atr, 2)
    else:
        sl = round(entry + sl_mult * atr, 2)
        tp = round(entry - tp_mult * atr, 2)
    return sl, tp

def tighten_stop_only(side: str, current_sl: float, entry_price: float, atr_now: float):
    if side == "LONG":
        proposed = round(entry_price - 1.2 * atr_now, 2)
        return max(current_sl, proposed)
    else:
        proposed = round(entry_price + 1.2 * atr_now, 2)
        return min(current_sl, proposed)

# ============== 模型工具 ==============
def _load_model_and_scaler(model_path, scaler_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def _infer_prob_row(model, scaler, feat_row):
    # 保持 DataFrame（帶欄位名），避免 sklearn 警告
    X = feat_row[scaler.feature_names_in_]
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[0]
    return float(p[0]), float(p[1])  # down_prob, up_prob

# ============== 主要對外函式（名稱不變） ==============
def run_classification_strategy(
    df: pd.DataFrame,
    model_path: str = "models/xgb_cls_model.json",
    scaler_path: str = "models/xgb_cls_scaler.joblib",
    prob_threshold: float = 0.60,
    fee_bps: float = 8.0,
    slip_bps: float = 0.0
):
    """
    與你原本同名，但內部改為「動態 ATR/TP/SL 檢核 + 4 小時最長持倉」。
    需要 df 已有欄位：timestamp/open/high/low/close/volume 與 add_indicators 產出的技術指標列
    並且 add_indicators 要呼叫 core.feature.add_features（或內含相同特徵），確保模型輸入齊全。
    """
    df_feat = df.copy().dropna().reset_index(drop=True)
    if df_feat.empty:
        raise RuntimeError("add_indicators 後為空，請檢查資料或特徵回看長度。")

    model, scaler = _load_model_and_scaler(model_path, scaler_path)

    # 特徵齊全檢查（以第一列為準）
    missing = [f for f in scaler.feature_names_in_ if f not in df_feat.columns]
    if missing:
        raise ValueError(f"回測特徵缺失：{missing[:10]} ...（共 {len(missing)} 項）。請確認 add_indicators / add_features 有產生這些欄位。")

    trades = []
    in_pos = False
    pos = {}

    fee = float(fee_bps) / 10000.0
    slip = float(slip_bps) / 10000.0
    thr  = float(prob_threshold)

    if "timestamp" not in df_feat.columns:
        raise ValueError("df 需包含 timestamp 欄位（datetime 可解析）")
    df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for i in range(len(df_feat)):
        t = df_feat["timestamp"].iloc[i]
        row = df[df["timestamp"] == t]
        if row.empty:
            continue
        idx = row.index[0]
        dslice = df.iloc[:idx+1]

        down_p, up_p = _infer_prob_row(model, scaler, df_feat.iloc[i])

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

            # 本根判斷 TP/SL（同根同時命中 → SL 先）
            hi = float(df.loc[idx, "high"]); lo = float(df.loc[idx, "low"])
            sl = pos["sl"]; tp = pos["tp"]
            if side=="LONG":
                hit_tp = hi >= tp; hit_sl = lo <= sl
            else:
                hit_tp = lo <= tp; hit_sl = hi >= sl

            exit_reason = None
            if hit_tp and hit_sl:
                exit_reason = "SL"
            elif hit_tp:
                exit_reason = "TP"
            elif hit_sl:
                exit_reason = "SL"
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
                in_pos = False
                pos = {}
                continue

            continue  # 未出場 → 下一根

        # 無持倉：看要不要進場
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

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        stats = dict(
            trade_count=0, win_rate=0.0, avg_return=0.0, total_return=0.0,
            avg_holding_minutes=0.0
        )
    else:
        stats = dict(
            trade_count = int(len(trades_df)),
            win_rate    = float((trades_df["return_pct"] > 0).mean()),
            avg_return  = float(trades_df["return_pct"].mean()),
            total_return= float(trades_df["return_pct"].sum()),
            avg_holding_minutes = float(trades_df["holding_minutes"].mean())
        )

    return trades_df, stats
