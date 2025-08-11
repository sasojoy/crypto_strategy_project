# -*- coding: utf-8 -*-
"""
feature_engineering_h16.py
15m K線 + 4小時(=16根) 預測的特徵工程模組
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

# ---------- helpers ----------
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(window).mean()
    dn = (-d.clip(upper=0)).rolling(window).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100 / (1 + rs))

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def _slope_log(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    def _fit(s: pd.Series):
        if s.isna().any(): return np.nan
        y = np.log(s.values + 1e-12)
        vx = x - x.mean()
        b = (vx * (y - y.mean())).sum() / (vx**2).sum()
        return b
    return series.rolling(window).apply(_fit, raw=False) / (window + 1e-12)

def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0)
    return (s - m) / (sd + 1e-12)

def _merge_tf_feature(df15: pd.DataFrame, feat: pd.Series, on_col: str = "timestamp") -> pd.Series:
    aux = pd.DataFrame({on_col: feat.index, "_v": feat.values}).dropna()
    return pd.merge_asof(
        df15[[on_col]].sort_values(on_col),
        aux.sort_values(on_col),
        on=on_col, direction="backward"
    )["_v"]

# ---------- main features ----------
def build_features_h16(df15: pd.DataFrame, horizon_bars: int = 16):
    df = df15.copy()

    # 時間欄處理（修正 tz-aware dtype 導致的 issubdtype 問題）
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)

    o,h,l,c,v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # returns / momentum
    df["ret_1"]  = c.pct_change(1)
    df["ret_4"]  = c.pct_change(4)
    df["ret_16"] = c.pct_change(16)
    df["ret_32"] = c.pct_change(32)

    # displacement & volatility
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std(ddof=0)
    df["bb_z20"] = (c - ma20) / (std20 + 1e-12)
    df["rv_16"]  = df["ret_1"].rolling(16).std(ddof=0)

    # trend slopes (log price)
    df["slope_log_8"]  = _slope_log(c, 8)
    df["slope_log_16"] = _slope_log(c, 16)
    df["slope_log_32"] = _slope_log(c, 32)

    # EMA structure
    ema_fast = _ema(c, 12)
    ema_slow = _ema(c, 48)
    df["ema_fast_dist"] = (c - ema_fast) / (abs(ema_fast) + 1e-12)
    df["ema_fast_slow_gap"] = (ema_fast - ema_slow) / (abs(ema_slow) + 1e-12)
    df["mom_ratio"] = (c - ema_fast) / (abs(ema_fast - ema_slow) + 1e-12)

    # volume
    df["vol_chg"] = v.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df["vol_z48"] = _zscore(v, 48)

    # volatility
    df["atr14"] = _atr(df, 14)
    df["atr_ratio"] = df["atr14"] / (c + 1e-12)

    # RSI 15m
    df["rsi14_15m"] = _rsi(c, 14)

    # cross-timeframe RSI (1h, 4h) → align back to 15m
    dfi = df.set_index("timestamp")
    c_1h = dfi["close"].resample("1h").last().dropna()
    c_4h = dfi["close"].resample("4h").last().dropna()
    rsi_1h = _rsi(c_1h, 14)
    rsi_4h = _rsi(c_4h, 14)
    df["rsi14_1h"] = _merge_tf_feature(df, rsi_1h)
    df["rsi14_4h"] = _merge_tf_feature(df, rsi_4h)

    # target
    future_c = c.shift(-horizon_bars)
    df["y_up"] = (future_c > c).astype(int)

    feats = [
        "ret_1","ret_4","ret_16","ret_32",
        "bb_z20","rv_16","slope_log_8","slope_log_16","slope_log_32",
        "ema_fast_dist","ema_fast_slow_gap","mom_ratio",
        "vol_chg","vol_z48",
        "atr14","atr_ratio",
        "rsi14_15m","rsi14_1h","rsi14_4h",
    ]

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df, feats

def build_regime_masks_h16(px: pd.DataFrame, bbz_zmin: float = 0.25,
                           rsi_long_max: float = 35, rsi_short_min: float = 65):
    c = px["close"]
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std(ddof=0)
    bb_z = (c - ma20) / (std20 + 1e-12)
    d = c.diff()
    rs = (d.clip(lower=0).rolling(14).mean() /
          ((-d.clip(upper=0)).rolling(14).mean() + 1e-12))
    rsi14 = 100 - (100/(1+rs))
    bb_z = bb_z.fillna(0); rsi14 = rsi14.fillna(50)

    mask_long  = (bb_z >=  bbz_zmin) | (rsi14 <= rsi_long_max)
    mask_short = (bb_z <= -bbz_zmin) | (rsi14 >= rsi_short_min)

    atr_ratio = (px["atr14"] / (px["close"] + 1e-12)).fillna(0)
    ret1_abs  = px["close"].pct_change(1).abs().fillna(0)
    liq_mask  = (atr_ratio >= 0.003) & (ret1_abs <= 0.01)

    mask_long  = (mask_long.values  & liq_mask.values)
    mask_short = (mask_short.values & liq_mask.values)
    return mask_long, mask_short
