import pandas as pd
import numpy as np
from core.feature import add_features as _core_add_features   # 你線上版的特徵

# 與線上版本一致的參數
ATR_N = 14
ADX_N = 14

def _calc_atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    ds = df.copy()
    for col in ("high","low","close"):
        ds[col] = pd.to_numeric(ds[col], errors="coerce")
    ds = ds.dropna(subset=["high","low","close"])
    tr1 = (ds["high"] - ds["low"]).abs()
    tr2 = (ds["high"] - ds["close"].shift(1)).abs()
    tr3 = (ds["low"]  - ds["close"].shift(1)).abs()
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

def _compute_adx(df: pd.DataFrame, n: int = ADX_N) -> pd.Series:
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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    保留舊接口，但確保有：
      - EMA20, EMA50, ATR(14), ADX(14)
      - 並呼叫 core.feature.add_features() 補上模型需要的特徵
    """
    df = df.copy()
    for col in ("open","high","low","close","volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ATR14"] = _calc_atr(df, n=ATR_N)
    df["ADX14"] = _compute_adx(df, n=ADX_N)

    # 讓模型用的特徵與線上一致
    df_feat = _core_add_features(df.copy())

    # 確保 timestamp 保留
    if "timestamp" in df.columns and "timestamp" not in df_feat.columns:
        df_feat["timestamp"] = df["timestamp"]

    return df_feat.dropna().reset_index(drop=True)
