from __future__ import annotations
import pandas as pd
import numpy as np
from csp.utils.framefix import safe_reset_index
from csp.utils.diag import log_diag

"""
h16 特徵（以 15m 基礎 + 4H 聚合）
- 15m：EMA(9/21/50)、RSI、布林、K棒形狀（body/上下影線）
- 4H：ATR(14)、RSI(14)、EMA(21/50) 轉回 15m 時間軸（每個 4H 區間內 forward-fill）
- 標籤：未來 horizon 根 K 線的漲跌（二元分類；>0 為 1）
"""

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(n).mean()
    roll_down = pd.Series(down, index=series.index).rolling(n).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _ensure_time_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    """確保 DataFrame 具有 UTC 的 DatetimeIndex。

    - 若存在 'timestamp' 欄位，將其轉成 UTC 後設為 index。
    - 若已是 DatetimeIndex，則統一為 UTC 並命名為 'timestamp'。
    """
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        d = df.drop(columns=["timestamp"], errors="ignore").copy()
        d.index = ts
        d.index.name = "timestamp"
        return d

    if isinstance(df.index, pd.DatetimeIndex):
        d = df.copy()
        idx = d.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        d.index = idx
        d.index.name = "timestamp"
        return d

    raise KeyError("Input DataFrame must contain 'timestamp' column or DatetimeIndex")

def build_features_15m_4h(
    df15: pd.DataFrame,
    ema_windows=(9, 21, 50),
    rsi_window: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    atr_window: int = 14,
    h4_resample: str = "4H",
) -> pd.DataFrame:
    """
    傳入：含欄位 timestamp/open/high/low/close[/volume] 的 15m DataFrame，
          或已以 timestamp 作為 index 的 DataFrame。接受 naive 或 tz-aware 時間。
    回傳：加入 15m 與 4H 特徵後、對齊 15m 時間軸的 DataFrame（已去除起始 NaN）。
    """
    # Robustify: ensure 'timestamp' column exists and is UTC
    df = df15
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["timestamp"] = df.index
        else:
            df = df.reset_index(drop=False).rename(columns={"index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.set_index("timestamp")
    df = df.sort_index()

    # --- 15m features ---
    for w in ema_windows:
        df[f"ema_{w}"] = _ema(df["close"], w)
    df["rsi"] = _rsi(df["close"], rsi_window)

    m = df["close"].rolling(bb_window).mean()
    s = df["close"].rolling(bb_window).std()
    df["bb_mid"] = m
    df["bb_high"] = m + bb_std * s
    df["bb_low"] = m - bb_std * s

    # K棒形狀
    df["body"] = df["close"] - df["open"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1))
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"])

    # --- 4H resample features ---
    h4_rule = str(h4_resample).lower()  # 修正 FutureWarning：使用小寫 'h'
    volume_src = df.get("volume", pd.Series(index=df.index, dtype=float))

    h4 = pd.DataFrame({
        "open": df["open"].resample(h4_rule).first(),
        "high": df["high"].resample(h4_rule).max(),
        "low": df["low"].resample(h4_rule).min(),
        "close": df["close"].resample(h4_rule).last(),
        "volume": volume_src.resample(h4_rule).sum()
    })
    h4["atr_h4"] = _atr(h4, atr_window)
    h4["rsi_h4"] = _rsi(h4["close"], 14)
    h4["ema_h4_21"] = _ema(h4["close"], 21)
    h4["ema_h4_50"] = _ema(h4["close"], 50)

    # 將 4H 指標對齊回 15m（每個 4H 區間內 forward-fill）
    h4_to_15 = h4.reindex(df.index, method="ffill")

    # 合併：維持 15m 時間軸
    feats = pd.concat(
        [df, h4_to_15[["atr_h4", "rsi_h4", "ema_h4_21", "ema_h4_50"]]],
        axis=1
    )
    feats = safe_reset_index(feats, name="timestamp", overwrite=True)

    # 去掉起始 NaN（均線/布林/ATR 等產生的缺值）
    feats = feats.dropna()
    log_diag(
        f"about to reset_index: idx.name={feats.index.name}, cols={list(feats.columns)[:8]}... total_cols={len(feats.columns)}"
    )
    feats = feats.reset_index(drop=True)
    return feats

def make_labels(df: pd.DataFrame, horizon: int = 16) -> pd.Series:
    """
    以未來 horizon 根的收盤價相對報酬作為二元標籤：
      return = (close[t+h] - close[t]) / close[t]
      y = 1 if return > 0 else 0
    """
    future = df["close"].shift(-horizon)
    ret = (future - df["close"]) / df["close"]
    y = (ret > 0).astype(int)
    return y
