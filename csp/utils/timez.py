from __future__ import annotations
import pandas as pd

UTC = "UTC"
LOCAL = "Asia/Taipei"

def to_utc_ts(ts: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.to_datetime(ts, errors="raise")
    if ts.tzinfo is None:
        return ts.tz_localize(LOCAL).tz_convert(UTC)
    return ts.tz_convert(UTC)

def ensure_utc_index(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns:
        ts = pd.to_datetime(df[col], utc=True)
        df = df.drop(columns=[col]).copy()
        df.index = ts
    else:
        idx = pd.to_datetime(df.index, errors="raise")
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize(UTC)
        else:
            idx = idx.tz_convert(UTC)
        df = df.copy()
        df.index = idx
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()

def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz=UTC)

def last_closed_15m(now: pd.Timestamp | None = None) -> pd.Timestamp:
    if now is None:
        now = now_utc()
    return now.floor("15min")
