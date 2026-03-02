from __future__ import annotations

import pandas as pd


UTC = "UTC"
LOCAL = "Asia/Taipei"


def to_utc_ts(ts) -> pd.Timestamp:
    t = pd.to_datetime(ts, errors="raise")
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize(LOCAL).tz_convert(UTC)
    return t.tz_convert(UTC)

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
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # ensure no extra 'timestamp' column remains and keep neutral index name
    if col in df.columns:
        df = df.drop(columns=[col])
    df.index.name = None
    return df

def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz=UTC)


# NEW: bulletproof wrapper—accepts None, str, naive, aware
def safe_ts_to_utc(ts) -> pd.Timestamp:
    if ts is None:
        return now_utc()
    return to_utc_ts(ts)


def ensure_aware_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if getattr(ts, "tzinfo", None) is None:
        return ts.tz_localize(UTC)
    return ts.tz_convert(UTC)

def last_closed_15m(now: pd.Timestamp | None = None) -> pd.Timestamp:
    if now is None:
        now = now_utc()
    return now.floor("15min")
