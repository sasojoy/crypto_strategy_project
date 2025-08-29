import pandas as pd
import numpy as np

UTC = "UTC"

def ensure_utc_index(df, ts_col="timestamp"):
    """
    Ensure df.index is UTC tz-aware, using ts_col if provided.
    - If ts_col exists, parse with utc=True then set_index(ts_col).
    - If index is datetime-like and tz-naive, tz_localize('UTC').
    - If index is tz-aware, convert to UTC.
    """
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col, drop=True)
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize(UTC)
    else:
        df.index = df.index.tz_convert(UTC)
    return df.sort_index()

def ensure_utc_ts(ts):
    """Return a UTC-aware pandas Timestamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(UTC)
    return ts.tz_convert(UTC)

def now_utc():
    return pd.Timestamp.now(tz=UTC)

def floor_to(ts, freq="15min"):
    ts = ensure_utc_ts(ts)
    return ts.floor(freq)

def ceil_to(ts, freq="15min"):
    ts = ensure_utc_ts(ts)
    return ts.ceil(freq)
