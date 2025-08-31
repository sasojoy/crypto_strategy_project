import pandas as pd

UTC = "UTC"

def safe_ts_to_utc(ts):
    """Return a UTC-aware pandas.Timestamp from ts (scalar)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(UTC)
    return ts.tz_convert(UTC)

def safe_index_to_utc(idx):
    """Return a UTC-aware DatetimeIndex from ``idx``."""
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            return idx.tz_localize(UTC)
        return idx.tz_convert(UTC)
    # ``idx`` might be array-like or Series
    idx = pd.to_datetime(idx, errors="raise")
    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)
    if getattr(idx, "tz", None) is None:
        return idx.tz_localize(UTC)
    return idx.tz_convert(UTC)

def safe_series_to_utc(s):
    """Return a UTC-aware Series of datetimes."""
    # s may be datetime64[ns], object, etc.
    s = pd.to_datetime(s, utc=True, errors="raise")
    return s.dt.tz_convert(UTC)

def normalize_df_to_utc(df):
    """Normalize ``df`` so its index is UTC-aware.

    If a ``timestamp`` column exists, it will be converted to UTC and used as
    the index.  Otherwise, an existing DatetimeIndex (or one that can be parsed
    from the current index) will be converted.  A mirror ``timestamp`` column is
    guaranteed to exist.
    """
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(safe_ts_to_utc)
        df = df.set_index("timestamp", drop=True)
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = safe_index_to_utc(df.index)
        elif not isinstance(df.index, pd.RangeIndex):
            try:
                df.index = safe_index_to_utc(df.index)
            except Exception:
                pass
    if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = df.index
    return df.sort_index()

def now_utc():
    return pd.Timestamp.now(tz=UTC)

def floor_utc(ts, freq="15min"):
    return safe_ts_to_utc(ts).floor(freq)

def ceil_utc(ts, freq="15min"):
    return safe_ts_to_utc(ts).ceil(freq)
