import pandas as pd

UTC = "UTC"

def safe_ts_to_utc(ts):
    """Return a UTC-aware pandas.Timestamp from ts (scalar)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(UTC)
    return ts.tz_convert(UTC)

def safe_index_to_utc(idx):
    """Return a UTC-aware DatetimeIndex from idx."""
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            return idx.tz_localize(UTC)
        return idx.tz_convert(UTC)
    idx = pd.to_datetime(idx, utc=True, errors="raise")
    return pd.DatetimeIndex(idx, tz=UTC)

def safe_series_to_utc(s):
    """Return a UTC-aware Series of datetimes."""
    # s may be datetime64[ns], object, etc.
    s = pd.to_datetime(s, utc=True, errors="raise")
    return s.dt.tz_convert(UTC)

def normalize_df_to_utc_index(df, ts_col="timestamp"):
    """
    Ensure df has a UTC DatetimeIndex using ts_col if present.
    Also mirror a 'timestamp' column for legacy code.
    """
    if ts_col and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="raise")
        df = df.set_index(ts_col, drop=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="raise")
    # Make index UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize(UTC)
    else:
        df.index = df.index.tz_convert(UTC)
    if "timestamp" not in df.columns:
        df["timestamp"] = df.index
    return df.sort_index()

def now_utc():
    return pd.Timestamp.now(tz=UTC)

def floor_utc(ts, freq="15min"):
    return safe_ts_to_utc(ts).floor(freq)

def ceil_utc(ts, freq="15min"):
    return safe_ts_to_utc(ts).ceil(freq)
