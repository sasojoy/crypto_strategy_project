from .tz_safe import (
    UTC,
    safe_ts_to_utc,
    safe_index_to_utc,
    safe_series_to_utc,
    normalize_df_to_utc_index,
    now_utc,
    floor_utc,
    ceil_utc,
)

# Backward-compatible wrappers

def ensure_utc_ts(ts):
    return safe_ts_to_utc(ts)

def ensure_utc_index(df, ts_col="timestamp"):
    return normalize_df_to_utc_index(df, ts_col=ts_col)

def floor_to(ts, freq="15min"):
    return floor_utc(ts, freq)

def ceil_to(ts, freq="15min"):
    return ceil_utc(ts, freq)
