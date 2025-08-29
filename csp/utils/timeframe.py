import pandas as pd

UTC = "UTC"

def normalize_df_ts(df, ts_col="timestamp"):
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col, drop=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="raise")
    df.index = df.index.tz_localize(UTC) if df.index.tz is None else df.index.tz_convert(UTC)
    if "timestamp" not in df.columns:
        df["timestamp"] = df.index
    return df.sort_index()
