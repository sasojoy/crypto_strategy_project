from .tz_safe import normalize_df_to_utc


def normalize_df_ts(df, ts_col="timestamp"):
    if ts_col != "timestamp" and ts_col in df.columns:
        df = df.rename(columns={ts_col: "timestamp"})
    return normalize_df_to_utc(df)
