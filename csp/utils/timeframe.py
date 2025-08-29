from .tz_safe import normalize_df_to_utc_index


def normalize_df_ts(df, ts_col="timestamp"):
    return normalize_df_to_utc_index(df, ts_col=ts_col)
