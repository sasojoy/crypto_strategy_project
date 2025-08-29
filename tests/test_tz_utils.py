import pandas as pd
from csp.utils.tz_safe import normalize_df_to_utc_index, UTC


def test_ensure_utc_index_naive_and_tw():
    df_naive = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:15:00"],
        "open": [1, 2],
        "high": [1, 2],
        "low": [1, 2],
        "close": [1, 2],
    })
    out1 = normalize_df_to_utc_index(df_naive, ts_col="timestamp")
    assert str(out1.index.tz) == UTC

    df_tw = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 08:00:00", "2024-01-01 08:15:00"]).tz_localize("Asia/Taipei"),
        "open": [1, 2],
        "high": [1, 2],
        "low": [1, 2],
        "close": [1, 2],
    })
    out2 = normalize_df_to_utc_index(df_tw, ts_col="timestamp")
    assert str(out2.index.tz) == UTC
