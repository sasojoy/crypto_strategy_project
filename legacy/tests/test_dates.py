import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from csp.utils.dates import (
    parse_date_local,
    to_utc_start,
    to_utc_end_inclusive,
    resolve_time_range_like,
    slice_by_utc,
    TZ_LOCAL,
    TZ_UTC,
)


def test_parse_date_local_returns_local_midnight():
    dt = parse_date_local("2024-01-15")
    assert dt.tzinfo == TZ_LOCAL
    assert dt.hour == 0 and dt.minute == 0 and dt.second == 0


def test_to_utc_conversions():
    dt_local = parse_date_local("2024-01-15")
    start_utc = to_utc_start(dt_local)
    end_utc = to_utc_end_inclusive(dt_local)
    assert start_utc.tz == TZ_UTC
    assert end_utc.tz == TZ_UTC
    assert start_utc.hour == 16
    assert end_utc.hour == 15
    assert end_utc.minute == 59
    assert end_utc.second == 59


def test_resolve_time_range_like_days_and_slice():
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    start, end = resolve_time_range_like({"days": 5}, idx)
    assert start == pd.Timestamp("2024-01-05", tz="UTC")
    assert end == pd.Timestamp("2024-01-10", tz="UTC")

    df = pd.DataFrame({
        "timestamp": idx,
        "value": range(10),
    })
    sliced = slice_by_utc(df, start_utc=start, end_utc=end)
    assert sliced["value"].tolist() == [4, 5, 6, 7, 8, 9]
