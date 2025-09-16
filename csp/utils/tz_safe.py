import pandas as pd

from .timez import UTC


def now_utc() -> pd.Timestamp:
    """Return the current UTC timestamp."""

    return pd.Timestamp.now(tz=UTC)


def safe_ts_to_utc(ts) -> pd.Timestamp:
    """Convert ``ts`` to a timezone-aware UTC ``Timestamp``."""

    if ts is None:
        return now_utc()

    if isinstance(ts, pd.Timestamp):
        if pd.isna(ts):
            return now_utc()
        t = ts
    else:
        if pd.isna(ts):
            return now_utc()
        t = pd.Timestamp(ts)
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize(UTC)
    return t.tz_convert(UTC)


def interval_to_pandas_freq(interval: str) -> str:
    """Normalise common interval strings (e.g. ``"15m"``) to pandas freq."""

    s = str(interval).strip()
    if not s:
        return s
    sl = s.lower()
    try:
        if sl.endswith("m") and not sl.endswith("min"):
            n = int(sl[:-1] or "1")
            return f"{n}min"
        if sl.endswith("h"):
            n = int(sl[:-1] or "1")
            return f"{n}H"
        if sl.endswith("d"):
            n = int(sl[:-1] or "1")
            return f"{n}D"
    except ValueError:
        return s

    if sl.endswith("min"):
        head = sl[:-3] or "1"
        try:
            return f"{int(head)}min"
        except ValueError:
            return s
    return s


def safe_index_to_utc(idx):
    """Return a UTC-aware ``DatetimeIndex`` from ``idx``."""

    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            return idx.tz_localize(UTC)
        return idx.tz_convert(UTC)
    idx = pd.to_datetime(idx, errors="raise")
    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)
    if getattr(idx, "tz", None) is None:
        return idx.tz_localize(UTC)
    return idx.tz_convert(UTC)


def safe_series_to_utc(s):
    """Return a UTC-aware ``Series`` of datetimes."""

    s = pd.to_datetime(s, utc=True, errors="raise")
    return s.dt.tz_convert(UTC)


def normalize_df_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``df`` has a UTC ``DatetimeIndex`` and mirror ``timestamp`` column."""

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


def floor_utc(ts, freq_or_interval="15min"):
    return safe_ts_to_utc(ts).floor(interval_to_pandas_freq(freq_or_interval))


def ceil_utc(ts, freq_or_interval="15min"):
    return safe_ts_to_utc(ts).ceil(interval_to_pandas_freq(freq_or_interval))


def round_utc(ts, freq_or_interval="15min"):
    return safe_ts_to_utc(ts).round(interval_to_pandas_freq(freq_or_interval))
