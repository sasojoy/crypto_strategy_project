import pandas as pd

from csp.utils.time import safe_ts_to_utc, now_utc
from .timez import UTC


def _to_pandas_freq(freq_or_interval: str) -> str:
    """Normalize common interval strings to pandas frequency codes.

    Exchanges often use shorthand like ``"15m"`` or ``"1h"`` to denote time
    intervals.  Pandas expects explicit frequency strings such as ``"15min"`` or
    ``"1H"``.  This helper converts the common abbreviations into the formats
    pandas understands and normalises minute aliases.
    """

    s = str(freq_or_interval).strip()
    if not s:
        return s
    sl = s.lower()
    try:
        if sl.endswith("m") and not sl.endswith("min"):
            n = int(sl[:-1] or "1")
            # Use ``min`` to avoid ``m`` being interpreted as month-end
            return f"{n}min"
        if sl.endswith("h"):
            n = int(sl[:-1] or "1")
            return f"{n}H"
        if sl.endswith("d"):
            n = int(sl[:-1] or "1")
            return f"{n}D"
    except ValueError:
        # Fall back to original string if parsing the integer fails
        pass

    # If already a valid frequency, normalise minute alias and upper-case
    return s.replace("min", "T").upper() if sl.endswith("min") else s.upper()

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

def floor_utc(ts, freq_or_interval="15min"):
    return safe_ts_to_utc(ts).floor(_to_pandas_freq(freq_or_interval))


def ceil_utc(ts, freq_or_interval="15min"):
    return safe_ts_to_utc(ts).ceil(_to_pandas_freq(freq_or_interval))


def round_utc(ts, freq_or_interval="15min"):
    return safe_ts_to_utc(ts).round(_to_pandas_freq(freq_or_interval))
