from __future__ import annotations
import pandas as pd
from datetime import datetime, time
from zoneinfo import ZoneInfo

TZ_LOCAL = ZoneInfo("Asia/Taipei")
TZ_UTC = ZoneInfo("UTC")

def parse_date_local(d: str) -> datetime:
    d = d.strip()
    dt = datetime.strptime(d, "%Y-%m-%d")
    return datetime.combine(dt.date(), time(0, 0, 0), tzinfo=TZ_LOCAL)

def to_utc_start(dt_local: datetime) -> pd.Timestamp:
    return pd.Timestamp(dt_local.astimezone(TZ_UTC))

def to_utc_end_inclusive(dt_local: datetime) -> pd.Timestamp:
    end_local = datetime.combine(dt_local.date(), time(23, 59, 59, 999999), tzinfo=TZ_LOCAL)
    return pd.Timestamp(end_local.astimezone(TZ_UTC))

def _idx_min_max_utc(df_index: pd.DatetimeIndex):
    if isinstance(df_index, pd.DatetimeIndex) and df_index.tz is not None:
        idx_min = pd.Timestamp(df_index.min()).tz_convert(TZ_UTC)
        idx_max = pd.Timestamp(df_index.max()).tz_convert(TZ_UTC)
    else:
        idx_min = pd.Timestamp(df_index.min(), tz=TZ_UTC)
        idx_max = pd.Timestamp(df_index.max(), tz=TZ_UTC)
    return idx_min, idx_max

def resolve_time_range_like(args, df_index: pd.DatetimeIndex):
    """
    支援物件/字典/命名參數樣式，抽出 start/end/days。
    優先順序：start/end > days > 全量。
    """
    if isinstance(args, dict):
        start = args.get("start"); end = args.get("end"); days = args.get("days")
    else:
        start = getattr(args, "start", None)
        end = getattr(args, "end", None)
        days = getattr(args, "days", None)

    idx_min, idx_max = _idx_min_max_utc(df_index)

    if start or end:
        s_utc = to_utc_start(parse_date_local(start)) if start else idx_min
        e_utc = to_utc_end_inclusive(parse_date_local(end)) if end else idx_max
        return max(s_utc, idx_min), min(e_utc, idx_max)

    if days:
        e_utc = idx_max
        s_utc = e_utc - pd.Timedelta(days=int(days))
        return max(s_utc, idx_min), e_utc

    return idx_min, idx_max

def slice_by_utc(df: pd.DataFrame, col="timestamp", start_utc=None, end_utc=None) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df[col], utc=True)
        df = df.set_index(ts)
    return df.loc[(df.index >= start_utc) & (df.index <= end_utc)].copy()
