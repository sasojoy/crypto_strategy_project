from __future__ import annotations
import time
from typing import List
import pandas as pd
import requests

from csp.utils.tz_safe import (
    normalize_df_to_utc,
    safe_ts_to_utc,
    now_utc,
    floor_utc,
)


INTERVAL_MIN = 15


def _klines_to_df(data: list, interval: str = "15m") -> pd.DataFrame:
    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    # Align timestamp to exact bar close time (open_time + interval)
    interval_td = pd.to_timedelta(interval)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True) + interval_td
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = normalize_df_to_utc(df)
    print(f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}")
    assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"
    return df

def fetch_latest_klines(
    symbol: str,
    interval: str = "15m",
    limit: int = 64,
    *,
    end_time: int | None = None,
    api_base: str = "https://api.binance.com",
    timeout: int = 10,
) -> pd.DataFrame:
    """Fetch latest klines with optional ``end_time`` (ms)."""
    url = f"{api_base}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    if end_time is not None:
        params["endTime"] = int(end_time)
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    return _klines_to_df(data, interval=interval)

def fetch_klines_range(symbol: str, interval: str, start_ts_ms: int, end_ts_ms: int,
                       api_base: str = "https://api.binance.com", timeout: int = 10, max_retries: int = 3) -> pd.DataFrame:
    """完整區間抓取（自動分頁，每頁最多 1000 根）。start/end 皆為 UTC 毫秒。"""
    url = f"{api_base}/api/v3/klines"
    all_rows: List[list] = []
    cur = int(start_ts_ms)
    while cur < end_ts_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ts_ms,
            "limit": 1000
        }
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, timeout=timeout)
                r.raise_for_status()
                chunk = r.json()
                break
            except Exception:
                if attempt + 1 == max_retries:
                    raise
                time.sleep(1.0)
        if not chunk:
            break
        all_rows.extend(chunk)
        last_close_time = int(chunk[-1][6])
        next_open = last_close_time + 1
        if next_open <= cur:
            break
        cur = next_open
        time.sleep(0.05)
    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    return _klines_to_df(all_rows, interval=interval)

def merge_history_and_live(hist_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    if live_df is None or live_df.empty:
        return hist_df
    df = pd.concat([hist_df, live_df])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df
