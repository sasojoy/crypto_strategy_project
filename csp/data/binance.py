from __future__ import annotations
import time
from typing import List
import pandas as pd
import requests

def _klines_to_df(data: list) -> pd.DataFrame:
    df = pd.DataFrame(
        data,
        columns=["open_time","open","high","low","close","volume",
                 "close_time","qav","trades","taker_base","taker_quote","ignore"]
    )
    df = df[["open_time","open","high","low","close","volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)

def fetch_latest_klines(symbol: str, interval: str = "15m", limit: int = 64,
                        api_base: str = "https://api.binance.com", timeout: int = 10) -> pd.DataFrame:
    url = f"{api_base}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    return _klines_to_df(data)

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
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    return _klines_to_df(all_rows)

def merge_history_and_live(hist_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    if live_df is None or live_df.empty:
        return hist_df
    df = pd.concat([hist_df, live_df], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df