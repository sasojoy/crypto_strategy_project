from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


def fetch_klines(
    symbol: str,
    interval: str = "15m",
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    *,
    base_url: str = "https://api.binance.com",
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch klines from Binance REST API.

    Parameters
    ----------
    symbol: str
        Trading pair symbol like ``"BTCUSDT"``.
    interval: str
        Kline interval, e.g. ``"15m"``.
    start_ts: int | None
        Optional start timestamp in milliseconds.
    end_ts: int | None
        Optional end timestamp in milliseconds.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp, open, high, low, close, volume`` with
        ``timestamp`` being UTC-aware ``pd.Timestamp``.
    """

    url = f"{base_url}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 1000}
    if start_ts is not None:
        params["startTime"] = int(start_ts)
    if end_ts is not None:
        params["endTime"] = int(end_ts)

    data = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json()
            break
        except Exception:
            if attempt + 1 == max_retries:
                raise
            time.sleep(retry_delay)
    if not data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    cols = [
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
    ]
    df = pd.DataFrame(data, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Drop klines that are not yet closed
    interval_td = pd.to_timedelta(interval)
    cutoff = pd.Timestamp.utcnow().floor(interval_td)
    df = df[df["timestamp"] < cutoff]
    return df.reset_index(drop=True)


def update_csv_with_latest(
    symbol: str,
    csv_path: str,
    interval: str = "15m",
) -> pd.DataFrame:
    """Update local CSV with latest closed klines from Binance.

    This function reads existing CSV, fetches missing klines from Binance,
    appends them (dropping duplicates) and writes back atomically.
    It returns the updated DataFrame. If fetching fails, the original
    DataFrame is returned with ``df.attrs["stale"] = True``.
    """

    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    interval_td = pd.to_timedelta(interval)
    last_closed = pd.Timestamp.utcnow().floor(interval_td)

    last_ts = df["timestamp"].iloc[-1] if not df.empty else None
    if last_ts is None:
        days = int(os.getenv("DAYS", 30))
        start_dt = (last_closed - pd.Timedelta(days=days)).floor(interval_td)
    else:
        start_dt = last_ts + interval_td
    start_ts = int(start_dt.timestamp() * 1000)

    end_ts = int(last_closed.timestamp() * 1000)

    need = max(0, int((last_closed - start_dt) / interval_td))
    before_len = len(df)
    try:
        new_df = fetch_klines(symbol, interval=interval, start_ts=start_ts, end_ts=end_ts)
    except Exception as e:
        print(f"[WARN] fetch failed for {symbol}: {e}")
        df.attrs["stale"] = True
        return df

    df = pd.concat([df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    cutoff = pd.Timestamp.utcnow().floor(interval_td)
    df = df[df["timestamp"] < cutoff].reset_index(drop=True)

    appended = len(df) - before_len
    last_ts2 = df["timestamp"].iloc[-1] if not df.empty else None
    last_str = last_ts2.isoformat() if last_ts2 is not None else "none"
    print(f"[FETCH] {symbol} need={need} appended={appended} last_ts={last_str}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    # gentle rate limit
    time.sleep(0.25)
    return df
