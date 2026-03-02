from __future__ import annotations

import datetime as dt
from typing import List

import pandas as pd
import requests


INTERVAL_MAP = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
}


def _to_millis(ts: dt.datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(ts.timestamp() * 1000)


def fetch_binance_klines_public(
    symbol: str,
    start_time_utc: dt.datetime,
    end_time_utc: dt.datetime,
    interval: str = "15m",
    api_base: str = "https://api.binance.com",
    timeout: float = 10.0,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch klines from Binance public endpoint without API key.

    Returns a DataFrame with columns [open, high, low, close, volume] indexed by
    UTC timestamps.
    """

    if interval not in INTERVAL_MAP:
        raise ValueError(f"Unsupported interval: {interval}")

    url = f"{api_base}/api/v3/klines"
    st = start_time_utc
    frames: List[pd.DataFrame] = []
    step = dt.timedelta(seconds=INTERVAL_MAP[interval] * (limit - 1))

    while st < end_time_utc:
        en = min(st + step, end_time_utc)
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": _to_millis(st),
            "endTime": _to_millis(en),
            "limit": limit,
        }
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        raw = r.json()
        if not raw:
            break
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "_ct1",
                "_ct2",
                "_ct3",
                "_ct4",
                "_ct5",
                "_ct6",
            ],
        )
        df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.drop(columns=["open_time"]).set_index("timestamp").sort_index()
        frames.append(df)
        st = (
            df.index[-1].to_pydatetime().replace(tzinfo=dt.timezone.utc)
            + dt.timedelta(seconds=INTERVAL_MAP[interval])
        )

    if not frames:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype="float64",
        )

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out

