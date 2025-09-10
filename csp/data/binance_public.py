from __future__ import annotations

import time
from typing import Optional, List, Tuple
import requests
import pandas as pd
import numpy as np


def _to_millis(ts) -> int:
    """Convert various timestamp inputs to milliseconds since epoch (UTC).

    Accepts None, int/float (incl. numpy scalars), or anything parseable by
    :func:`pandas.Timestamp`. Naive datetimes are treated as UTC, while
    timezone-aware objects are converted to UTC.
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return int(ts)

    t = pd.Timestamp(ts)
    if t.tzinfo is None or t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.value // 10**6)

def fetch_klines(
    symbol: str,
    interval: str,
    end_ts_utc: Optional[object] = None,  # pandas.Timestamp/str/int
    *,
    base_url: str = "https://api.binance.com",
    limit: int = 500,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    透過 Binance 公開 API 取得最多 `limit` 根 K 線（以 endTime 截止，往回取）。
    免金鑰、只讀公開端點。
    回傳 DataFrame(index=close_time[ns, tz=UTC], columns=[open,high,low,close,volume]).
    若未提供 `end_ts_utc` 則以當下時間為截止點。
    """
    end_ms = _to_millis(end_ts_utc)
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(max(int(limit), 1), 1000),
    }
    if end_ms:
        params["endTime"] = end_ms

    s = session or requests.Session()
    url = f"{base_url}/api/v3/klines"
    resp = s.get(url, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()  # list of lists

    if not raw:
        return pd.DataFrame(
            columns=["open","high","low","close","volume","open_time","close_time"]
        ).set_index(pd.DatetimeIndex([], tz="UTC"))

    # Binance kline fields:
    # 0 openTime,1 open,2 high,3 low,4 close,5 volume,6 closeTime,7 quoteAssetVolume,
    # 8 numberOfTrades,9 takerBuyBaseVolume,10 takerBuyQuoteVolume,11 ignore
    rows = []
    for r in raw:
        rows.append({
            "open_time":  pd.to_datetime(r[0], unit="ms", utc=True),
            "open":       float(r[1]),
            "high":       float(r[2]),
            "low":        float(r[3]),
            "close":      float(r[4]),
            "volume":     float(r[5]),
            "close_time": pd.to_datetime(r[6], unit="ms", utc=True),
        })
    df = pd.DataFrame(rows)
    df = df.set_index("close_time").sort_index()
    df.index.name = None
    df["timestamp"] = df.index
    return df[["timestamp","open","high","low","close","volume","open_time"]]
