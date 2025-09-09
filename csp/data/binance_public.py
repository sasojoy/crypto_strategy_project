from __future__ import annotations

import time
from typing import Optional, List, Tuple
import requests
import pandas as pd

def _to_millis(ts) -> int:
    """接受 pandas Timestamp/str/int，轉成毫秒 UNIX。"""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return int(ts)
    # pandas.Timestamp 或可被 pandas 解析的字串
    return int(pd.Timestamp(ts, tz="UTC").value // 10**6)

def fetch_klines(
    symbol: str,
    interval: str,
    end_ts_utc,                     # pandas.Timestamp/str/int
    *,
    base_url: str = "https://api.binance.com",
    limit: int = 500,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    透過 Binance 公開 API 取得最多 `limit` 根 K 線（以 endTime 截止，往回取）。
    免金鑰、只讀公開端點。
    回傳 DataFrame(index=close_time[ns, tz=UTC], columns=[open,high,low,close,volume]).
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
    return df[["open","high","low","close","volume","open_time"]]
