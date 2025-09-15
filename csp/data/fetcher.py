from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from csp.utils.framefix import safe_reset_index

from csp.utils.tz_safe import (
    normalize_df_to_utc,
    safe_ts_to_utc,
    now_utc as _now_utc,
    floor_utc,
)


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
    interval_td = pd.to_timedelta(interval)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True) + interval_td
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df = normalize_df_to_utc(df)
    print(f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}")
    assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"

    cutoff = floor_utc(_now_utc(), interval)
    df = df.loc[df.index <= cutoff]
    return df


def update_csv_with_latest(
    symbol: str,
    csv_path: str,
    interval: str = "15m",
    now_utc_ts: Optional[pd.Timestamp] = None,
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
    else:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = normalize_df_to_utc(df)
    print(f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}")
    assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"

    interval_td = pd.to_timedelta(interval)
    # 若呼叫端沒提供時間，取目前 UTC；避免名稱遮蔽工具函式
    now_ts = _now_utc() if now_utc_ts is None else safe_ts_to_utc(now_utc_ts)
    last_closed = floor_utc(now_ts, interval)

    last_ts = df.index[-1] if not df.empty else None
    if last_ts is None:
        days = int(os.getenv("DAYS", 30))
        start_dt = floor_utc(last_closed - pd.Timedelta(days=days), interval)
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

    df = pd.concat([df, new_df])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    cutoff = last_closed
    df = df.loc[df.index <= cutoff]

    appended = len(df) - before_len
    last_ts2 = df.index[-1] if not df.empty else None
    last_str = last_ts2.isoformat() if last_ts2 is not None else "none"
    print(f"[FETCH] {symbol} need={need} appended={appended} last_ts={last_str}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    safe_reset_index(df, name="timestamp", overwrite=True).to_csv(tmp, index=False)
    os.replace(tmp, path)
    time.sleep(0.25)
    return df


def fetch_inc(symbol: str, csv_path: str) -> dict:
    """
    讀 CSV 最後一根時間，向來源補齊至最新的「已收」一根，追加寫回。
    重用 update_csv_with_latest。
    """
    before = 0
    try:
        before = len(pd.read_csv(csv_path))
    except Exception:
        before = 0
    df = update_csv_with_latest(symbol, csv_path)
    appended = len(df) - before
    last_ts = df["timestamp"].iloc[-1].isoformat() if len(df) else "none"
    return {"ok": True, "mode": "inc", "appended": int(appended), "last_ts": last_ts}


def fetch_full(symbol: str, csv_path: str) -> dict:
    """
    從來源 full 下載該 symbol 的 15m 歷史到最新「已收」一根，覆蓋寫回 CSV。
    """
    days = int(os.getenv("DAYS", 30))
    interval = "15m"
    now_ts = floor_utc(_now_utc(), interval)
    start_dt = floor_utc(now_ts - pd.Timedelta(days=days), interval)
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(now_ts.timestamp() * 1000)
    df = fetch_klines(symbol, interval=interval, start_ts=start_ts, end_ts=end_ts)
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_reset_index(df, name="timestamp", overwrite=True).to_csv(path, index=False)
    last_ts = df.index[-1].isoformat() if len(df) else "none"
    print(f"[FETCH] {symbol} FULL rows={len(df)} last_ts={last_ts}")
    return {"ok": True, "mode": "full", "rows": int(len(df)), "last_ts": last_ts}
