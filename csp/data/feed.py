from __future__ import annotations

from typing import Optional, Callable
import os
import pandas as pd
import numpy as np
import logging
import time

from . import binance_public

log = logging.getLogger(__name__)


def _get_fetch_fn(cfg) -> Optional[Callable]:
    mode = (cfg.get("fetch", {}) or {}).get("mode", "csv_only")
    if mode in ("none", "csv_only", None):
        log.debug("fetch disabled (csv_only/none)")
        return None
    if mode == "public":
        fcfg = cfg.get("fetch", {}) or {}
        base_url = fcfg.get("base_url", "https://api.binance.com")
        interval = fcfg.get("interval", "15m")
        batch_limit = int(fcfg.get("batch_limit", 500))
        def _fn(symbol: str, end_ts_utc):
            return binance_public.fetch_klines(
                symbol=symbol,
                interval=interval,
                end_ts_utc=end_ts_utc,
                base_url=base_url,
                limit=batch_limit,
            )
        return _fn
    log.warning("unknown fetch.mode=%s -> treat as csv_only", mode)
    return None


def read_or_fetch_latest(cfg, symbol: str, csv_path: str, now_ts_in=None):
    """
    讀取本地 CSV，若資料過舊且允許，透過公開 API 補足至接近 now_ts。
    回傳：(df, latest_close_ts_utc)
    """
    # 讀 CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        df = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    # index 統一成 UTC
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop(columns=[c for c in ("timestamp",) if c in df.columns])
        df.index = ts
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = None

    now_ts_utc = pd.Timestamp.utcnow() if now_ts_in is None else pd.Timestamp(now_ts_in, tz="UTC")
    latest_close = df.index.max()

    fetch_fn = _get_fetch_fn(cfg)
    if fetch_fn is None:
        log.info("[DIAG] %s: fetch disabled (csv_only/none) -> use local CSV only", symbol)
        return df, latest_close

    # 若資料過舊，嘗試補資料
    anchor = now_ts_utc.floor("15min")
    diff_min = (anchor - latest_close).total_seconds() / 60.0
    if diff_min <= 0.0:
        return df, latest_close

    fcfg = cfg.get("fetch", {}) or {}
    max_retries = int(fcfg.get("max_retries", 3))
    retry_sleep = float(fcfg.get("retry_sleep_sec", 1.0))

    tries = 0
    cur_latest = latest_close
    appended_frames = []

    while tries <= max_retries and cur_latest < anchor:
        tries += 1
        log.info("[FETCH] %s retried=%d endTime=%s", symbol, tries-1, anchor.isoformat())
        fetched = fetch_fn(symbol, end_ts_utc=anchor)
        if fetched is None or fetched.empty:
            time.sleep(retry_sleep)
            continue
        # 只保留新於現有 latest 的資料
        fetched = fetched[fetched.index > cur_latest]
        if fetched.empty:
            time.sleep(retry_sleep)
            continue
        appended_frames.append(fetched)
        cur_latest = fetched.index.max()
        # 下一輪以更靠近 anchor 的截止時間再抓一次（或結束）
        anchor = cur_latest

    if appended_frames:
        add_df = pd.concat(appended_frames, axis=0).sort_index()
        keep_cols = [c for c in ("open","high","low","close","volume") if c in add_df.columns]
        add_df = add_df[keep_cols]
        df = pd.concat([df, add_df], axis=0).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out = df.copy()
        out.insert(0, "timestamp", out.index)
        out.to_csv(csv_path, index=False)
        latest_close = df.index.max()
        log.info("[UPDATE] %s CSV appended: +%d rows; latest=%s", symbol, len(add_df), latest_close.isoformat())
    else:
        log.warning("[STALE] %s: anchor=%s latest_close=%s diff_min=%.2f (fetch tried, no new rows)",
                    symbol, anchor.isoformat(), latest_close.isoformat(), diff_min)

    return df, latest_close
