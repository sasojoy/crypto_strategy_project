# -*- coding: utf-8 -*-
"""realtime_v2.py
新增「初始化 warmup 歷史」的日期區間支援（可選）。
- 若不提供任何日期參數，行為完全不變。
- 支援來源：
  1) 函式參數 init_date_args={'start': 'YYYY-MM-DD', ...}
  2) 環境變數 START_DATE / END_DATE / DAYS （僅用於 warmup 初始化歷史）
"""
from __future__ import annotations
import os
import pandas as pd

try:
    from csp.utils.dates import resolve_time_range_like, slice_by_utc
except Exception:
    resolve_time_range_like = None
    slice_by_utc = None

def _read_date_args_from_env():
    start = os.getenv("START_DATE")  # YYYY-MM-DD
    end = os.getenv("END_DATE")
    days = os.getenv("DAYS")
    if days is not None:
        try:
            days = int(days)
        except:
            days = None
    return {"start": start, "end": end, "days": days}

def _apply_init_warmup(df: pd.DataFrame, date_args: dict | None):
    if not date_args or resolve_time_range_like is None or slice_by_utc is None:
        return df
    # 確保 timestamp 可解析為 UTC
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        idx = df["timestamp"]
    else:
        idx = df.index
    utc_start, utc_end = resolve_time_range_like(date_args, idx)
    sliced = slice_by_utc(df, col="timestamp", start_utc=utc_start, end_utc=utc_end)
    if sliced.empty:
        raise ValueError(f"No data in selected init warmup range: {utc_start} ~ {utc_end} (UTC)")
    return sliced

# ==== 你原本的即時主流程 ====
def initialize_history(df: pd.DataFrame, *, init_date_args: dict | None = None):
    """在你原本的初始化步驟（建立特徵/狀態）前，選擇性地先做日期範圍切片。"""
    # 若未提供，嘗試從環境變數帶入（僅 warmup）
    if init_date_args is None:
        env_args = _read_date_args_from_env()
        if any(v is not None for v in env_args.values()):
            init_date_args = env_args

    df2 = _apply_init_warmup(df, init_date_args)
    # 接著做你原本的初始化處理，例如：
    # state = build_state_from_history(df2)
    # return state
    return df2  # 佔位：請替換成你的實作（回傳 state/特徵緩存等）
