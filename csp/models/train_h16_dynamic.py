# -*- coding: utf-8 -*-
"""train_h16_dynamic.py (fixed)
- 接受 **DataFrame** 或 **CSV 檔路徑** 作為輸入。
- 若為路徑：自動以 pandas 讀入，並嘗試辨識 timestamp 欄位。
- 支援日期區間（環境變數 START_DATE/END_DATE/DAYS，或由呼叫端提供）與 warmup 緩衝。
- 若未提供任何日期設定，行為與舊版一致。
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Union, Tuple, Optional, Dict

import pandas as pd

try:
    from csp.utils.dates import resolve_time_range_like, slice_by_utc
except Exception:
    resolve_time_range_like = None
    slice_by_utc = None

# === 可依你的專案調整 ===
INDICATOR_MAX_WINDOW = 200   # e.g., MA/EMA/BBands 最大視窗
HORIZON_MAX = 192            # e.g., 你的多 horizon 最大值
BAR_MINUTES = 15             # 15m
SAFETY_BARS = 10             # 額外保險

# === 讀入工具 ===
def _read_csv_smart(path: Union[str, Path]) -> pd.DataFrame:
    """以最寬鬆方式讀 CSV 並找出 timestamp 欄位。
    假設檔案至少有 ['timestamp', 'open', 'high', 'low', 'close'] 或類似欄位。
    """
    df = pd.read_csv(path)
    # 嘗試找 timestamp 欄位名稱
    ts_candidates = ["timestamp", "time", "open_time", "datetime"]
    ts_col = None
    for c in ts_candidates:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        # 兜底：若 index 是時間字串也嘗試 parse
        # 若找不到就讓後續報錯，幫助你定位實際欄位名
        if df.index.name:
            try:
                df.index = pd.to_datetime(df.index, utc=True)
                return df
            except Exception:
                pass
        raise ValueError(f"找不到 timestamp 欄位（嘗試過 {ts_candidates}）。請確認 CSV 欄位。")

    # 轉成 UTC 時間
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if df[ts_col].isna().all():
        raise ValueError(f"timestamp 欄位 '{ts_col}' 解析失敗，請確認格式。")
    # 設為 index（不破壞原欄位也可以，但後續流程假設能用 index 篩選）
    df = df.set_index(ts_col)
    # 排序 & 去重（以防資料混亂）
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

# === 日期參數 ===
def _read_date_args_from_env() -> Dict[str, Optional[Union[str, int]]]:
    start = os.getenv("START_DATE")  # YYYY-MM-DD
    end = os.getenv("END_DATE")
    days = os.getenv("DAYS")
    if days is not None:
        try:
            days = int(days)
        except Exception:
            days = None
    return {"start": start, "end": end, "days": days}

def _resolve_date_args(date_args=None, **kwargs) -> Dict[str, Optional[Union[str, int]]]:
    # 優先順序：顯示參數 > date_args 字典 > 環境變數
    d = {}
    for k in ("start", "end", "days"):
        if kwargs.get(k, None) is not None:
            d[k] = kwargs[k]
    if not d and isinstance(date_args, dict):
        d = {k: date_args.get(k) for k in ("start", "end", "days") if date_args.get(k) is not None}
    if not d:
        d = _read_date_args_from_env()
    return d

# === 日期區間應用（含 warmup） ===
def _apply_date_range(df: pd.DataFrame, date_args: dict) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if resolve_time_range_like is None or slice_by_utc is None:
        # 未安裝 utils，就直接回傳原 df
        return df, None, None
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        # 若呼叫端塞進來是非 DatetimeIndex，但含 timestamp 欄位，這裡兜底處理
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
            idx = df.index
        else:
            raise ValueError("資料缺少 DatetimeIndex 或 timestamp 欄位，無法做日期切片。")
    utc_start, utc_end = resolve_time_range_like(date_args, idx)
    # warmup 緩衝（避免技術指標/label 在區間起頭失真）
    buf_bars = max(INDICATOR_MAX_WINDOW, HORIZON_MAX) + SAFETY_BARS
    buf_minutes = buf_bars * BAR_MINUTES
    utc_start_warm = utc_start - pd.Timedelta(minutes=buf_minutes)
    sliced = slice_by_utc(df, col="timestamp", start_utc=utc_start_warm, end_utc=utc_end)
    if sliced.empty:
        raise ValueError(f"No data in selected range: {utc_start} ~ {utc_end} (UTC)")
    return sliced, utc_start, utc_end

# === 對外訓練入口 ===
def train(input_data: Union[pd.DataFrame, str, Path], cfg: dict, *, date_args: dict | None = None, **kwargs):
    """
    - input_data: DataFrame 或 CSV 路徑字串/Path
    - cfg: 你的設定物件/字典
    - date_args: 可選 {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD', 'days': 90}
    - 也可傳 **kwargs: start=..., end=..., days=...
    備註：這裡只負責「資料讀取 + 日期區間 + warmup」，後續特徵/訓練保持你原本的流程。
    """
    # 1) 讀資料
    if isinstance(input_data, (str, Path)):
        df = _read_csv_smart(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"train(input_data=...) 需要 DataFrame 或 CSV 路徑，實際收到：{type(input_data)}")

    # 2) 解析日期參數
    dargs = _resolve_date_args(date_args, **kwargs)

    # 3) 套日期區間（含 warmup）
    if any(v is not None for v in dargs.values()):
        df2, utc_start, utc_end = _apply_date_range(df, dargs)
    else:
        df2, utc_start, utc_end = df, None, None

    # 4) ==== 你的原本特徵/標籤/訓練流程 ====
    # features = build_features(df2)
    # labels = build_labels(df2)
    # df_train = features if utc_start is None else features.loc[features.index >= utc_start]
    # model = fit_model(df_train, labels.loc[df_train.index])
    # save(model, cfg)...
    #
    # 這裡暫留為範本，避免覆蓋你的既有邏輯。

    # 5) 回傳最小訓練摘要（供呼叫端日誌使用）
    return {
        "used_range_utc": (str(utc_start) if utc_start is not None else None,
                           str(utc_end) if utc_end is not None else None),
        "rows_in": int(len(df)),
        "rows_used": int(len(df2)),
    }
