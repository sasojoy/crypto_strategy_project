# csp/data/loader.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

from csp.utils.tz_safe import (
    normalize_df_to_utc_index,
    safe_ts_to_utc,
    now_utc,
    floor_utc,
)

def _pick_timestamp_col(df: pd.DataFrame) -> str:
    # 常見欄位別名
    candidates = ["timestamp", "open_time", "time", "date", "datetime"]
    low = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in low:
            return low[name]
    # 若沒找到，嘗試有 'time' 字樣的第一個欄位
    for c in df.columns:
        if "time" in c.lower():
            return c
    raise ValueError("找不到時間欄位，請至少提供 timestamp/open_time/time/date/datetime。")

def load_15m_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)
    # 欄位名稱轉小寫（保留原始名稱用於對應）
    orig_cols = df.columns.tolist()
    lower_map = {c: c.lower() for c in orig_cols}
    df.rename(columns=lower_map, inplace=True)

    ts_col = _pick_timestamp_col(df)
    s = df[ts_col]

    # 1) 若是全數字（或數字字串），推斷秒/毫秒
    is_numeric_like = pd.api.types.is_numeric_dtype(s) or s.astype(str).str.fullmatch(r"\d+").all()
    if is_numeric_like:
        s = s.astype("int64")
        unit = "ms" if s.max() > 1_000_000_000_000 else "s"
        df["timestamp"] = pd.to_datetime(s, unit=unit, utc=True)
    else:
        # 2) 否則就當成字串日期（含時區），讓 pandas 自動 parse
        df["timestamp"] = pd.to_datetime(s, utc=True, errors="raise")

    # 確保基本價格欄位存在（常見別名）
    alias = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    for want in ["open", "high", "low", "close"]:
        if want not in df.columns:
            # 嘗試從常見縮寫映射
            if want[0] in df.columns:
                df.rename(columns={want[0]: want}, inplace=True)
            elif alias.get(want[0], "") in df.columns:
                df.rename(columns={alias[want[0]]: want}, inplace=True)

    missing = [x for x in ["open", "high", "low", "close"] if x not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要欄位：{missing}（至少需要 open/high/low/close）")

    # 排序並確保 UTC index
    cols = ["timestamp", "open", "high", "low", "close"]
    if "volume" in df.columns:
        cols.append("volume")
    df = df[cols]
    df = normalize_df_to_utc_index(df, ts_col="timestamp")
    print(f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}")
    assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"
    return df
