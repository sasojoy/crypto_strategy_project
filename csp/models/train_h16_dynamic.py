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
from csp.utils.io import load_cfg
from csp.utils.timez import ensure_utc_index, to_utc_ts

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
    ts_candidates = ["timestamp", "time", "open_time", "datetime"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        if df.index.name:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError(f"找不到 timestamp 欄位（嘗試過 {ts_candidates}）。請確認 CSV 欄位。")
            df = ensure_utc_index(df)
        else:
            raise ValueError(f"找不到 timestamp 欄位（嘗試過 {ts_candidates}）。請確認 CSV 欄位。")
    else:
        if ts_col != "timestamp":
            df = df.rename(columns={ts_col: "timestamp"})
        df = ensure_utc_index(df, "timestamp")
    print(f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}")
    assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"
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
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("資料缺少 DatetimeIndex，無法做日期切片。")

    start_str = date_args.get("start")
    end_str = date_args.get("end")
    days = date_args.get("days")

    if start_str or end_str:
        utc_start = to_utc_ts(start_str) if start_str else idx.min()
        utc_end = to_utc_ts(end_str) if end_str else idx.max()
    elif days is not None:
        utc_end = idx.max()
        utc_start = utc_end - pd.Timedelta(days=int(days))
    else:
        return df, None, None

    buf_bars = max(INDICATOR_MAX_WINDOW, HORIZON_MAX) + SAFETY_BARS
    buf_minutes = buf_bars * BAR_MINUTES
    utc_start_warm = utc_start - pd.Timedelta(minutes=buf_minutes)
    sliced = df.loc[utc_start_warm:utc_end]
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
    cfg = load_cfg(cfg)
    assert isinstance(cfg, dict), f"cfg must be dict, got {type(cfg)}"
    # 1) 讀資料
    if isinstance(input_data, (str, Path)):
        df = _read_csv_smart(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = ensure_utc_index(input_data.copy())
        print(f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}")
        assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"
    else:
        raise TypeError(f"train(input_data=...) 需要 DataFrame 或 CSV 路徑，實際收到：{type(input_data)}")

    # 2) 解析日期參數
    dargs = _resolve_date_args(date_args, **kwargs)

    # 3) 套日期區間（含 warmup）
    if any(v is not None for v in dargs.values()):
        df2, utc_start, utc_end = _apply_date_range(df, dargs)
    else:
        df2, utc_start, utc_end = df, None, None
    df2 = df2.reset_index().rename_axis(None)

    # 4) ==== 特徵 / 標籤 / 模型訓練 ====
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import joblib
    import xgboost as xgb
    from csp.features.h16 import build_features_15m_4h, make_labels
    from csp.core.feature import add_features
    from csp.utils.config import get_symbol_features

    # 若提供路徑字串，嘗試推斷 symbol 名稱
    symbol = kwargs.get("symbol")
    if symbol is None and isinstance(input_data, (str, Path)):
        name = Path(input_data).name.upper()
        if "BTC" in name:
            symbol = "BTCUSDT"
        elif "ETH" in name:
            symbol = "ETHUSDT"
        elif "BCH" in name:
            symbol = "BCHUSDT"

    # 取得特徵參數並建立特徵
    feat_params = get_symbol_features(cfg, symbol) if symbol else get_symbol_features(cfg, "BTCUSDT")
    feats = build_features_15m_4h(
        df2,
        ema_windows=tuple(feat_params["ema_windows"]),
        rsi_window=feat_params["rsi_window"],
        bb_window=feat_params["bb_window"],
        bb_std=feat_params["bb_std"],
        atr_window=feat_params["atr_window"],
        h4_resample=feat_params["h4_resample"],
    )
    feats = add_features(
        feats,
        prev_high_period=feat_params["prev_high_period"],
        prev_low_period=feat_params["prev_low_period"],
        bb_window=feat_params["bb_window"],
        atr_window=feat_params["atr_window"],
        atr_percentile_window=feat_params["atr_percentile_window"],
    )

    # 建立標籤並對齊
    horizon = int(cfg.get("train", {}).get("target_horizon_bars", 16))
    y = make_labels(feats, horizon=horizon)
    feats = feats.iloc[:-horizon].reset_index(drop=True)
    y = y.iloc[:-horizon].reset_index(drop=True)

    # 擷取特徵欄位順序
    feature_cols = [c for c in feats.columns if c not in ["timestamp", "open", "high", "low", "close", "volume"]]
    X = feats[feature_cols].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    test_size = float(cfg.get("train", {}).get("test_size", 0.2))
    random_state = int(cfg.get("train", {}).get("random_state", 42))
    X_train, X_valid, y_train, y_valid = train_test_split(
        Xs, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    xgb_params = cfg.get("train", {}).get("xgb", {})
    model = xgb.XGBClassifier(**xgb_params, use_label_encoder=False)
    model.fit(X_train, y_train)

    acc = float(accuracy_score(y_valid, model.predict(X_valid))) if len(y_valid) else 0.0

    # === 保存模型與附檔 ===
    out_dir = Path(kwargs.get("models_dir_override") or cfg["io"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_h16_sklearn.joblib"
    joblib.dump(model, model_path)
    scaler_path = out_dir / "scaler_h16.joblib"
    joblib.dump(scaler, scaler_path)

    feature_path = out_dir / "feature_names.json"
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    positive_ratio = float(y.mean()) if len(y) else 0.0
    meta = {
        "feature_cols": feature_cols,
        "positive_ratio": positive_ratio,
        "model_type": "xgbclassifier",
        "valid_accuracy": acc,
    }
    meta_path = out_dir / "meta_h16.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ETH 正例比例警告
    warning = None
    if symbol == "ETHUSDT":
        print(f"[TRAIN] ETH positive ratio={positive_ratio:.4f}")
        if positive_ratio < 0.03:
            warning = "建議調整標記門檻或設定 scale_pos_weight=(neg/pos)"
            print(f"[WARN] {warning}")

    # 5) 回傳最小訓練摘要（供呼叫端日誌使用）
    return {
        "used_range_utc": (str(utc_start) if utc_start is not None else None,
                           str(utc_end) if utc_end is not None else None),
        "rows_in": int(len(df)),
        "rows_used": int(len(df2)),
        "valid_accuracy": acc,
        "positive_ratio": positive_ratio,
        "warning": warning,
    }
