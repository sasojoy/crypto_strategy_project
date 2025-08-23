# -*- coding: utf-8 -*-
"""realtime_v2.py
新增「初始化 warmup 歷史」的日期區間支援（可選）。
- 若不提供任何日期參數，行為完全不變。
- 支援來源：
  1) 函式參數 init_date_args={'start': 'YYYY-MM-DD', ...}
  2) 環境變數 START_DATE / END_DATE / DAYS （僅用於 warmup 初始化歷史）
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from dateutil import tz

from csp.data.loader import load_15m_csv
from csp.features.h16 import build_features_15m_4h
from csp.core.feature import add_features
from csp.utils.config import get_symbol_features
from csp.utils.logger import get_logger
from csp.utils.io import load_cfg

try:
    from csp.utils.dates import resolve_time_range_like, slice_by_utc
except Exception:
    resolve_time_range_like = None
    slice_by_utc = None

TW = tz.gettz("Asia/Taipei")

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


# === Realtime inference utilities ===
def _infer_symbol_from_path(csv_path: str) -> Optional[str]:
    name = Path(csv_path).name.upper()
    if "BTC" in name:
        return "BTCUSDT"
    if "ETH" in name:
        return "ETHUSDT"
    if "BCH" in name:
        return "BCHUSDT"
    return None


def _load_model_bundle(cfg: Dict[str, Any], symbol: str):
    mdir = Path(cfg["io"]["models_dir"]) / symbol
    if not mdir.exists():
        raise FileNotFoundError(f"Model directory not found for {symbol}: {mdir}")

    model_path_joblib = mdir / "xgb_h16_sklearn.joblib"
    model_path_json = mdir / "xgb_h16.json"
    scaler_path = mdir / "scaler_h16.joblib"
    feature_path = mdir / "feature_names.json"
    meta_path = mdir / "meta_h16.json"

    if model_path_joblib.exists():
        model = joblib.load(model_path_joblib)
        model_type = "sklearn"
        model_path = model_path_joblib
    elif model_path_json.exists():
        bst = xgb.Booster(); bst.load_model(str(model_path_json))
        model = bst
        model_type = "booster"
        model_path = model_path_json
    else:
        raise FileNotFoundError(f"No model file found under {mdir}")

    scaler = joblib.load(scaler_path)
    if feature_path.exists():
        feature_names = json.load(open(feature_path, "r", encoding="utf-8"))
    else:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        feature_names = meta.get("feature_cols", [])
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "model_type": model_type,
        "meta": meta,
    }


def _compute_tp_sl(price: float, atr: float, side: str, atr_cfg: Dict[str, Any]):
    if side == "long":
        tp = price + atr * float(atr_cfg["long"]["tp_mult"])
        sl = price - atr * float(atr_cfg["long"]["sl_mult"])
    else:
        tp = price - atr * float(atr_cfg["short"]["tp_mult"])
        sl = price + atr * float(atr_cfg["short"]["sl_mult"])
    return float(tp), float(sl)


def _decide_side(proba_up: float, long_thr: float, short_thr: float) -> Optional[str]:
    if proba_up >= long_thr:
        return "long"
    if (1.0 - proba_up) >= short_thr:
        return "short"
    return None


def run_once(csv_path: str, cfg: Dict[str, Any] | str, *, debug: bool | None = None) -> Dict[str, Any]:
    """Load latest data, run model inference and return trading signal."""
    cfg = load_cfg(cfg)
    assert isinstance(cfg, dict), f"cfg must be dict, got {type(cfg)}"
    sym = _infer_symbol_from_path(csv_path)

    log = get_logger("realtime", cfg.get("io", {}).get("logs_dir", "logs"))

    df15 = load_15m_csv(csv_path)
    df15 = initialize_history(df15)

    # live fetch handled externally; df15 already up-to-date

    feat_params = get_symbol_features(cfg, sym)
    feats = build_features_15m_4h(
        df15,
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

    bundle = _load_model_bundle(cfg, sym)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_names"]
    X = feats[feature_cols].values
    Xs = scaler.transform(X)

    if bundle["model_type"] == "sklearn":
        proba_seq = np.clip(model.predict_proba(Xs)[:, 1], 0.0, 1.0)
    else:
        dmat = xgb.DMatrix(Xs, feature_names=feature_cols)
        proba_seq = np.clip(model.predict(dmat, output_margin=False), 0.0, 1.0)

    last = feats.iloc[-1]
    proba_up = float(proba_seq[-1])
    price = float(last["close"])
    atr_h4 = float(last["atr_h4"])
    ts = last["timestamp"].tz_convert(TW)

    long_thr = float(cfg["execution"]["long_prob_threshold"])
    short_thr = float(cfg["execution"]["short_prob_threshold"])
    atr_cfg = cfg["execution"]["atr_tp_sl"]
    side = _decide_side(proba_up, long_thr, short_thr)

    log.info(f"最新訊號 [{sym}] @ {ts}")
    log.info(f"price={price:.2f}, proba_up={proba_up:.3f}, atr_h4={atr_h4:.2f}")

    # Debug info
    dbg = debug if debug is not None else (os.getenv("DEBUG") == "1")
    diag_low_var = bool(len(proba_seq) >= 20 and (proba_seq[-20:] < 0.02).all())
    if dbg:
        print(f"[DEBUG] symbol={sym}")
        print(f"[DEBUG] model_path={bundle['model_path']}")
        print(f"[DEBUG] scaler_path={bundle['scaler_path']}")
        print(f"[DEBUG] X.shape={Xs.shape}")
        last_row = Xs[-1]
        bad = not np.isfinite(last_row).all()
        print(f"[DEBUG] last row has NaN/inf? {bad}")
        if len(last_row):
            print(f"[DEBUG] last row stats: min={float(np.min(last_row)):.6f}, max={float(np.max(last_row)):.6f}, mean={float(np.mean(last_row)):.6f}")
        tail = proba_seq[-200:]
        if len(tail):
            p50 = float(np.percentile(tail, 50))
            p90 = float(np.percentile(tail, 90))
            pmax = float(np.max(tail))
            print(f"[DEBUG] proba_up last200 p50={p50:.6f}, p90={p90:.6f}, max={pmax:.6f}")

    tp = sl = None
    if side:
        tp, sl = _compute_tp_sl(price, atr_h4, side, atr_cfg)
        log.info(f"進場方向: {side.upper()} | TP={tp:.2f}, SL={sl:.2f}")
    else:
        log.info("無進場訊號（等待）")

    return {
        "symbol": sym,
        "price": price,
        "proba_up": proba_up,
        "atr_h4": atr_h4,
        "side": side,
        "tp": tp,
        "sl": sl,
        "diag_low_var": diag_low_var,
    }
