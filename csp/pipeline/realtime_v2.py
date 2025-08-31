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
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from dateutil import tz

from csp.utils.tz_safe import (
    normalize_df_to_utc,
    safe_ts_to_utc,
    now_utc,
    floor_utc,
)

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
    # 若 END_DATE 過舊（早於現在 1 日以上），忽略並警告
    if end:
        try:
            end_dt = pd.to_datetime(end, utc=True)
            if (pd.Timestamp.utcnow() - end_dt).total_seconds() > 86400:
                print(f"[WARN] END_DATE {end} too old; ignoring")
                end = None
        except Exception:
            pass
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


def _load_model_bundle(cfg: Dict[str, Any], symbol: str, debug: bool = False):
    """Load model components for a symbol. Print diagnostics and allow empty result."""
    mdir = Path(cfg["io"]["models_dir"]) / symbol
    files_loaded = []
    print(f"[MODEL] {symbol} dir={mdir}")
    if not mdir.exists():
        print(f"[WARN] model directory not found for {symbol}: {mdir}")
        return None

    model = None
    model_type = None
    model_path = None
    model_path_joblib = mdir / "xgb_h16_sklearn.joblib"
    model_path_json = mdir / "xgb_h16.json"
    scaler_path = mdir / "scaler_h16.joblib"
    feature_path = mdir / "feature_names.json"
    meta_path = mdir / "meta_h16.json"

    if model_path_joblib.exists():
        model = joblib.load(model_path_joblib)
        model_type = "sklearn"
        model_path = model_path_joblib
        files_loaded.append(model_path_joblib.name)
    elif model_path_json.exists():
        bst = xgb.Booster(); bst.load_model(str(model_path_json))
        model = bst
        model_type = "booster"
        model_path = model_path_json
        files_loaded.append(model_path_json.name)

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        files_loaded.append(scaler_path.name)
    else:
        scaler = None

    if feature_path.exists():
        feature_names = json.load(open(feature_path, "r", encoding="utf-8"))
        files_loaded.append(feature_path.name)
    else:
        meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}
        feature_names = meta.get("feature_cols", [])
        if meta_path.exists():
            files_loaded.append(meta_path.name)
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}

    print(f"[MODEL] loaded files: {files_loaded} (count={len(files_loaded)})")
    if model is None or scaler is None or not feature_names:
        return None

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_path": str(model_path) if model_path else None,
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


def run_once(csv_path: str, cfg: Dict[str, Any] | str, *, df: pd.DataFrame | None = None, debug: bool | None = None) -> Dict[str, Any]:
    """Load latest data, run model inference and return trading signal."""
    cfg = load_cfg(cfg)
    assert isinstance(cfg, dict), f"cfg must be dict, got {type(cfg)}"
    sym = _infer_symbol_from_path(csv_path)

    log = get_logger("realtime", cfg.get("io", {}).get("logs_dir", "logs"))

    if df is None:
        df15 = load_15m_csv(csv_path)
    else:
        df15 = normalize_df_to_utc(df)
        print(f"[DIAG] df.index.tz={df15.index.tz}, head_ts={df15.index[:3].tolist()}")
    assert str(df15.index.tz) == "UTC", "[DIAG] index not UTC"
    df15 = initialize_history(df15)

    # 時序檢查
    latest_ts = safe_ts_to_utc(df15.index[-1])
    now_ts = now_utc()
    lag_minutes = (now_ts - latest_ts).total_seconds() / 60.0
    print(
        f"[TS] latest_kline_ts UTC={latest_ts.isoformat()} | TW={latest_ts.tz_convert(TW).isoformat()}"
    )
    print(
        f"[TS] now UTC={now_ts.isoformat()} | TW={now_ts.tz_convert(TW).isoformat()} | lag_minutes={lag_minutes:.2f}"
    )
    if lag_minutes > 15:
        return {
            "symbol": sym,
            "price": float(df15["close"].iloc[-1]),
            "proba_up": 0.0,
            "score": 0.0,
            "side": "NONE",
            "reason": "stale_data",
        }

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
    feature_cols = feat_params.get("feature_columns")
    if feature_cols is None:
        feature_cols = feats.columns.tolist()
    row = feats[feature_cols].tail(1)
    if row.isna().any(axis=1).iloc[0]:
        nan_cols = row.columns[row.isna().any()].tolist()
        print(f"[WARN] feature NaN columns: {nan_cols}")
        feats[feature_cols] = feats[feature_cols].ffill()
        row = feats[feature_cols].tail(1)
        if row.isna().any(axis=1).iloc[0]:
            last = feats.iloc[-1]
            return {
                "symbol": sym,
                "price": float(last.get("close", 0.0)),
                "proba_up": 0.0,
                "score": 0.0,
                "side": "NONE",
                "reason": "feature_nan",
            }

    bundle = _load_model_bundle(cfg, sym, debug=bool(debug))
    if not bundle:
        last = feats.iloc[-1]
        return {
            "symbol": sym,
            "price": float(last.get("close", 0.0)),
            "proba_up": 0.0,
            "score": 0.0,
            "side": "NONE",
            "reason": "no_models_loaded",
        }

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_names"]

    # --- Diagnostics around feature matrix ---
    os.makedirs("logs/diag", exist_ok=True)
    feature_list = feature_cols
    try:
        X = feats[feature_list]
    except Exception as _e:
        print(
            f"[DIAG][{sym}] feature_list mismatch: {repr(_e)}; "
            f"got={list(feats.columns)[:10]}..."
        )
        X = feats[feature_cols]
    x_last = X.iloc[-1].replace([np.inf, -np.inf], np.nan)
    nan_cols = x_last[x_last.isna()].index.tolist()
    print(
        f"[DIAG][{sym}] x_last finite? {np.isfinite(x_last.fillna(0)).all()}  "
        f"nan_count={x_last.isna().sum()}"
    )
    if nan_cols:
        ctx = {
            "symbol": sym,
            "nan_cols": nan_cols,
            "x_last": x_last.to_dict(),
            "columns": list(X.columns),
            "ts": str(X.index[-1]) if hasattr(X, "index") else None,
        }
        with open(f"logs/diag/{sym}_nan_ctx.json", "w") as f:
            json.dump(ctx, f, ensure_ascii=False, indent=2)
        print(f"[DIAG][{sym}] dumped logs/diag/{sym}_nan_ctx.json")

    model_path = bundle.get("model_path")
    scaler_path = bundle.get("scaler_path")
    print(
        f"[DIAG][{sym}] model_path={model_path} exists={os.path.exists(model_path) if model_path else False}"
    )
    if scaler_path is not None:
        print(
            f"[DIAG][{sym}] scaler_path={scaler_path} exists={os.path.exists(scaler_path)}"
        )

    try:
        X1 = X.iloc[[-1]].copy()
        if scaler is not None:
            X1 = scaler.transform(X1)
            ok = np.isfinite(X1).all()
            print(f"[DIAG][{sym}] scaler.transform finite? {ok}")
            if not ok:
                np.save(f"logs/diag/{sym}_X1.npy", X1)
                print(f"[DIAG][{sym}] dumped logs/diag/{sym}_X1.npy")
    except Exception as e:
        print(f"[DIAG][{sym}] scaler.transform failed: {e}")
        print(f"[DIAG][{sym}] traceback:\n{traceback.format_exc()}")

    score = np.nan
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X1)[0]
            print(f"[DIAG][{sym}] proba={proba}")
            score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            pred = model.predict(X1)[0]
            print(f"[DIAG][{sym}] pred={pred}")
            score = float(pred) if np.isfinite(pred) else np.nan
    except Exception as e:
        print(f"[DIAG][{sym}] predict failed: {e}")
        print(f"[DIAG][{sym}] traceback:\n{traceback.format_exc()}")
        score = np.nan
        reason = f"PREDICT_EXCEPTION:{type(e).__name__}"

    # 再跑完整序列以供其他診斷使用
    try:
        Xs = scaler.transform(X) if scaler is not None else X.values
        if hasattr(model, "predict_proba"):
            proba_seq = np.clip(model.predict_proba(Xs)[:, 1], 0.0, 1.0)
        else:
            dmat = xgb.DMatrix(Xs, feature_names=feature_cols)
            proba_seq = np.clip(model.predict(dmat, output_margin=False), 0.0, 1.0)
    except Exception as e:
        print(f"[DIAG][{sym}] bulk predict failed: {e}")
        print(f"[DIAG][{sym}] traceback:\n{traceback.format_exc()}")
        proba_seq = np.array([])

    last = feats.iloc[-1]
    price = float(last["close"])
    atr_h4 = float(last.get("atr_h4", 0.0))
    ts = last["timestamp"].tz_convert(TW)

    proba_up = float(score) if np.isfinite(score) else np.nan
    long_thr = float(cfg["execution"]["long_prob_threshold"])
    short_thr = float(cfg["execution"]["short_prob_threshold"])
    atr_cfg = cfg["execution"]["atr_tp_sl"]
    side = _decide_side(proba_up, long_thr, short_thr) if np.isfinite(proba_up) else None

    log.info(f"最新訊號 [{sym}] @ {ts}")
    log.info(f"price={price:.2f}, proba_up={proba_up:.3f}, atr_h4={atr_h4:.2f}")

    # Debug info
    dbg = debug if debug is not None else (os.getenv("DEBUG") == "1")
    diag_low_var = bool(len(proba_seq) >= 20 and (proba_seq[-20:] < 0.02).all()) if len(proba_seq) else False
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

    if isinstance(proba_up, float) and (np.isnan(proba_up) or not np.isfinite(proba_up)):
        if 'reason' not in locals():
            reason = "NAN_FEATURES" if nan_cols else "UNKNOWN_NAN"
        result = {
            "symbol": sym,
            "price": price,
            "proba_up": float('nan'),
            "score": float('nan'),
            "side": "NONE",
            "atr_h4": atr_h4,
            "tp": None,
            "sl": None,
            "diag_low_var": diag_low_var,
            "diag_X_last": X.iloc[-1].to_dict(),
            "reason": reason,
        }
    else:
        result = {
            "symbol": sym,
            "price": price,
            "proba_up": proba_up,
            "score": float(proba_up),
            "atr_h4": atr_h4,
            "side": side if side else "NONE",
            "tp": tp,
            "sl": sl,
            "diag_low_var": diag_low_var,
            "diag_X_last": X.iloc[-1].to_dict(),
            "reason": "OK",
        }
    return result
