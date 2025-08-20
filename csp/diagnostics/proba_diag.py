from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import xgboost as xgb

from csp.features.h16 import build_features_15m_4h
from csp.utils.config import get_symbol_features


def load_io_from_cfg(cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    csv_path = cfg.get("io", {}).get("csv_paths", {}).get(symbol)
    models_root = Path(cfg.get("io", {}).get("models_dir", "models")) / symbol
    feat_params = get_symbol_features(cfg, symbol)
    return {"csv_path": csv_path, "models_dir": models_root, "feat_params": feat_params}


def load_model_and_scaler(paths: Dict[str, Any], symbol: str, horizon: int) -> Dict[str, Any]:
    models_root = Path(paths["models_dir"])
    mdir = models_root / f"h{horizon}"
    if not mdir.exists():
        mdir = models_root

    feature_path = models_root / "feature_names.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"feature_names.json not found for {symbol} under {models_root}")
    feature_names = json.load(open(feature_path, "r", encoding="utf-8"))

    model_path_joblib = mdir / f"xgb_h{horizon}_sklearn.joblib"
    model_path_json = mdir / f"xgb_h{horizon}.json"
    scaler_path = mdir / f"scaler_h{horizon}.joblib"
    meta_path = mdir / f"meta_h{horizon}.json"

    if model_path_joblib.exists():
        model = joblib.load(model_path_joblib)
        model_type = "sklearn"
        model_path = model_path_joblib
    elif model_path_json.exists():
        model = xgb.Booster(); model.load_model(str(model_path_json))
        model_type = "booster"
        model_path = model_path_json
    else:
        raise FileNotFoundError(f"No model file found under {mdir}")

    scaler = joblib.load(scaler_path)
    positive_ratio: Optional[float] = None
    if meta_path.exists():
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        positive_ratio = meta.get("positive_ratio")

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_names_path": str(feature_path),
        "model_type": model_type,
        "positive_ratio": positive_ratio,
    }


def build_features(df, horizon: int, feat_params: Dict[str, Any]):
    feats = build_features_15m_4h(
        df,
        ema_windows=tuple(feat_params["ema_windows"]),
        rsi_window=feat_params["rsi_window"],
        bb_window=feat_params["bb_window"],
        bb_std=feat_params["bb_std"],
        atr_window=feat_params["atr_window"],
        h4_resample=feat_params["h4_resample"],
    )
    return feats, list(feats.columns)


def infer_proba(model, X: np.ndarray, api: str, feature_names: Optional[Iterable[str]] = None) -> np.ndarray:
    if api == "sklearn":
        return np.clip(model.predict_proba(X)[:, 1], 0.0, 1.0)
    dmat = xgb.DMatrix(X, feature_names=list(feature_names) if feature_names else None)
    return np.clip(model.predict(dmat, output_margin=False), 0.0, 1.0)


def summarize_proba(prob_up: np.ndarray, last_n: int = 200) -> Dict[str, float]:
    if prob_up.size == 0:
        return {"p50": np.nan, "p90": np.nan, "max": np.nan, "mean": np.nan}
    tail = prob_up[-last_n:]
    return {
        "p50": float(np.percentile(tail, 50)),
        "p90": float(np.percentile(tail, 90)),
        "max": float(np.max(tail)),
        "mean": float(np.mean(tail)),
    }


def sanity_checks(row: np.ndarray) -> Dict[str, float]:
    return {
        "has_nan": bool(np.isnan(row).any()),
        "has_inf": bool(np.isinf(row).any()),
        "min": float(np.min(row)),
        "max": float(np.max(row)),
        "mean": float(np.mean(row)),
    }


def print_debug(enabled: bool, *, symbol: str, model_path: str, scaler_path: str,
                feature_names_path: str, X: np.ndarray, summary: Dict[str, float],
                sanity: Dict[str, float], last_n: int) -> None:
    if not (enabled or os.environ.get("DEBUG") == "1"):
        return
    print(f"symbol={symbol}")
    print(f"model_path={model_path}")
    print(f"scaler_path={scaler_path}")
    print(f"feature_names_path={feature_names_path}")
    print(f"X.shape={X.shape}")
    print(f"last row has NaN? {sanity['has_nan']} / inf? {sanity['has_inf']}")
    print(
        f"last row stats: min={sanity['min']:.6f}, max={sanity['max']:.6f}, mean={sanity['mean']:.6f}"
    )
    print(
        f"proba_up last{last_n} p50={summary['p50']:.6f}, "
        f"p90={summary['p90']:.6f}, max={summary['max']:.6f}, mean={summary['mean']:.6f}"
    )

