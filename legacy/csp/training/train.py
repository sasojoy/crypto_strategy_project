from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import json
import joblib

from csp.models.train_h16_dynamic import train
from csp.utils.io import load_cfg


def _bundle_model(sym_dir: Path) -> None:
    """Create ``model.pkl`` bundle from individual artifacts if present."""

    model_path = sym_dir / "xgb_h16_sklearn.joblib"
    scaler_path = sym_dir / "scaler_h16.joblib"
    bundle_path = sym_dir / "model.pkl"

    if not (model_path.exists() and scaler_path.exists()):
        return

    try:
        bundle = {"model": joblib.load(model_path), "scaler": joblib.load(scaler_path)}
        joblib.dump(bundle, bundle_path)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"[WARN] bundle save failed for {sym_dir.name}: {exc}")


def train_for_symbol(symbol: str, out_dir: str | Path, cfg: Dict[str, Any] | str,
                     *, cfg_path: str | None = None) -> Dict[str, Any]:
    """Train model for ``symbol`` and persist artifacts under ``out_dir``.

    Parameters
    ----------
    symbol:
        Trading pair symbol, e.g. ``"BTCUSDT"``.
    out_dir:
        Directory where per-symbol sub-directories are created.
    cfg:
        Strategy configuration dictionary or path. ``load_cfg`` ensures a
        dictionary is always used internally.
    cfg_path:
        Optional configuration path, retained for compatibility with callers
        that wish to log the original source. It is unused but accepted for
        symmetry with ``optimize_then_train_symbol``.
    """

    _ = cfg_path  # compatibility no-op; retained for signature symmetry

    cfg_dict = load_cfg(cfg)
    csv_map = cfg_dict.get("io", {}).get("csv_paths", {})
    csv_path = csv_map.get(symbol)
    if not csv_path:
        raise ValueError(f"[TRAIN] {symbol}: csv path not configured")

    models_root = Path(out_dir)
    sym_dir = models_root / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TRAIN] {symbol} <- {csv_path} -> {sym_dir}")
    result = train(csv_path, cfg_dict, models_dir_override=str(sym_dir), symbol=symbol)

    positive_ratio = result.get("positive_ratio")
    if positive_ratio is not None:
        print(f"[INFO] {symbol} positive ratio={positive_ratio:.4f}")
    if result.get("warning"):
        print(f"[WARN] {symbol}: {result['warning']}")

    _bundle_model(sym_dir)

    meta = {
        "symbol": symbol,
        "trained_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "data_range": result.get("used_range_utc"),
        "version": 1,
        "positive_ratio": positive_ratio,
    }
    with open(sym_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    bundle_path = sym_dir / "model.pkl"
    manifest_path = (models_root / symbol / "model.pkl").as_posix() if bundle_path.exists() else None

    return {
        "symbol": symbol,
        "csv_path": str(csv_path),
        "out_dir": str(sym_dir),
        "result": result,
        "positive_ratio": positive_ratio,
        "warning": result.get("warning"),
        "model_bundle": str(bundle_path) if bundle_path.exists() else None,
        "manifest_path": manifest_path,
    }
