from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
from csp.features.h16 import build_features_15m_4h
from csp.core.feature import add_features
from csp.data.labeling import make_labels
from csp.models.classifier_multi import MultiThresholdClassifier
from csp.utils.config import get_symbol_features
from csp.utils.io import load_cfg
from csp.utils.tz_safe import (
    normalize_df_to_utc,
    safe_ts_to_utc,
    now_utc,
    floor_utc,
)


def load_csv(csv_path: str, days: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_df_to_utc(df)
    print(f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}")
    assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"
    if days is not None:
        limit = days * 24 * 60 // 15
        df = df.iloc[-limit:]
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    csv_map = cfg["io"]["csv_paths"]
    model_cfg = cfg.get("model", {})
    horizons = model_cfg.get("horizons", [16])
    thresholds = model_cfg.get("thresholds", [0.0])
    model_type = model_cfg.get("type", "xgboost")
    calibrate = model_cfg.get("calibrate", "isotonic")
    cv_cfg = model_cfg.get("cv")
    out_dir_tpl = model_cfg.get("out_dir", "models/{SYMBOL}/cls_multi")

    for sym in cfg.get("symbols", []):
        csv_path = csv_map.get(sym)
        if not csv_path:
            print(f"[SKIP] {sym}: no csv path")
            continue
        print(f"[LOAD] {sym} from {csv_path}")
        df_raw = load_csv(csv_path, args.days)
        feat_params = get_symbol_features(cfg, sym)
        feats = build_features_15m_4h(
            df_raw.reset_index(),
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
        df_labels = make_labels(feats, horizons, thresholds)
        feats = feats.loc[df_labels.index]
        feature_cols = [
            c for c in feats.columns if c not in ["timestamp", "open", "high", "low", "close", "volume"]
        ]
        X = feats[feature_cols]
        clf = MultiThresholdClassifier(horizons, thresholds, model_type=model_type)
        clf.fit(X, df_labels, cv=cv_cfg, calibrate=calibrate)
        out_dir = Path(out_dir_tpl.replace("{SYMBOL}", sym))
        clf.save(str(out_dir))
        print(f"[SAVED] {sym} -> {out_dir}")
