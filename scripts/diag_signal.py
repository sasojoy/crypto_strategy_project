import os
import json
import math
import argparse
from datetime import datetime, timezone

import pandas as pd
import pytz


def _fmt_ts(ts_utc):
    tz = pytz.timezone("Asia/Taipei")
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument(
        "--fresh-min",
        type=float,
        default=5.0,
        help="資料新鮮度門檻（分鐘）",
    )
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    from csp.core.feature import add_features  # type: ignore
    from csp.strategy.aggregator import aggregate_signal
    try:
        from csp.models.classifier_multi import MultiThresholdClassifier
    except Exception as e:  # pragma: no cover - defensive
        print("❌ 無法 import MultiThresholdClassifier，請確認 Issue3 是否完成。", e)
        return

    models_dir = cfg["io"].get("models_dir", "models")
    csv_paths = cfg["io"]["csv_paths"]
    symbols = args.symbols or list(csv_paths.keys())

    for sym in symbols:
        print("=" * 90)
        print(f"[SYMBOL] {sym}")
        csv_path = csv_paths.get(sym)
        if not csv_path or not os.path.exists(csv_path):
            print(f"❌ CSV 不存在: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        lower_cols = {c.lower() for c in df.columns}
        need = {"open", "high", "low", "close", "volume", "timestamp"}
        if not need.issubset(lower_cols):
            print(f"❌ 缺少必要欄位，現有: {sorted(lower_cols)}")
            continue

        ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
        age_min = (pd.Timestamp.utcnow() - ts).total_seconds() / 60.0
        print(f"📈 最後一根: {_fmt_ts(ts)} | 與現在差距: {age_min:.1f} 分鐘")

        dff = add_features(df.copy())
        last = dff.iloc[-1]
        nan_cols = [c for c in dff.columns if pd.isna(last[c])]
        print(f"🧪 最後一列 NaN 欄位數: {len(nan_cols)}")
        if nan_cols:
            print("  → ", ", ".join(nan_cols[:30]))

        model_dir = os.path.join(models_dir, sym, "cls_multi")
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            print(f"❌ 找不到模型 meta: {meta_path}")
            continue
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        feat_cols = meta.get("feature_columns") or []
        if not feat_cols:
            print("❌ meta.json 缺 feature_columns")
            continue

        miss = [c for c in feat_cols if c not in dff.columns]
        if miss:
            print(f"❌ 特徵欄位缺少: {miss[:30]}")
            continue

        X = dff[feat_cols].tail(1)
        if X.isna().any().any():
            bad = list(X.columns[X.isna().any()])
            print(f"❌ 輸入特徵仍有 NaN: {bad[:30]}")
            continue

        m = MultiThresholdClassifier.load(model_dir)
        prob_map = m.predict_proba(X)

        if not prob_map:
            print("❌ predict_proba 回傳空 dict")
            continue

        bad_prob = [
            (k, v)
            for k, v in prob_map.items()
            if v is None or not (0.0 <= float(v) <= 1.0) or math.isnan(float(v))
        ]
        if bad_prob:
            print(f"❌ prob_map 含越界/NaN：{bad_prob[:10]}")
        else:
            sample = list(prob_map.items())[:5]
            print(f"✅ prob_map 個數={len(prob_map)}，樣本={sample}")

        th = cfg.get("strategy", {}).get("enter_threshold", 0.75)
        method = cfg.get("strategy", {}).get("aggregator_method", "max_weighted")
        sig = aggregate_signal(prob_map, enter_threshold=th, method=method)
        print(f"🧭 aggregate_signal → {sig}")

        if age_min > args.fresh_min:
            print(f"⚠️ 資料可能過舊（>{args.fresh_min} min）")


if __name__ == "__main__":
    main()

