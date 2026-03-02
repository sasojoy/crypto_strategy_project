import os, json, math, argparse
import pandas as pd
from datetime import datetime
import pytz

from csp.utils.tz_safe import (
    normalize_df_to_utc,
    safe_ts_to_utc,
    now_utc,
    floor_utc,
)


def _fmt_ts(ts_utc):
    tz = pytz.timezone("Asia/Taipei")
    ts = safe_ts_to_utc(ts_utc)
    return ts.tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--fresh-min", type=float, default=5.0)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))

    # 調整 import 路徑以符合專案
    from csp.core.feature import add_features
    from csp.strategy.aggregator import aggregate_signal
    try:
        from csp.models.classifier_multi import MultiThresholdClassifier
    except Exception as e:
        print("❌ 無法 import MultiThresholdClassifier（Issue3 是否完成？）", e)
        return

    models_dir = cfg["io"].get("models_dir", "models")
    csv_paths  = cfg["io"]["csv_paths"]
    symbols = args.symbols or list(csv_paths.keys())

    for sym in symbols:
        print("="*90)
        print(f"[SYMBOL] {sym}")
        csv_path = csv_paths.get(sym)
        if not csv_path or not os.path.exists(csv_path):
            print(f"❌ CSV 不存在: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df = normalize_df_to_utc(df)
        print(
            f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}"
        )
        assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"
        cols_lower = {c.lower() for c in df.columns}
        need = {"open","high","low","close","volume","timestamp"}
        if not need.issubset(cols_lower):
            print(f"❌ 缺必要欄位。現有：{sorted(df.columns)}")
            continue

        ts = df.index[-1]
        age_min = (pd.Timestamp.utcnow() - ts).total_seconds()/60.0
        print(f"📈 最後一根: {_fmt_ts(ts)} | 與現在差距: {age_min:.2f} 分")

        # 產特徵並檢查最後一列 NaN
        dff = add_features(df.copy())
        last = dff.iloc[-1]
        nan_cols = [c for c in dff.columns if pd.isna(last[c])]
        print(f"🧪 最後一列 NaN 欄位數: {len(nan_cols)}")
        if nan_cols:
            print("  → ", ", ".join(nan_cols[:30]))

        # 模型與欄位對齊
        model_dir = os.path.join(models_dir, sym, "cls_multi")
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            print(f"❌ 缺 meta: {meta_path}")
            continue
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        feat_cols = meta.get("feature_columns") or []
        if not feat_cols:
            print("❌ meta.json 無 feature_columns")
            continue

        miss = [c for c in feat_cols if c not in dff.columns]
        if miss:
            print(f"❌ 特徵缺少: {miss[:30]}")
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

        bad_prob = [(k,v) for k,v in prob_map.items() if v is None or not (0.0 <= float(v) <= 1.0) or math.isnan(float(v))]
        if bad_prob:
            print(f"❌ prob_map 含越界/NaN：{bad_prob[:10]}")
        else:
            sample = list(prob_map.items())[:6]
            print(f"✅ prob_map 個數={len(prob_map)}，樣本={sample}")

        th = cfg.get("strategy",{}).get("enter_threshold",0.75)
        method = cfg.get("strategy",{}).get("aggregator_method","max_weighted")
        sig = aggregate_signal(prob_map, enter_threshold=th, method=method)
        print(f"🧭 aggregate_signal → {sig}")

        if age_min > args.fresh_min:
            print(f"⚠️ 資料過舊（>{args.fresh_min} 分）")

if __name__ == "__main__":
    main()
