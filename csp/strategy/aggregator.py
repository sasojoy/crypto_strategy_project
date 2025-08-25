from __future__ import annotations
import math, os, json
import pandas as pd
from dateutil import tz


TZ_TW = tz.gettz("Asia/Taipei")

def _weight(h: int) -> float:
    return math.sqrt(max(1, int(h)))


def _clean_prob_map(prob_map: dict) -> dict:
    clean = {}
    for k, v in (prob_map or {}).items():
        try:
            f = float(v)
        except Exception:
            continue
        if math.isnan(f) or f < 0.0 or f > 1.0:
            continue
        clean[k] = f
    return clean


def aggregate_signal(prob_map: dict, enter_threshold: float = 0.75, method: str = "max_weighted") -> dict:
    clean = _clean_prob_map(prob_map)
    if not clean:
        return {"side":"NONE","score":0.0,"prob_up_max":0.0,"prob_down_max":1.0,
                "chosen_h":None,"chosen_t":None,"reason":"empty_or_invalid_inputs"}

    if method == "majority":
        ups = sum(1 for _,p in clean.items() if p >= enter_threshold)
        downs = sum(1 for _,p in clean.items() if (1.0 - p) >= enter_threshold)
        if   ups > downs and ups > 0: side, score = "LONG", 1.0
        elif downs > ups and downs>0: side, score = "SHORT", 1.0
        else:                         side, score = "NONE", 0.0
        pu = max(clean.values())
        return {"side":side,"score":float(score),"prob_up_max":float(pu),
                "prob_down_max":float(1.0-pu),"chosen_h":None,"chosen_t":None,
                "reason":"majority"}

    scored = []
    total_weight = 0.0
    for (h,t), p in clean.items():
        w = _weight(h)
        total_weight += w
        scored.append(((h,t), p*w))
    if not scored or total_weight <= 0 or all((s is None or math.isnan(s[1]) or s[1]==0.0) for s in scored):
        return {"side":"NONE","score":0.0,"prob_up_max":0.0,"prob_down_max":1.0,
                "chosen_h":None,"chosen_t":None,"reason":"empty_or_invalid_inputs"}
    (chosen_ht, score) = max(scored, key=lambda x: x[1])
    (ch, ct) = chosen_ht
    pu = max(clean.values())
    side = "LONG" if pu >= enter_threshold else "NONE"
    score = 0.0 if (score is None or math.isnan(score)) else float(score)
    return {"side":side,"score":score,"prob_up_max":float(pu),"prob_down_max":float(1.0-pu),
            "chosen_h":int(ch),"chosen_t":float(ct),
            "reason":"ok" if side!="NONE" else "below_threshold"}


# get_latest_signal（如已存在，請覆蓋為更嚴格版）
def get_latest_signal(symbol: str, cfg: dict, fresh_min: float = 5.0, *, debug: bool = False) -> dict | None:
    from csp.core.feature import add_features
    try:
        from csp.models.classifier_multi import MultiThresholdClassifier
    except Exception:
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}

    csv_path = cfg["io"]["csv_paths"].get(symbol)
    if not csv_path or not os.path.exists(csv_path):
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_data"}

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_timestamp"}
    ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
    now_utc = pd.Timestamp.utcnow()
    lag_minutes = (now_utc - ts).total_seconds() / 60.0
    print(f"[TS] {symbol} latest_kline_ts UTC={ts.isoformat()} | TW={ts.tz_convert(TZ_TW).isoformat()}")
    print(f"[TS] {symbol} now UTC={now_utc.isoformat()} | TW={now_utc.tz_convert(TZ_TW).isoformat()} | lag_minutes={lag_minutes:.2f}")
    if lag_minutes > 15:
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "stale_data"}
    if lag_minutes > fresh_min:
        return None

    dff = add_features(df.copy())

    model_dir = os.path.join(cfg["io"].get("models_dir","models"), symbol, "cls_multi")
    if debug:
        files = os.listdir(model_dir) if os.path.exists(model_dir) else []
        print(f"[DEBUG] model_dir={model_dir} files={files} total={len(files)}")
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    feat_cols = meta.get("feature_columns") or []
    if not feat_cols:
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
    for c in feat_cols:
        if c not in dff.columns:
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "feature_missing"}

    X = dff[feat_cols].tail(1)
    if X.isna().any(axis=1).iloc[0]:
        nan_cols = X.columns[X.isna().any()].tolist()
        if debug:
            print(f"[WARN] feature NaN columns: {nan_cols}")
        dff[feat_cols] = dff[feat_cols].ffill()
        X = dff[feat_cols].tail(1)
        if X.isna().any(axis=1).iloc[0]:
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "feature_nan"}

    m = MultiThresholdClassifier.load(model_dir)
    prob_map = m.predict_proba(X)
    th = cfg.get("strategy",{}).get("enter_threshold",0.75)
    method = cfg.get("strategy",{}).get("aggregator_method","max_weighted")
    sig = aggregate_signal(prob_map, enter_threshold=th, method=method)
    if not sig.get("side"):
        sig["side"] = "NONE"
    if sig.get("score") is None or math.isnan(sig.get("score")):
        sig["score"] = 0.0
    sig["symbol"] = symbol
    sig["ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return sig
