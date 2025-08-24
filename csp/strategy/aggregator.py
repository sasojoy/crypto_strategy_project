from __future__ import annotations
import math, os, json
import pandas as pd


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
                "chosen_h":None,"chosen_t":None,"reason":"empty_or_nan_prob_map"}

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

    scored = [((h,t), p*_weight(h)) for (h,t), p in clean.items()]
    if not scored:
        return {"side":"NONE","score":0.0,"prob_up_max":0.0,"prob_down_max":1.0,
                "chosen_h":None,"chosen_t":None,"reason":"scored_empty"}
    (chosen_ht, score) = max(scored, key=lambda x: x[1])
    (ch, ct) = chosen_ht
    pu = max(clean.values())
    side = "LONG" if pu >= enter_threshold else "NONE"
    score = 0.0 if (score is None or math.isnan(score)) else float(score)
    return {"side":side,"score":score,"prob_up_max":float(pu),"prob_down_max":float(1.0-pu),
            "chosen_h":int(ch),"chosen_t":float(ct),
            "reason":"ok" if side!="NONE" else "below_threshold"}


# get_latest_signal（如已存在，請覆蓋為更嚴格版）
def get_latest_signal(symbol: str, cfg: dict, fresh_min: float = 5.0) -> dict | None:
    from csp.core.feature import add_features
    try:
        from csp.models.classifier_multi import MultiThresholdClassifier
    except Exception:
        return None

    csv_path = cfg["io"]["csv_paths"].get(symbol)
    if not csv_path or not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
    age_min = (pd.Timestamp.utcnow() - ts).total_seconds()/60.0
    if age_min > fresh_min:
        return None

    dff = add_features(df.copy())

    model_dir = os.path.join(cfg["io"].get("models_dir","models"), symbol, "cls_multi")
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    feat_cols = meta.get("feature_columns") or []
    if not feat_cols:
        return None
    for c in feat_cols:
        if c not in dff.columns:
            return None

    X = dff[feat_cols].tail(1)
    if X.isna().any().any():
        return None

    m = MultiThresholdClassifier.load(model_dir)
    prob_map = m.predict_proba(X)
    th = cfg.get("strategy",{}).get("enter_threshold",0.75)
    method = cfg.get("strategy",{}).get("aggregator_method","max_weighted")
    sig = aggregate_signal(prob_map, enter_threshold=th, method=method)
    sig["symbol"] = symbol
    sig["ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return sig
