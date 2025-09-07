from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(Path(__file__).resolve().parent, "..")))
import joblib
from csp.models.train_h16_dynamic import train
from csp.utils.io import load_cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--symbol", help="train single symbol")
    ap.add_argument("--symbols", help="comma separated symbols")
    ap.add_argument("--out-dir", help="output directory for models")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    csv_map = cfg["io"]["csv_paths"]
    base_models = Path(args.out_dir or cfg["io"].get("models_dir", "models"))

    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = cfg.get("symbols", [])

    manifest = {"generated_at": None, "models": {}}
    missing = []
    for sym in symbols:
        csv_path = csv_map.get(sym)
        if not csv_path:
            print(f"[SKIP] {sym}: no csv path")
            continue

        out_dir = base_models / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TRAIN] {sym} <- {csv_path}  -> {out_dir}")
        res = train(csv_path, cfg, models_dir_override=str(out_dir), symbol=sym)
        pr = res.get("positive_ratio")
        if pr is not None:
            print(f"[INFO] {sym} positive ratio={pr:.4f}")
        if res.get("warning"):
            print(f"[WARN] {sym}: {res['warning']}")

        model_path = out_dir / "xgb_h16_sklearn.joblib"
        scaler_path = out_dir / "scaler_h16.joblib"
        bundle_path = out_dir / "model.pkl"
        try:
            bundle = {"model": joblib.load(model_path), "scaler": joblib.load(scaler_path)}
            joblib.dump(bundle, bundle_path)
        except Exception as e:
            print(f"[ERROR] bundle save failed for {sym}: {e}")

        meta = {
            "symbol": sym,
            "trained_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "data_range": res.get("used_range_utc"),
            "version": 1,
            "positive_ratio": pr,
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if bundle_path.exists():
            rel_path = (base_models / sym / "model.pkl").as_posix()
            manifest["models"][sym] = {"path": rel_path}
        else:
            missing.append(sym)

    manifest["generated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    base_models.mkdir(parents=True, exist_ok=True)
    with open(base_models / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if missing:
        raise SystemExit(f"model.pkl missing for: {', '.join(missing)}")
