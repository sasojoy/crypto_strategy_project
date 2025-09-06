from __future__ import annotations
import argparse
from pathlib import Path
from csp.models.train_h16_dynamic import train
from csp.utils.io import load_cfg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--symbol", help="train single symbol")
    ap.add_argument("--symbols", help="comma separated symbols")
    ap.add_argument("--out", help="output directory when training single symbol")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    csv_map = cfg["io"]["csv_paths"]
    base_models = Path(cfg["io"]["models_dir"])

    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = cfg["symbols"]

    for sym in symbols:
        csv_path = csv_map.get(sym)
        if not csv_path:
            print(f"[SKIP] {sym}: no csv path")
            continue

        out_dir = Path(args.out) if args.out and args.symbol else base_models / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TRAIN] {sym} <- {csv_path}  -> {out_dir}")
        res = train(csv_path, cfg, models_dir_override=str(out_dir), symbol=sym)
        pr = res.get("positive_ratio")
        if pr is not None:
            print(f"[INFO] {sym} positive ratio={pr:.4f}")
        if res.get("warning"):
            print(f"[WARN] {sym}: {res['warning']}")

        # ensure output is not empty
        if not any(out_dir.iterdir()):
            raise SystemExit(f"Model output empty for {sym}")
