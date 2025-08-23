from __future__ import annotations
import argparse
from pathlib import Path
from csp.models.train_h16_dynamic import train
from csp.utils.io import load_cfg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    csv_map = cfg["io"]["csv_paths"]
    base_models = Path(cfg["io"]["models_dir"])

    for sym in cfg["symbols"]:
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
