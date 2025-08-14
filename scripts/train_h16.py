import argparse
from pathlib import Path
from csp.models.train_h16_dynamic import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to 15m CSV")
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--outdir", default=None, help="Optional output dir (e.g., models/BTCUSDT)")
    args = ap.parse_args()
    if args.outdir:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res = train(args.csv, args.cfg, models_dir_override=args.outdir)
    print(res)
