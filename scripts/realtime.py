import argparse
from csp.pipeline.realtime_v2 import run_once

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to 15m CSV (latest rows used)")
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    out = run_once(args.csv, args.cfg, debug=args.debug)
    print(out)
