
import argparse, json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from csp.data.binance import fetch_to_csv
from csp.utils.io import load_cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--symbols", nargs="*", help="Override symbols list")
    ap.add_argument("--days", type=int, default=365, help="How many days back to fetch")
    ap.add_argument("--start", help="UTC start time ISO8601, overrides --days (e.g., 2024-07-01T00:00:00Z)")
    ap.add_argument("--end", help="UTC end time ISO8601, default now")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    symbols = args.symbols or cfg.get("symbols") or [cfg["symbol"]]
    csv_paths = cfg.get("io", {}).get("csv_paths", {})
    interval = cfg.get("bar_interval", "15m")

    if args.start:
        start = datetime.fromisoformat(args.start.replace("Z","+00:00"))
    else:
        start = datetime.now(timezone.utc) - timedelta(days=args.days)
    end = datetime.fromisoformat(args.end.replace("Z","+00:00")) if args.end else None

    out = {}
    for s in symbols:
        out_path = csv_paths.get(s, f"resources/{s.lower()}_{interval}.csv")
        print(f"Fetching {s} {interval} -> {out_path}")
        out[s] = fetch_to_csv(s, interval, start, end, out_path)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
