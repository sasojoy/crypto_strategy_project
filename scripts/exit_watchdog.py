from __future__ import annotations

import argparse
import time
from datetime import datetime

import requests
import yaml
from dateutil import tz

from csp.runtime.exit_watchdog import check_exit_once

TW = tz.gettz("Asia/Taipei")


def fetch_price(symbol: str, base_url: str = "https://api.binance.com") -> float:
    url = f"{base_url}/api/v3/ticker/price"
    resp = requests.get(url, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return float(data["price"])


def main():
    ap = argparse.ArgumentParser(description="Periodic exit watchdog")
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--interval-sec", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    pos_file = cfg.get("io", {}).get("position_file")
    if not pos_file:
        print("No position_file in config")
        return

    while True:
        try:
            pos = yaml.safe_load(open(pos_file, "r", encoding="utf-8")) or {}
        except Exception:
            pos = {}
        sym = pos.get("symbol")
        if sym:
            try:
                price = fetch_price(sym)
            except Exception as e:
                print(f"[WARN] fetch price failed for {sym}: {e}")
                price = None
            if price is not None:
                now = datetime.now(tz=TW)
                res = check_exit_once(cfg, price, now, dry_run=args.dry_run)
                print(res)
        else:
            print("[INFO] no open position")
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
