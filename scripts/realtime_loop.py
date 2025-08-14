from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta

import yaml
from dateutil import tz

from csp.pipeline.realtime_v2 import run_once
from csp.utils.notifier import notify

TW = tz.gettz("Asia/Taipei")


def next_quarter_with_delay(now: datetime, delay_sec: int = 15) -> datetime:
    base = now.replace(second=0, microsecond=0)
    minute = (base.minute // 15) * 15
    slot = base.replace(minute=minute)
    if now >= slot + timedelta(seconds=delay_sec):
        slot += timedelta(minutes=15)
    return slot + timedelta(seconds=delay_sec)


def run_once_all(cfg_path: str) -> dict:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    symbols = cfg.get("symbols", [])
    csv_map = cfg.get("io", {}).get("csv_paths", {})
    results = {}

    for sym in symbols:
        csv_path = csv_map.get(sym)
        if not csv_path:
            print(f"[SKIP] {sym}: No CSV path in config")
            continue
        print(f"[REALTIME] {sym} <- {csv_path}")
        try:
            res = run_once(csv_path, cfg_path, symbol=sym)
        except Exception as e:
            res = {"symbol": sym, "side": None, "error": str(e)}
        results[sym] = res

    # Summary notify
    lines = []
    for sym, r in results.items():
        if "error" in r:
            lines.append(f"{sym}: ERROR {r['error']}")
        else:
            side = r.get("side") or "-"
            price = r.get("price", float("nan"))
            pu = r.get("proba_up", float("nan"))
            lines.append(f"{sym}: {side} | P={price:.2f} | proba_up={pu:.3f}")
    notify("⏱️ 多幣別即時訊號\n" + "\n".join(lines), cfg.get("notify", {}).get("telegram"))

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def main():
    ap = argparse.ArgumentParser(description="Run realtime every 15m + delay seconds (with live Binance fetch).")
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--delay-sec", type=int, default=15)
    args = ap.parse_args()

    while True:
        now = datetime.now(tz=TW)
        target = next_quarter_with_delay(now, args.delay_sec)
        wait = (target - now).total_seconds()
        if wait > 0:
            print(f"[LOOP] 現在 {now.strftime('%F %T%z')}，等到 {target.strftime('%F %T%z')} 再跑（{int(wait)} 秒）")
            time.sleep(wait)
        try:
            run_once_all(args.cfg)
        except Exception as e:
            print(f"[ERROR] loop run failed: {e}")


if __name__ == "__main__":
    main()