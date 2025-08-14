from __future__ import annotations

import argparse
import json
from dateutil import tz
import yaml

from csp.pipeline.realtime_v2 import run_once
TZ_TW = tz.gettz("Asia/Taipei")
from csp.utils.notifier import notify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    symbols = cfg.get("symbols", [])
    csv_map = (cfg.get("io", {}) or {}).get("csv_paths", {})

    results = {}

    for sym in symbols:
        csv_path = csv_map.get(sym)
        if not csv_path:
            print(f"[SKIP] {sym}: No CSV path in config")
            continue
        print(f"[REALTIME] {sym} <- {csv_path}")
        try:
            # run_once 會自動挑選 models/<SYMBOL>/ 或全域 models/
            res = run_once(csv_path, cfg_path=args.cfg, symbol=sym)
        except Exception as e:
            res = {"symbol": sym, "side": None, "error": str(e)}
        results[sym] = res

    # Build summary lines safely
    lines = ["📊 多幣別即時訊號"]
    for sym, r in results.items():
        if "error" in r:
            lines.append(f"{sym}: ERROR {r['error']}")
            continue

        side_display = r["side"].upper() if r.get("side") else "WAIT"
        price = r.get("price")
        pu = r.get("proba_up")
        tp = r.get("tp")
        sl = r.get("sl")

        base = f"{sym}: {side_display} | P={price:.2f} | proba_up={pu:.3f}"
        if r.get("side"):
            base += f" | TP={tp:.2f} | SL={sl:.2f}"
        lines.append(base)

    msg = "\n".join(lines)
    notify(msg, cfg.get("notify", {}).get("telegram"))

    # also print structured json
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()