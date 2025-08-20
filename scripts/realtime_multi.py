from __future__ import annotations

import argparse
import json
from dateutil import tz
import yaml

from csp.data.fetcher import update_csv_with_latest
from csp.pipeline.realtime_v2 import run_once
TZ_TW = tz.gettz("Asia/Taipei")
from csp.utils.notifier import notify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    symbols = cfg.get("symbols", [])
    csv_map = (cfg.get("io", {}) or {}).get("csv_paths", {})
    live_cfg = (cfg.get("io", {}) or {}).get("live_fetch", {}) or {}

    results = {}

    for sym in symbols:
        csv_path = csv_map.get(sym)
        if not csv_path:
            print(f"[SKIP] {sym}: No CSV path in config")
            continue
        print(f"[REALTIME] {sym} <- {csv_path}")
        stale = False
        if live_cfg.get("enabled"):
            try:
                df = update_csv_with_latest(sym, csv_path, interval=live_cfg.get("interval", "15m"))
                last_ts = df["timestamp"].iloc[-1]
                print(f"  last closed UTC={last_ts.isoformat()} | TW={(last_ts.tz_convert(TZ_TW)).isoformat()}")
                stale = bool(df.attrs.get("stale"))
            except Exception as e:
                print(f"[WARN] live fetch failed for {sym}: {e}")
                stale = True
        try:
            # run_once 會自動挑選 models/<SYMBOL>/ 或全域 models/
            res = run_once(csv_path, cfg_path=args.cfg, debug=args.debug)
        except Exception as e:
            res = {"symbol": sym, "side": None, "error": str(e)}
        if stale:
            res["warning"] = "STALE DATA"
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

        note = " [STALE DATA]" if r.get("warning") else ""
        base = f"{sym}: {side_display} | P={price:.2f} | proba_up={pu:.3f}{note}"
        if r.get("side"):
            base += f" | TP={tp:.2f} | SL={sl:.2f}"
        if r.get("diag_low_var"):
            base += " [DIAG:LOW VAR]"
        lines.append(base)

    msg = "\n".join(lines)
    notify(msg, cfg.get("notify", {}).get("telegram"))

    # also print structured json
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
