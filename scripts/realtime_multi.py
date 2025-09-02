from __future__ import annotations

import argparse
import json
import traceback
from dateutil import tz

from csp.pipeline.realtime_v2 import run_once
from csp.strategy.aggregator import sanitize_score
TZ_TW = tz.gettz("Asia/Taipei")
from csp.utils.notifier import notify
from csp.utils.io import load_cfg
from csp.utils.validate_data import ensure_data_ready
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
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
                info = ensure_data_ready(sym, csv_path)
                last_ts = pd.to_datetime(info.get("last_ts"), utc=True)
                print(f"  last closed UTC={last_ts.isoformat()} | TW={(last_ts.tz_convert(TZ_TW)).isoformat()}")
            except Exception as e:
                print(f"[WARN] data fetch failed for {sym}: {e}")
                stale = True
        try:
            # run_once 會自動挑選 models/<SYMBOL>/ 或全域 models/
            res = run_once(csv_path, cfg, debug=args.debug)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERR][{sym}] {repr(e)}")
            print(f"[ERR][{sym}] traceback:\n{tb}")
            res = {"symbol": sym, "side": None, "error": str(e)}
        res["score"] = sanitize_score(res.get("score", res.get("proba_up")))
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
        score = sanitize_score(r.get("score", r.get("proba_up", 0.0)))
        tp = r.get("tp")
        sl = r.get("sl")

        note = " [STALE DATA]" if r.get("warning") else ""
        base = f"{sym}: {side_display} | P={price:.2f} | score={score:.3f}{note}"
        if r.get("side") and r["side"] not in ("NONE", None):
            base += f" | TP={tp:.2f} | SL={sl:.2f}"
        if r.get("diag_low_var"):
            base += " [DIAG:LOW VAR]"
        if r.get("reason"):
            base += f" (reason={r['reason']})"
        lines.append(base)

    msg = "\n".join(lines)
    notify(msg, cfg.get("notify", {}).get("telegram"))

    # also print structured json
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
