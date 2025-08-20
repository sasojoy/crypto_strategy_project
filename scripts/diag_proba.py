from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

from csp.diagnostics import proba_diag
from csp.data.loader import load_15m_csv
from csp.pipeline.realtime_v2 import initialize_history


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--symbol")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--horizon", nargs="+", type=int, default=[2, 4, 8, 16, 48, 192])
    ap.add_argument("--last-n", type=int, default=200)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--use-model-of")
    ap.add_argument("--save-report")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))

    if args.all:
        symbols = cfg.get("symbols", [])
    elif args.symbol:
        symbols = [args.symbol]
    else:
        raise SystemExit("--symbol or --all required")

    model_override = args.use_model_of
    results = []

    for sym in symbols:
        try:
            paths = proba_diag.load_io_from_cfg(cfg, sym)
            if not paths.get("csv_path"):
                raise FileNotFoundError(f"csv path not found for {sym}")
            df = load_15m_csv(paths["csv_path"])
            df = initialize_history(df)
            feat_df, _ = proba_diag.build_features(df, max(args.horizon), paths["feat_params"])
        except Exception as e:
            for h in args.horizon:
                results.append({"symbol": sym, "horizon": h, "status": "ERROR", "error": str(e)})
            continue

        for h in args.horizon:
            model_sym = model_override if model_override else sym
            try:
                model_paths = proba_diag.load_io_from_cfg(cfg, model_sym)
                bundle = proba_diag.load_model_and_scaler(model_paths, model_sym, h)
                feature_names = bundle["feature_names"]
                X = feat_df[feature_names].values
                Xs = bundle["scaler"].transform(X)
                proba_seq = proba_diag.infer_proba(
                    bundle["model"], Xs, api=bundle["model_type"], feature_names=feature_names
                )
                summary = proba_diag.summarize_proba(proba_seq, last_n=args.last_n)
                sanity = proba_diag.sanity_checks(Xs[-1])
                proba_diag.print_debug(
                    args.debug,
                    symbol=sym,
                    model_path=bundle["model_path"],
                    scaler_path=bundle["scaler_path"],
                    feature_names_path=bundle["feature_names_path"],
                    X=Xs,
                    summary=summary,
                    sanity=sanity,
                    last_n=args.last_n,
                )
                results.append(
                    {
                        "symbol": sym,
                        "horizon": h,
                        "positive_ratio": bundle.get("positive_ratio", "N/A"),
                        "proba_p50": summary["p50"],
                        "proba_p90": summary["p90"],
                        "proba_max": summary["max"],
                        "proba_mean": summary["mean"],
                        "X_last_min": sanity["min"],
                        "X_last_max": sanity["max"],
                        "X_last_mean": sanity["mean"],
                        "has_nan": sanity["has_nan"],
                        "has_inf": sanity["has_inf"],
                        "model_path": bundle["model_path"],
                        "scaler_path": bundle["scaler_path"],
                        "feature_names_path": bundle["feature_names_path"],
                        "cross_model_symbol": model_override if model_override else "",
                        "status": "OK",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "symbol": sym,
                        "horizon": h,
                        "status": "ERROR",
                        "error": str(e),
                        "cross_model_symbol": model_override if model_override else "",
                    }
                )

    df_res = pd.DataFrame(results)
    if args.save_report:
        out_path = Path(args.save_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_path, index=False)
    else:
        print(df_res)


if __name__ == "__main__":
    main()
