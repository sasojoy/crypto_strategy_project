from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(Path(__file__).resolve().parent, "..")))

from csp.training import train_for_symbol

try:
    from csp.training import optimize_then_train_symbol  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    optimize_then_train_symbol = None  # type: ignore

from csp.utils.io import load_cfg


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_symbols(spec: str | None) -> list[str]:
    if not spec:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="csp/configs/strategy.yaml")
    parser.add_argument("--symbol", help="train single symbol")
    parser.add_argument("--symbols", help="comma separated symbols (overrides cfg.symbols)")
    parser.add_argument("--out-dir", help="output directory for models")
    parser.add_argument("--optimize", action="store_true", help="enable hyperparameter optimization before training")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials (default 50)")
    parser.add_argument("--timeout-min", type=int, default=None, help="Optuna timeout in minutes (default 20)")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    symbols: list[str]
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols_env = os.getenv("CSP_TRAIN_SYMBOLS")
        symbols_spec = args.symbols or symbols_env
        symbols = _split_symbols(symbols_spec) or cfg.get("symbols", [])

    if not symbols:
        raise SystemExit("No symbols provided via --symbol/--symbols/CSP_TRAIN_SYMBOLS or cfg.symbols")

    models_root = Path(
        args.out_dir
        or os.getenv("CSP_TRAIN_OUTDIR")
        or cfg.get("io", {}).get("models_dir", "models")
    )
    models_root.mkdir(parents=True, exist_ok=True)

    do_opt = args.optimize or _env_flag("CSP_OPTIMIZE", False)
    n_trials = args.n_trials if args.n_trials is not None else int(os.getenv("CSP_OPT_N_TRIALS", "50"))
    timeout_min = (
        args.timeout_min
        if args.timeout_min is not None
        else int(os.getenv("CSP_OPT_TIMEOUT_MIN", "20"))
    )

    print(
        "[TRAIN] Effective Optimize Params:",
        json.dumps(
            {
                "optimize": do_opt,
                "n_trials": n_trials,
                "timeout_min": timeout_min,
                "symbols": symbols,
            },
            ensure_ascii=False,
        ),
    )

    manifest = {"generated_at": None, "models": {}}
    missing: list[str] = []

    for sym in symbols:
        if do_opt:
            if optimize_then_train_symbol is None:
                raise RuntimeError("optimize_then_train_symbol not available in this environment")
            result = optimize_then_train_symbol(
                symbol=sym,
                out_dir=str(models_root),
                cfg=cfg,
                n_trials=n_trials,
                timeout_min=timeout_min,
                cfg_path=args.cfg,
            )
            if result.get("best_params"):
                print(f"[TRAIN][{sym}] best_params={result['best_params']}")
        else:
            result = train_for_symbol(symbol=sym, out_dir=str(models_root), cfg=cfg, cfg_path=args.cfg)

        manifest_path = result.get("manifest_path")
        if manifest_path:
            manifest_models = manifest.setdefault("models", {})
            manifest_models[sym] = {"path": manifest_path}
        else:
            missing.append(sym)

    manifest["generated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    manifest_path = models_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    if missing:
        raise SystemExit(f"model.pkl missing for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
