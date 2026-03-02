from __future__ import annotations

import sys, os, traceback, datetime as _dt


def log_diag(msg: str):
    print(f"[DIAG] {msg}", file=sys.stderr, flush=True)


def log_trace(prefix: str, exc: BaseException):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_diag(f"{prefix} type={type(exc).__name__} msg={exc}")
    log_diag(f"TRACEBACK\n{tb}")
    try:
        os.makedirs("logs/diag", exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/diag/trace_{ts}.log", "a", encoding="utf-8") as f:
            f.write(f"{prefix}: {type(exc).__name__}: {exc}\n{tb}\n")
    except Exception:
        pass

