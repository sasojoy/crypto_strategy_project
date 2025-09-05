from __future__ import annotations

import sys, traceback, datetime as _dt, os


def log_diag(msg: str):
    print(f"[DIAG] {msg}", file=sys.stderr)


def log_trace(prefix: str, exc: BaseException):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_diag(f"{prefix} type={type(exc).__name__} msg={exc}")
    log_diag(f"TRACEBACK\n{tb}")
    # 同時落地到檔案，方便 systemd 以外檢查
    try:
        os.makedirs("logs/diag", exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/diag/trace_{ts}.log", "a", encoding="utf-8") as f:
            f.write(f"{prefix}: {type(exc).__name__}: {exc}\n{tb}\n")
    except Exception:
        pass

