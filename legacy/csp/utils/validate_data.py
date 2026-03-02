import pandas as pd
import numpy as np
import sys
from csp.data.fetcher import fetch_inc, fetch_full

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def validate_csv(path: str, tail_check: int = 10):
    """
    回傳 (ok, reason)
    ok=True 表示通過；False 時 reason 可能是：
      - read_fail:... / schema_miss / tz_parse_fail:... / not_sorted / duplicate_ts / nan_tail
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"read_fail:{e}"

    if any(c not in df.columns for c in REQUIRED_COLS):
        return False, "schema_miss"

    try:
        ts = pd.to_datetime(df["timestamp"], utc=True)
    except Exception as e:
        return False, f"tz_parse_fail:{e}"

    if not ts.is_monotonic_increasing:
        return False, "not_sorted"
    if ts.duplicated().any():
        return False, "duplicate_ts"

    num = df[["open", "high", "low", "close", "volume"]].replace([np.inf, -np.inf], np.nan)
    if num.tail(tail_check).isna().any().any():
        return False, "nan_tail"

    return True, "ok"


def ensure_data_ready(symbol: str, csv_path: str, fetch_policy: str = "auto", do_validate: bool = True):
    """Validate local CSV and fetch data as needed.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    csv_path : str
        Path to local CSV file.
    fetch_policy : str
        "auto" | "inc" | "full" | "none". When "auto", valid CSV triggers incremental fetch,
        otherwise a full refetch. "inc"/"full" force respective behavior when
        validation succeeds. "none" skips any network fetch and requires the local
        CSV to already be valid. Validation failure always triggers full refetch
        unless fetch_policy is "none", in which case it aborts.
    do_validate : bool
        If False, skip validation and just perform fetch according to policy.
    """
    if do_validate:
        ok, reason = validate_csv(csv_path)
    else:
        ok, reason = True, "skipped"

    if fetch_policy == "none":
        if not ok:
            print(
                f"[FATAL] {symbol} invalid csv ({reason}) but fetch_policy=none -> abort"
            )
            sys.exit(2)
        print(f"[VALID] {symbol} ok (fetch_policy=none -> skip fetch)")
        return {"ok": True, "reason": reason, "fetch": "none"}

    if not ok:
        print(f"[REPAIR] {symbol} invalid csv ({reason}) -> FULL refetch")
        res = fetch_full(symbol, csv_path)
        if not res.get("ok"):
            print(f"[FATAL] full refetch failed for {symbol}")
            sys.exit(2)
        ok2, reason2 = validate_csv(csv_path)
        if not ok2:
            print(f"[FATAL] still invalid after full refetch: {symbol} ({reason2})")
            sys.exit(2)
        print(f"[VALID] {symbol} ok")
        return res

    # ok
    print(f"[VALID] {symbol} ok")
    if fetch_policy == "full":
        return fetch_full(symbol, csv_path)
    else:
        # inc or auto
        return fetch_inc(symbol, csv_path)
