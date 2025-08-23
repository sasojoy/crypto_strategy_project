from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
from dateutil import parser as dateparser

from csp.utils.notifier import notify_trade_close
from csp.strategy.aggregator import get_latest_signal


def _load_position(path: str) -> Dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return None
    if not data or not data.get("symbol"):
        return None
    return data


def _append_trade(log_path: Path, trade: Dict[str, Any]):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "entry_time",
        "exit_time",
        "symbol",
        "side",
        "entry_price",
        "exit_price",
        "qty",
        "pnl",
        "pnl_ratio",
        "reason",
    ]
    line = ",".join(str(trade.get(c, "")) for c in cols)
    header_needed = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write(",".join(cols) + "\n")
        f.write(line + "\n")


def check_exit_once(cfg: Dict[str, Any], latest_price: float, now_ts: datetime, dry_run: bool = False) -> Dict[str, Any]:
    """
    依序檢查 TP、SL、時間止損、模型翻向。
    若觸發則平倉、記錄交易、清除 position、通知。
    回傳 dict 包含 action (hold|close)、reason、pnl、position。
    """
    io_cfg = cfg.get("io", {})
    pos_file = io_cfg.get("position_file")
    if not pos_file:
        return {"action": "hold", "reason": "no_position_file", "pnl": 0.0, "position": None}

    pos = _load_position(pos_file)
    if not pos:
        return {"action": "hold", "reason": "no_position", "pnl": 0.0, "position": None}

    side = pos.get("side")
    entry_price = float(pos.get("entry_price", 0.0))
    qty = float(pos.get("qty", 0.0) or 0.0)
    entry_time_raw = pos.get("entry_time")
    try:
        entry_ts = dateparser.parse(entry_time_raw) if entry_time_raw else None
    except Exception:
        entry_ts = None
    if entry_ts is None:
        entry_ts = now_ts
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.replace(tzinfo=timezone.utc)

    pnl_price = (latest_price - entry_price) if side == "long" else (entry_price - latest_price)
    pnl = pnl_price * qty
    pnl_ratio = (pnl_price / entry_price) if entry_price else 0.0

    risk = cfg.get("risk", {})
    tp_ratio = float(risk.get("take_profit_ratio", 0.0))
    sl_ratio = float(risk.get("stop_loss_ratio", 0.0))
    max_hold_min = int(risk.get("max_holding_minutes", 0))
    flip_thr = float(risk.get("flip_threshold", 1.1))

    reason = None
    if tp_ratio and pnl_ratio >= tp_ratio:
        reason = "tp"
    elif sl_ratio and pnl_ratio <= -sl_ratio:
        reason = "sl"
    elif max_hold_min:
        held_min = (now_ts - entry_ts).total_seconds() / 60.0
        if held_min >= max_hold_min:
            reason = "time"
    if reason is None:
        sig = get_latest_signal(pos.get("symbol"), cfg)
        if sig:
            sig_side = str(sig.get("side", "NONE")).lower()
            sig_conf = float(sig.get("score", 0.0))
            if sig_side in ("long", "short") and sig_side != str(side).lower() and sig_conf >= flip_thr:
                reason = "flip"

    if reason is None:
        return {"action": "hold", "reason": None, "pnl": pnl, "position": pos}

    # close
    hold_minutes = (now_ts - entry_ts).total_seconds() / 60.0
    if not dry_run:
        notify_trade_close(
            symbol=pos.get("symbol"),
            side=str(side).upper(),
            entry_price=entry_price,
            exit_price=latest_price,
            opened_at=entry_ts.isoformat(),
            closed_at=now_ts.isoformat(),
            reason=reason,
            pnl_ratio=pnl_ratio,
            holding_minutes=hold_minutes,
            telegram_conf=cfg.get("notify", {}).get("telegram"),
        )
        trade = {
            "entry_time": entry_ts.isoformat(),
            "exit_time": now_ts.isoformat(),
            "symbol": pos.get("symbol"),
            "side": side,
            "entry_price": entry_price,
            "exit_price": latest_price,
            "qty": qty,
            "pnl": pnl,
            "pnl_ratio": pnl_ratio,
            "reason": reason,
        }
        logs_dir = Path(io_cfg.get("logs_dir", "logs"))
        _append_trade(logs_dir / "trades.csv", trade)
        # clear position
        Path(pos_file).write_text("{}")
    else:
        # dry run, skip notify/log/clear but still return close action
        pass

    return {"action": "close", "reason": reason, "pnl": pnl, "position": pos}
