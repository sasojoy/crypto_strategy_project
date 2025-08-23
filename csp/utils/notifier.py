import os
from datetime import datetime

import requests
import pytz

from csp.utils.logger import get_logger

log = get_logger("notify")

def _fmt_local(ts_utc_iso: str | None, tz: str = "Asia/Taipei") -> str:
    """Return formatted local time string from an optional UTC ISO timestamp."""
    tzinfo = pytz.timezone(tz)
    if ts_utc_iso:
        try:
            dt = datetime.fromisoformat(ts_utc_iso.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.utc)
        except Exception:
            dt = datetime.utcnow().replace(tzinfo=pytz.utc)
    else:
        dt = datetime.utcnow().replace(tzinfo=pytz.utc)
    return dt.astimezone(tzinfo).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_num(x: float | int, digits: int = 2) -> str:
    """Format numbers with thousand separator and given decimals."""
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return str(x)


def _fmt_pct(x: float, digits: int = 2) -> str:
    """Format ratio as percentage with sign."""
    try:
        return f"{x*100:+.{digits}f}%"
    except Exception:
        return str(x)


def notify(message: str, telegram_conf: dict | None = None):
    print(f"[NOTIFY] {message}")
    if not telegram_conf or not telegram_conf.get("enabled", False):
        log.info("notify: telegram disabled or no config")
        return

    # 先用 YAML 值，再退回環境變數
    bot_token = telegram_conf.get("bot_token") or os.getenv(telegram_conf.get("bot_token_env", "BOT_TOKEN"))
    chat_id = telegram_conf.get("chat_id") or os.getenv(telegram_conf.get("chat_id_env", "CHAT_ID"))

    if not bot_token or not chat_id:
        log.error("notify: BOT_TOKEN or CHAT_ID missing (both config & env)")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
        if resp.status_code != 200:
            log.error(f"notify: HTTP {resp.status_code}, resp={resp.text}")
        else:
            log.info("notify: telegram sent ok")
    except Exception as e:
        log.exception(f"notify: telegram exception: {e}")


def notify_signal(
    symbol: str,
    signal: dict,
    price: float,
    telegram_conf: dict | None = None,
    tz: str = "Asia/Taipei",
) -> None:
    """Notify latest tradable signal."""
    ts = signal.get("ts")
    local = _fmt_local(ts, tz)
    side = signal.get("side", "NONE")
    score = _fmt_num(signal.get("score", 0.0), 2)
    pu = _fmt_num(signal.get("prob_up_max", 0.0), 2)
    pd = _fmt_num(signal.get("prob_down_max", 0.0), 2)
    h = signal.get("chosen_h")
    t = _fmt_pct(signal.get("chosen_t", 0.0), 2)
    msg = (
        f"[🔔 最新訊號] {symbol} @ {local}\n"
        f"side={side}  score={score}\n"
        f"prob_up_max={pu}  prob_down_max={pd}\n"
        f"chosen_h={h}  chosen_t={t}\n"
        f"price={_fmt_num(price, 2)}"
    )
    notify(msg, telegram_conf)


def notify_trade_open(
    symbol: str,
    side: str,
    entry_price: float,
    qty: float,
    sizing: dict,
    signal: dict | None = None,
    cfg: dict | None = None,
    tz: str = "Asia/Taipei",
) -> None:
    """Notify successful trade entry."""
    telegram_conf = None
    if cfg and isinstance(cfg, dict):
        telegram_conf = cfg.get("notify", {}).get("telegram") if "notify" in cfg else cfg
    ts = signal.get("ts") if signal else None
    local = _fmt_local(ts, tz)
    line1 = f"[🟢 進場] {symbol} {side} @ {_fmt_num(entry_price, 2)}  ({local})"
    line2 = f"qty={_fmt_num(qty, 4)}  mode={sizing.get('mode', '-') }"
    lev = sizing.get("leverage")
    if lev:
        line2 += f"  lev={_fmt_num(lev,0)}x"
    kf = sizing.get("kelly_f")
    if kf is not None:
        line2 += f"  kelly_f={_fmt_num(kf,2)}"
    risk = _fmt_pct(sizing.get("risk_per_trade", 0.0), 2)
    tp = _fmt_pct(sizing.get("tp_ratio", 0.0), 2)
    sl = _fmt_pct(-sizing.get("sl_ratio", 0.0), 2)
    line3 = f"risk={risk} of equity | TP={tp} SL={sl}"
    line4 = ""
    if signal:
        line4 = (
            f"basis: score={_fmt_num(signal.get('score', 0.0),2)}, "
            f"h={signal.get('chosen_h')}, t={_fmt_pct(signal.get('chosen_t',0.0),2)}, "
            f"prob_up_max={_fmt_num(signal.get('prob_up_max',0.0),2)}"
        )
    msg = "\n".join([line1, line2, line3] + ([line4] if line4 else []))
    notify(msg, telegram_conf)


def notify_trade_close(
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    opened_at: str,
    closed_at: str,
    reason: str,
    pnl_ratio: float,
    holding_minutes: float,
    telegram_conf: dict | None = None,
    tz: str = "Asia/Taipei",
) -> None:
    """Notify trade exit."""
    local = _fmt_local(closed_at, tz)
    pnl = _fmt_pct(pnl_ratio, 2)
    msg = (
        f"[✅ 平倉] {symbol} {side}  {reason}  ({local})\n"
        f"entry={_fmt_num(entry_price,2)} → exit={_fmt_num(exit_price,2)}  PnL={pnl}\n"
        f"hold={_fmt_num(holding_minutes,0)} min"
    )
    notify(msg, telegram_conf)


def notify_guard(
    event: str,
    detail: dict | None = None,
    telegram_conf: dict | None = None,
    tz: str = "Asia/Taipei",
) -> None:
    """Notify guard or abnormal events."""
    local = _fmt_local(None, tz)
    if event == "min_notional_reject" and detail:
        symbol = detail.get("symbol", "")
        side = detail.get("side", "")
        price = _fmt_num(detail.get("price", 0.0), 2)
        need = _fmt_num(detail.get("min", 0.0), 2)
        got = _fmt_num(detail.get("notional", 0.0), 2)
        msg = (
            f"[⚠️ 拒單] {symbol} {side} @ {price}  ({local})\n"
            f"reason=min_notional_reject  need≥{need} USDT, got={got}"
        )
    else:
        lines = [f"[🛑 風控] {event}  ({local})"]
        if detail:
            detail_line = " ".join(f"{k}={v}" for k, v in detail.items())
            lines.append(f"detail: {detail_line}")
        msg = "\n".join(lines)
    notify(msg, telegram_conf)

