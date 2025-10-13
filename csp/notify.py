import os
import logging
import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
log = logging.getLogger("notify")


def _fmt(val, fmt=None, dash="-"):
    if val is None:
        return dash
    try:
        if fmt:
            return format(val, fmt)
        return str(val)
    except Exception:
        return dash


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "-"
    try:
        return f"{float(val)*100:.2f}%"
    except Exception:
        return "-"


def render_multi_signals(signals: dict, build: str = "-", host: str = "-") -> str:
    lines = [f"⏱️ 多幣別即時訊號 (build={build}, host={host})"]
    for sym in sorted(signals.keys()):
        s = signals.get(sym, {}) or {}
        thr = s.get("threshold")
        try:
            thr_fmt = f"{float(thr):.2f}"
        except Exception:
            thr_fmt = "-"
        line = (
            f"{sym}: {s.get('side', 'NONE')} | "
            f"score={_fmt(s.get('score'), '.3f')} (thr={thr_fmt}) | "
            f"h={_fmt(s.get('h'))} | "
            f"pt={_fmt_pct(s.get('pt'))} sl={_fmt_pct(s.get('sl'))} | "
            f"↑={_fmt(s.get('up_price'), ',.2f')} ↓={_fmt(s.get('down_price'), ',.2f')} | "
            f"price={_fmt(s.get('price'), ',.2f')}"
        )
        reason = s.get("reason")
        if reason:
            line += f" | reason={reason}"
        lines.append(line)
    return "\n".join(lines)


def notify(text_or_payload, build: str = "-", host: str = "-"):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.info("notify: telegram disabled or no config")
        return False
    text = (
        render_multi_signals(text_or_payload, build, host)
        if isinstance(text_or_payload, dict)
        else str(text_or_payload)
    )
    r = requests.post(TELEGRAM_API.format(token=token), json={"chat_id": chat_id, "text": text})
    if not r.ok:
        log.warning("notify: telegram not ok %s %s", r.status_code, r.text)
    return r.ok
