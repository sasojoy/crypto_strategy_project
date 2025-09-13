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


def render_multi_signals(signals: dict, build: str = "-", host: str = "-") -> str:
    lines = [f"⏱️ 多幣別即時訊號 (build={build}, host={host})"]
    for sym in sorted(signals.keys()):
        s = signals.get(sym, {}) or {}
        lines.append(
            f"{sym}: {s.get('side','-')}"
            f" | score={_fmt(s.get('score'), '.3f')}"
            f" | h={_fmt(s.get('chosen_h'))}"
            f" | pt={_fmt(s.get('chosen_t'), '+.2%')}"
            f" | ↑={_fmt(s.get('prob_up_max'), '.2%')} ↓={_fmt(s.get('prob_down_max'), '.2%')}"
            f" | price={_fmt(s.get('price'), ',.2f')}"
            f" | reason={_fmt(s.get('reason'))}"
        )
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
