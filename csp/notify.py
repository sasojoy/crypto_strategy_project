import os
import logging
import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
log = logging.getLogger("notify")


def _fmt(val, fn=str, dash="-"):
    try:
        return fn(val) if val is not None else dash
    except Exception:
        return dash


def _fmt_pct(x):
    return f"{float(x):.2%}"


def _fmt_score(x):
    return f"{float(x):.3f}"


def _fmt_price(x):
    return f"{float(x):,.2f}"


def render_multi_signals(signals: dict, build: str = "-", host: str = "-") -> str:
    """
    signals: {
      "BTCUSDT": {
         "side": "LONG", "score": 0.9, "price": 114273.61,
         "chosen_h": 16, "chosen_t": 0.0012,
         "prob_up_max": 0.53, "prob_down_max": 0.12,
         "reason": "-"
      }, ...
    }
    """
    lines = [f"⏱️ 多幣別即時訊號 (build={build}, host={host})"]
    for sym in sorted(signals.keys()):
        s = signals[sym] or {}
        lines.append(
            f"{sym}: {_fmt(s.get('side'))}"
            f" | score={_fmt(s.get('score'), _fmt_score)}"
            f" | h={_fmt(s.get('chosen_h'))}"
            f" | pt={_fmt(s.get('chosen_t'), _fmt_pct)}"
            f" | ↑={_fmt(s.get('prob_up_max'), _fmt_pct)} ↓={_fmt(s.get('prob_down_max'), _fmt_pct)}"
            f" | price={_fmt(s.get('price'), _fmt_price)}"
            f" | reason={_fmt(s.get('reason'))}"
        )
    return "\n".join(lines)


def notify(text_or_payload, build: str = "-", host: str = "-"):
    """
    支援：
      - 傳入字串：直接送
      - 傳入 dict（multi-symbol payload）：會自動格式化為多行訊息
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.info("notify: telegram disabled or no config")
        return False
    if isinstance(text_or_payload, dict):
        text = render_multi_signals(text_or_payload, build, host)
    else:
        text = str(text_or_payload)
    r = requests.post(TELEGRAM_API.format(token=token), json={"chat_id": chat_id, "text": text})
    if not r.ok:
        log.warning("notify: telegram response not ok: %s %s", r.status_code, r.text)
    return r.ok

