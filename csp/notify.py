import os
import requests
import logging

log = logging.getLogger("notify")

def telegram_enabled() -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    return bool(token and chat_id)

def telegram_send(text: str) -> bool:
    """
    Send message to Telegram. Never raise; log and return False on failure.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.info("notify: telegram disabled or no config")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=10)
        ok = False
        resp = None
        try:
            resp = r.json()
            ok = bool(resp.get("ok"))
        except Exception:
            resp = {"text": r.text[:200]}
        if not ok:
            log.warning("notify: telegram send failed status=%s resp=%s", r.status_code, resp)
        return ok
    except requests.RequestException as e:
        log.warning("notify: telegram exception=%s msg=%s", type(e).__name__, e)
        return False


def format_multi_signals(build: str, host: str, items: list[dict]) -> str:
    """
    items: [{"symbol":"BTCUSDT","side":"LONG|SHORT|NONE","score":0.957,"reason":"-","chosen_h":None,"proba":None}, ...]
    """
    lines = ["⏱️ 多幣別即時訊號 (build=%s, host=%s)" % (build, host)]
    for it in items:
        sym = it.get("symbol", "?")
        side = it.get("side", "?")
        score = it.get("score", 0.0)
        reason = it.get("reason", "-")
        chosen_h = it.get("chosen_h", "-")
        proba = it.get("proba", "-")
        lines.append(f"{sym}: {side} | score={score:.3f} | chosen_h={chosen_h} | proba={proba} | reason={reason}")
    return "\n".join(lines)
