import os
import requests
from csp.utils.logger import get_logger

log = get_logger("notify")

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
