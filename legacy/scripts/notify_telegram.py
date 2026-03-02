import os
import requests


def send_telegram_message(text: str, token: str = None, chat_id: str = None) -> bool:
    tok = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[TG][WARN] Missing TELEGRAM_*"); return False
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "Markdown"}, timeout=20)
    print("[TG]", r.status_code, r.text[:200]); return r.ok


if __name__ == "__main__":
    import sys; send_telegram_message(sys.argv[1] if len(sys.argv)>1 else "(empty)")
