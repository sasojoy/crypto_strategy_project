#!/usr/bin/env python3
import os, sys, json, urllib.parse, requests

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    text = " ".join(sys.argv[1:]) or "smoke"
    if not token or not chat_id:
        print("[SMOKE] missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID")
        return 0
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=8)
        ok = False
        try:
            data = r.json()
            ok = bool(data.get("ok"))
            desc = data.get("description")
        except Exception:
            desc = r.text[:300]
        print(f"[SMOKE] status={r.status_code} ok={ok} desc={desc}")
    except Exception as e:
        print(f"[SMOKE] exception={type(e).__name__} msg={e}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
