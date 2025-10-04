"""Utility helpers for sending Telegram notifications."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import requests


def send_telegram_message(
    token: str,
    chat_id: str,
    message: str,
    parse_mode: str | None = "Markdown",
    disable_notification: bool = False,
) -> Dict[str, Any]:
    if not token:
        raise ValueError("Telegram token is required")
    if not chat_id:
        raise ValueError("Telegram chat_id is required")
    if not message:
        raise ValueError("message must not be empty")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": message,
        "disable_notification": disable_notification,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    response = requests.post(url, json=payload, timeout=15)
    response.raise_for_status()
    data = response.json()
    if not data.get("ok", False):
        raise RuntimeError(f"Telegram API returned error: {data}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a Telegram message")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"))
    parser.add_argument("--message", required=True)
    parser.add_argument("--disable-notification", action="store_true")
    parser.add_argument("--no-parse-mode", action="store_true")

    args = parser.parse_args()
    parse_mode = None if args.no_parse_mode else "Markdown"
    send_telegram_message(
        token=args.token,
        chat_id=args.chat_id,
        message=args.message,
        parse_mode=parse_mode,
        disable_notification=args.disable_notification,
    )
    print("Message sent")


if __name__ == "__main__":
    main()
