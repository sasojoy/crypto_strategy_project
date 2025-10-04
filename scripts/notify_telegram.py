"""Telegram notification helpers.

This module can be imported both from scripts and from other modules while
still supporting CLI usage.  The :func:`send_telegram_message` helper reads the
Telegram credentials from arguments or environment variables, which allows it to
be reused in CI contexts where the script is executed as a module.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import requests

__all__ = ["send_telegram_message"]


def send_telegram_message(text: str, token: str | None = None, chat_id: str | None = None) -> Dict[str, Any]:
    """Send a Telegram message using the Bot API.

    Parameters
    ----------
    text:
        Message body to send.  Markdown formatting is enabled by default.
    token:
        Telegram bot token.  Falls back to the ``TELEGRAM_BOT_TOKEN``
        environment variable when omitted.
    chat_id:
        Target chat ID.  Falls back to the ``TELEGRAM_CHAT_ID`` environment
        variable when omitted.
    """

    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    if not token:
        raise ValueError("Telegram token is required")
    if not chat_id:
        raise ValueError("Telegram chat_id is required")
    if not text:
        raise ValueError("text must not be empty")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }

    response = requests.post(url, json=payload, timeout=15)
    response.raise_for_status()

    data = response.json()
    if not data.get("ok", False):
        raise RuntimeError(f"Telegram API returned error: {data}")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a Telegram message")
    parser.add_argument("--message", required=True, help="Message text")
    parser.add_argument("--token", default=None, help="Telegram bot token")
    parser.add_argument("--chat-id", default=None, help="Telegram chat ID")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    send_telegram_message(args.message, token=args.token, chat_id=args.chat_id)
    print("Message sent")


if __name__ == "__main__":
    main()
