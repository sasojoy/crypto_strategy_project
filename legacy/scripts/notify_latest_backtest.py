#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/notify_latest_backtest.py
讀取 logs/ci_run.json，將「最新回測摘要」推送到 Telegram。
優先推送 [SUMMARY ALL] {...} 的原文；沒有則以 best 指標組合訊息。
"""
import os, sys, json, glob
from datetime import datetime, timezone

# 允許從包或直呼腳本 import
try:
    from scripts.notify_telegram import send_telegram_message
except Exception:
    def send_telegram_message(text: str, token: str = None, chat_id: str = None) -> bool:
        import requests
        tok = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if not tok or not chat:
            print("[TG][WARN] Missing TELEGRAM_*"); return False
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "Markdown"}, timeout=20)
        print("[TG]", r.status_code, r.text[:200]); return r.ok

def find_ci_log():
    # 先看常見路徑，再全域搜尋
    for p in ("logs/ci_run.json", "artifacts/ci_run.json"):
        if os.path.exists(p): return p
    for p in glob.glob("**/ci_run.json", recursive=True):
        return p
    return None

def extract_summary_from_log(d: dict):
    summary_line = None
    for s in d.get("steps", []):
        tail = s.get("tail", "") or s.get("stdout", "")
        for line in tail.splitlines():
            if line.startswith("[SUMMARY ALL]"):
                summary_line = line.strip()
    best = d.get("best", {}) or {}
    hit = bool(d.get("hit_target", False))
    return summary_line, best, hit

def format_msg(summary_line, best, hit, meta):
    if summary_line:
        header = "✅ 達標" if hit else "❌ 未達標"
        return f"{header}\n{summary_line}\n{meta}"
    win = best.get("win_rate"); ret = best.get("total_return"); thr = best.get("threshold")
    try:
        win_s = f"{float(win):.2%}" if win is not None else "n/a"
        ret_s = f"{float(ret):.2%}" if ret is not None else "n/a"
    except Exception:
        win_s, ret_s = str(win), str(ret)
    header = "✅ 達標" if hit else "❌ 未達標"
    return f"{header} | win_rate={win_s} total_return={ret_s} | thr={thr}\n{meta}"

def main():
    path = find_ci_log()
    if not path:
        send_telegram_message("❌ 找不到 logs/ci_run.json，無法彙報最新回測結果。")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    summary_line, best, hit = extract_summary_from_log(d)
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    run_at = d.get("start_utc") or d.get("start") or datetime.now(timezone.utc).isoformat()
    meta = f"(build={sha} at {run_at})"
    msg = format_msg(summary_line, best, hit, meta)
    print(msg)
    ok = send_telegram_message(msg)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
