import json, os, re
from scripts.notify_telegram import send_telegram_message


def main():
    msg = "❌ CI失敗：沒有找到 logs/ci_run.json"
    try:
        log_path = os.path.join("logs", "ci_run.json")
        if not os.path.exists(log_path):
            raise FileNotFoundError(log_path)
        data = json.load(open(log_path, "r", encoding="utf-8"))
        hit = bool(data.get("hit_target", False))
        best = data.get("best", {})
        # 盡力在 steps tail 裡找 "[SUMMARY ALL] {json}"，若有就原樣發送
        summary_line = None
        for s in data.get("steps", []):
            for line in s.get("tail", "").splitlines():
                if re.match(r"^\[SUMMARY ALL\]", line):
                    summary_line = line
        if summary_line:
            msg = summary_line  # 你要的格式
        else:
            win = float(best.get("win_rate", 0.0))
            ret = float(best.get("total_return", 0.0))
            badge = "✅ 達標" if hit else "❌ 未達標"
            msg = f"[RESULT] {badge} | win_rate={win:.2%} total_return={ret:.2%} | thr={best.get('threshold')}"
    except Exception as e:
        msg = f"❌ CI失敗：讀取/解析 logs 失敗：{e}"
    print(msg)
    send_telegram_message(msg)


if __name__ == "__main__":
    main()
