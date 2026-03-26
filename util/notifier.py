import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import requests
from datetime import datetime
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TRADES_FILE = "logs/trades.csv"
ENABLE_LOG = True

def _fmt_pct(v):
    try:
        return f"{float(v):.2%}"
    except Exception:
        return str(v)

def _fmt_price(v):
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return str(v)

def notify(
    summary=None,
    signals=None,
    current_price=None,
    is_regression=None,
    holding=False,
    entry_price=None,
    entry_time=None,
    extra_msg=None,
    # 🔽 新增：策略/風控欄位（可不傳）
    side=None,
    sl=None, tp=None,
    sl_mult=None, tp_mult=None,
    tp_sl_mode=None,
    bars_held=None,          # 以「根」為單位
    reason=None,             # 出場原因（TP/SL/超時/其它）
    # 🔽 新增：市況快照（可不傳）
    atr_now=None, adx_now=None,
    high_vol=None, low_vol=None, strong_trend=None, ranging=None,
):
    lines = []

    # 1) 標題 + 價格
    if holding and entry_price and entry_time:
        lines.append("⚠️ BTC 尚在持倉中")
        lines.append(f"進場時間：{entry_time}")
        lines.append(f"進場價：{_fmt_price(entry_price)}")
        if current_price is not None:
            lines.append(f"目前價：{_fmt_price(current_price)}")
        if side:
            lines.append(f"方向：{side}")
        if bars_held is not None:
            lines.append(f"已持有：{int(bars_held)*15} 分鐘（{int(bars_held)} 根）")
    else:
        # 進場前/一般播報
        if summary:
            lines.append("📊 BTC 預測結果")
            for name, values in summary.items():
                try:
                    vals = ", ".join(_fmt_pct(v) for v in values)
                except TypeError:
                    vals = _fmt_pct(values)
                lines.append(f"{name}: {vals}")
        if signals is not None:
            lines.append("✅ 有進場訊號" if signals else "❌ 無明確進場訊號")
        if current_price is not None:
            lines.append(f"當前價格：{_fmt_price(current_price)}")

    # 2) TP/SL 與倍數（若有）
    if tp_sl_mode or sl is not None or tp is not None or sl_mult is not None or tp_mult is not None:
        tag = f"（{tp_sl_mode}）" if tp_sl_mode else ""
        lines.append(f"🎯 TP/SL{tag}")
        if tp is not None:
            lines.append(f"TP：{_fmt_price(tp)}")
        if sl is not None:
            lines.append(f"SL：{_fmt_price(sl)}")
        if (sl_mult is not None) or (tp_mult is not None):
            sm = f"{sl_mult:.2f}" if isinstance(sl_mult,(int,float)) else str(sl_mult)
            tm = f"{tp_mult:.2f}" if isinstance(tp_mult,(int,float)) else str(tp_mult)
            lines.append(f"倍數：SL×{sm} / TP×{tm}")

    # 3) 市況快照（若有）
    if any(v is not None for v in [atr_now, adx_now, high_vol, low_vol, strong_trend, ranging]):
        parts = []
        if atr_now is not None: parts.append(f"ATR={_fmt_price(atr_now)}")
        if adx_now is not None: parts.append(f"ADX={_fmt_price(adx_now)}")
        flags = []
        if high_vol:     flags.append("高波動")
        if low_vol:      flags.append("低波動")
        if strong_trend: flags.append("強趨勢")
        if ranging:      flags.append("震盪")
        if flags: parts.append(" / ".join(flags))
        if parts:
            lines.append("📈 市況：" + " | ".join(parts))

    # 4) 出場原因（若有）
    if reason:
        lines.append(f"🔔 出場原因：{reason}")

    # 5) 額外訊息
    if extra_msg:
        lines.append("")
        lines.append(str(extra_msg).strip())

    message = "\n".join(lines).strip()

    # 🖨️ 控制台
    print(message)

    # 📨 Telegram
    try:
        requests.post(
            f"https://api.telegram.org/bot{config['BOT_TOKEN']}/sendMessage",
            json={"chat_id": config['CHAT_ID'], "text": message}
        )
    except Exception as e:
        print(f"❌ Telegram 發送失敗：{e}")

    # 📝 紀錄 log
    if ENABLE_LOG and TRADES_FILE:
        try:
            dir_path = os.path.dirname(TRADES_FILE)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(TRADES_FILE, "a", encoding="utf-8") as f:
                safe_message = message.replace('"', '""')  # CSV 雙引號轉義
                f.write(f'{datetime.now()},"{safe_message.strip()}"\n')
        except Exception as e:
            print(f"❌ Log 紀錄失敗：{e}")
