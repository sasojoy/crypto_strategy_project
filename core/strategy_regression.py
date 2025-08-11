ENTRY_THRESHOLD = 0.003  # 0.3%
TAKE_PROFIT = 0.05       # +5%
STOP_LOSS = -0.02        # -2%

class StrategyManager:
    def __init__(self):
        self.entry_threshold = ENTRY_THRESHOLD
        self.take_profit = TAKE_PROFIT
        self.stop_loss = STOP_LOSS

    def should_enter(self, prediction_summary, index, rsi=None):
        if not prediction_summary:
            print("⛔ 無預測結果，略過進場判斷")
            return False

        try:
            max_horizon = max(prediction_summary, key=lambda h: prediction_summary[h][0])
            max_value = prediction_summary[max_horizon][0]
        except Exception as e:
            print(f"❌ 進場判斷錯誤：{e}")
            return False

        if max_value >= self.entry_threshold:
            print(f"📈 進場條件成立：h={max_horizon}, 預測值={max_value:.4f}")
            return True
        else:
            print(f"⏸️ 無進場訊號，最大值={max_value:.4f}")
            return False

    def should_exit(self, entry_price, current_price, holding_minutes, prediction_summary, entry_index=None):
        if not prediction_summary:
            print("⚠️ 預測為空，保守出場")
            return True

        try:
            max_pred = max([p[0] for p in prediction_summary.values()])
        except Exception as e:
            print(f"❌ 出場判斷錯誤：{e}")
            return True

        change = (current_price / entry_price) - 1

        if change >= self.take_profit:
            print(f"🎯 出場：達成停利 +{change:.2%}")
            return True
        elif change <= self.stop_loss:
            print(f"🛑 出場：觸發停損 {change:.2%}")
            return True
        elif max_pred < self.entry_threshold:
            print(f"🔽 出場：預測值下降 {max_pred:.4f} < 門檻")
            return True
        elif holding_minutes >= 240:
            print(f"⏰ 出場：持倉超過 240 分鐘")
            return True

        return False
