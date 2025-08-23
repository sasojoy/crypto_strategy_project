import os
import yaml


def load_cfg(cfg_or_path):
    """
    接受 YAML 路徑 (str) 或 dict，最後一律回傳 dict。
    用於所有 train/backtest/realtime 腳本與內部 train() 等入口。
    """
    if isinstance(cfg_or_path, dict):
        return cfg_or_path
    if isinstance(cfg_or_path, str) and os.path.exists(cfg_or_path):
        with open(cfg_or_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise ValueError(
        f"Invalid cfg input (expect dict or existing path): {type(cfg_or_path)} -> {cfg_or_path}"
    )
