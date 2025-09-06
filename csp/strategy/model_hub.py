from __future__ import annotations
import os, sys
from typing import Dict, Any
from csp.utils.diag import log_diag

def load_models_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    根據 cfg['models'] 載入模型，回傳 dict。
    支援最簡版本：只把路徑存在、可讀取當作已載入（若需要，之後可替換成真實的 model.load(...)）。
    結果為空時不丟例外；上層會用 reason=no_models_loaded 早退。
    """
    models = {}
    models_cfg = (cfg or {}).get("models", [])
    if not models_cfg:
        log_diag("model_hub: models list empty in cfg")
        return models

    for item in models_cfg:
        name = str(item.get("name") or "").strip()
        mtype = str(item.get("type") or "").strip().lower()
        path = str(item.get("path") or "").strip()
        if not name:
            log_diag(f"model_hub: skip unnamed model entry: {item}")
            continue
        if not path:
            log_diag(f"model_hub: model '{name}' missing path")
            continue
        if not os.path.exists(path):
            log_diag(f"model_hub: model '{name}' path not found: {path}")
            continue

        # TODO: 這裡可替換成實際框架的 load(...)，目前先保留路徑與型別
        models[name] = {"type": mtype, "path": path}
        log_diag(f"model_hub: loaded '{name}' ({mtype}) from {path}")
    if not models:
        log_diag("model_hub: no valid models loaded (all skipped)")
    return models
