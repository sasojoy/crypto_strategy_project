from __future__ import annotations
import os
from typing import Dict, Any, List
from csp.utils.diag import log_diag


def _as_list(v) -> List[dict]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def load_models_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    items = _as_list((cfg or {}).get("models"))
    if not items:
        log_diag("model_hub: models list empty in cfg")
        return models

    for item in items:
        name = str(item.get("name") or "").strip()
        mtype = str(item.get("type") or "").strip().lower()
        path = str(item.get("path") or "").strip()
        if not name:
            log_diag(f"model_hub: skip unnamed model: {item}")
            continue
        if not path:
            log_diag(f"model_hub: model '{name}' missing path")
            continue
        if not os.path.exists(path):
            log_diag(f"model_hub: path not found for '{name}': {path}")
            continue
        # TODO: replace with actual load; return metadata for now
        models[name] = {"type": mtype, "path": path}
        log_diag(f"model_hub: loaded '{name}' ({mtype}) from {path}")
    if not models:
        log_diag("model_hub: no valid models loaded (all skipped)")
    return models
