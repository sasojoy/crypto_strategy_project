from __future__ import annotations
import json
import glob
import os
from pathlib import Path
from typing import Dict, Any
import joblib
from csp.utils.diag import log_diag


def _read_manifest(path: str) -> Dict[str, str] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", {})
        return {sym: m.get("path") for sym, m in models.items() if m.get("path")}
    except Exception as e:
        log_diag(f"model_hub: manifest read failed: {e}")
        return None


def load_models(cfg: Dict[str, Any]) -> Dict[str, Any]:
    models_cfg = (cfg or {}).get("models", {})
    loaded: Dict[str, Any] = {}
    paths: Dict[str, str] = {}

    if models_cfg.get("use_manifest"):
        manifest_path = models_cfg.get("manifest_path", "models/manifest.json")
        paths = _read_manifest(manifest_path) or {}

    if not paths:
        for pattern in models_cfg.get("globs", []):
            for p in glob.glob(pattern):
                sym = Path(p).parent.name
                paths[sym] = p

    loaded_syms = []
    for sym, p in paths.items():
        try:
            obj = joblib.load(p)
            loaded[sym] = obj
            loaded_syms.append(sym)
        except Exception as e:
            log_diag(f"model_hub: failed to load {sym} from {p}: {e}")

    log_diag(f"model_hub: loaded={loaded_syms} count={len(loaded)}")
    return loaded
