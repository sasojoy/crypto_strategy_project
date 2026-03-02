# NEW utility to resolve absolute resources dir from repo root
import os

def repo_root() -> str:
    # assume this file is under csp/utils/; go two levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def resolve_resources_dir(cfg: dict) -> str:
    rt = (cfg or {}).get("realtime", {})
    cfg_dir = rt.get("resources_dir") or (cfg or {}).get("resources_dir", "resources")
    if os.path.isabs(cfg_dir):
        return cfg_dir
    return os.path.join(repo_root(), cfg_dir)
