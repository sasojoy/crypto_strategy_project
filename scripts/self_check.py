from pathlib import Path
import json, hashlib, inspect
import csp

root = Path(__file__).resolve().parents[1]
app = root / "scripts" / "realtime_loop.py"
sha8 = hashlib.sha1(app.read_bytes()).hexdigest()[:8]
try:
    rel = json.loads((root/"RELEASE.json").read_text())
    build = (rel.get("sha") or sha8)[:8]
    branch = rel.get("branch") or "unknown"
    built_at = rel.get("built_at") or "unknown"
except Exception:
    build, branch, built_at = sha8, "unknown", "unknown"

print("[SELF-CHECK]")
print("root =", root)
print("build =", build, "file_sha8 =", sha8, "branch =", branch, "built_at =", built_at)
print("csp_module_path =", inspect.getfile(csp))
