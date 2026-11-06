"""Write the *.meta.json that sits next to every script output."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from src.config import REPO_ROOT


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def write_meta(output_path, config, packages=(), extra=None):
    """Record what produced output_path: git commit (and whether the tree had uncommitted changes),
    the full config, the command line, and versions of the given packages."""
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    meta = {
        "output": str(output_path),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": sys.argv,
        "host": platform.node(),
        "python": platform.python_version(),
        "packages": versions,
        "config": config,
    }
    if extra:
        meta.update(extra)
    meta_path = str(output_path).rsplit(".", 1)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path
