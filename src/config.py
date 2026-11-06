"""Load config.yaml from the repo root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=REPO_ROOT / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(p):
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
