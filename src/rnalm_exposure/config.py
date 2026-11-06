"""Load configs/config.yaml and resolve the paths in it."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(p):
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def display_path(p):
    """Path relative to the repo root when inside it, so outputs don't record machine-specific paths."""
    p = Path(p).resolve()
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


def rnacentral_file(config, release, file):
    paths = config["paths"]
    return repo_path(paths["rnacentral_fasta"].format(raw_dir=paths["raw_dir"], release=release, file=file))


def candidate_files(config, candidate):
    """Paths of the RNAcentral files that make up one RNA-FM corpus candidate."""
    return [rnacentral_file(config, candidate["release"], f) for f in candidate["files"]]
