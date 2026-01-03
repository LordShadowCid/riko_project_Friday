from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import os
import yaml


def _find_repo_root(start: Path) -> Path:
    """Walk upward until we find the repo root marker files."""
    current = start.resolve()
    while True:
        if (current / "character_config.yaml").exists() or (current / ".git").exists():
            return current
        if current.parent == current:
            return start.resolve()
        current = current.parent


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "character_config.yaml"


def repo_root() -> Path:
    return _REPO_ROOT


def load_config(config_path: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    """Load the character config.

    Resolution order:
    1) explicit arg
    2) env var ANNABETH_CONFIG_PATH
    3) repo-root character_config.yaml
    """
    if config_path is None:
        config_path = os.environ.get("ANNABETH_CONFIG_PATH")

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = _REPO_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at '{path}'. Set ANNABETH_CONFIG_PATH or create character_config.yaml in repo root."
        )

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_repo_path(value: str | os.PathLike[str]) -> str:
    """Resolve a path that may be relative to the repo root into an absolute path string."""
    p = Path(value)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return str(p)
