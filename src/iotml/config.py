"""Load YAML settings and resolve project paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``override`` merged in, recursing into nested dicts."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file; an empty file gives an empty dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def load_config(*profiles: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load ``default.yaml`` and then each named profile (e.g. ``"quick"``) on top of it."""
    config = read_yaml(config_dir / "default.yaml")
    for name in profiles:
        config = deep_merge(config, read_yaml(config_dir / f"{name}.yaml"))
    return config


def load_labels(config_dir: Path = CONFIG_DIR) -> dict[str, str]:
    """Return a mapping from each Label to its Category, read from ``labels.yaml``."""
    categories = read_yaml(config_dir / "labels.yaml")["categories"]
    mapping: dict[str, str] = {}
    for category, labels in categories.items():
        for label in labels:
            if label in mapping:
                raise ValueError(f"Label {label!r} is listed under two categories")
            mapping[label] = category
    return mapping


def project_path(relative: str | Path) -> Path:
    """Resolve a path from the config against the project root."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
