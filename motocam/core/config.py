"""Loads /etc/motorcam/config.yaml (or a local dev copy) into a plain dict.

Kept intentionally simple (no schema validation library) -- every module
reads only the keys it needs via .get() with a sane default, so a partial
or dev-only config never crashes the app.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATHS = [
    Path("/etc/motorcam/config.yaml"),
    Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml",
]


def resolve_config_path(path: str | os.PathLike | None = None) -> Path:
    candidates = [Path(path)] if path else DEFAULT_CONFIG_PATHS
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No config file found. Looked in: {', '.join(str(c) for c in candidates)}"
    )


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    candidate = resolve_config_path(path)
    with candidate.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_config(config: dict[str, Any], path: str | os.PathLike) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
