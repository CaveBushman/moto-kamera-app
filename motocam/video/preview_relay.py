"""Bandwidth policy for the low-fps JPEG preview relay to control room."""
from __future__ import annotations


def preview_interval_s(fps: float | int | None, default_fps: float = 5.0) -> float:
    try:
        value = float(fps)
    except (TypeError, ValueError):
        value = default_fps
    value = max(0.5, min(30.0, value))
    return 1.0 / value


def preview_jpeg_quality(quality: int | str | None, default: int = 55) -> int:
    try:
        value = int(quality)
    except (TypeError, ValueError):
        value = default
    return max(20, min(90, value))


def scaled_preview_size(width: int, height: int, max_width: int | str | None) -> tuple[int, int]:
    try:
        limit = int(max_width)
    except (TypeError, ValueError):
        limit = 0
    if width <= 0 or height <= 0 or limit <= 0 or width <= limit:
        return width, height
    ratio = limit / width
    return limit, max(1, int(round(height * ratio)))
