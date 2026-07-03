"""Health check / supervisor (design doc 19.2).

Polls lightweight system stats (CPU temp/load, RAM) for the diagnostics
screen and telemetry payload. Uses psutil where available; falls back
to None fields on platforms where the sysfs thermal path doesn't exist
(e.g. macOS dev machines) rather than faking a temperature reading.
"""
from __future__ import annotations

import logging
from pathlib import Path

from motocam.core.protocol import SystemTelemetry

logger = logging.getLogger("motocam.watchdog")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

RPI_THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


class HealthMonitor:
    def __init__(self):
        self._video_fps = 0.0

    def set_video_fps(self, fps: float) -> None:
        self._video_fps = fps

    def sample(self) -> SystemTelemetry:
        return SystemTelemetry(
            cpu_temp_c=self._read_cpu_temp(),
            cpu_load_pct=self._safe_psutil(lambda: psutil.cpu_percent(interval=None)),
            ram_used_pct=self._safe_psutil(lambda: psutil.virtual_memory().percent),
            video_fps=self._video_fps,
        )

    @staticmethod
    def _safe_psutil(fn) -> float | None:
        # On some macOS versions (seen on 26/27 betas) psutil's mach struct
        # sizes lag the kernel and every call raises RuntimeError -- that's
        # a dev-machine annoyance, not something that should ever take the
        # whole control unit down (this used to escape a QTimer slot and
        # abort the process via PyQt6's default fatal-in-slot behaviour).
        if not PSUTIL_AVAILABLE:
            return None
        try:
            return fn()
        except (RuntimeError, OSError) as exc:
            logger.warning("psutil call failed (%s), reporting as unavailable", exc)
            return None

    @staticmethod
    def _read_cpu_temp() -> float | None:
        if RPI_THERMAL_PATH.is_file():
            try:
                return int(RPI_THERMAL_PATH.read_text().strip()) / 1000.0
            except (ValueError, OSError):
                return None
        return None
