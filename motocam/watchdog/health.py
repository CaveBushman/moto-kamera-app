"""Health check / supervisor (design doc 19.2).

Polls lightweight system stats (CPU temp/load, RAM) for the diagnostics
screen and telemetry payload. Uses psutil where available; falls back
to None fields on platforms where the sysfs thermal path doesn't exist
(e.g. macOS dev machines) rather than faking a temperature reading.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
import threading
import time
from pathlib import Path

from motocam.core.protocol import SystemTelemetry

logger = logging.getLogger("motocam.watchdog")
DEFAULT_SAMPLE_INTERVAL_S = 1.5

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

RPI_THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


class HealthMonitor:
    """Rate-limited system stats sampler. Use `latest()` from the UI thread
    (never blocks); `sample()` is the blocking variant for tests/scripts."""

    def __init__(self, sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S):
        self._video_fps = 0.0
        self._sample_interval_s = max(0.0, float(sample_interval_s))
        self._last_sample_at = 0.0
        self._last_sample: SystemTelemetry | None = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motocam-health")
        self._sample_future: Future | None = None

    def set_video_fps(self, fps: float) -> None:
        self._video_fps = fps
        with self._lock:
            if self._last_sample is not None:
                self._last_sample.video_fps = fps

    def sample(self) -> SystemTelemetry:
        """Blocking sample for tests and non-UI callers.

        The main window uses latest() so psutil/sysfs calls can never block
        the Qt event loop.
        """
        now = time.monotonic()
        with self._lock:
            if self._last_sample is not None and now - self._last_sample_at < self._sample_interval_s:
                return self._last_sample
        sample = self._collect_sample()
        with self._lock:
            self._last_sample = sample
            self._last_sample_at = now
            return self._last_sample

    def latest(self) -> SystemTelemetry:
        """Return cached stats immediately and refresh them in the background.

        This is the UI-safe path. On macOS psutil can occasionally block or
        raise from kernel struct mismatches; on the Ri field unit sysfs/psutil
        are normally quick, but they still should not run from a 500 ms Qt
        telemetry timer.
        """
        now = time.monotonic()
        new_future: Future | None = None
        with self._lock:
            if self._last_sample is None:
                self._last_sample = SystemTelemetry(video_fps=self._video_fps)
                self._last_sample_at = 0.0
            cached = self._last_sample
            due = now - self._last_sample_at >= self._sample_interval_s
            running = self._sample_future is not None and not self._sample_future.done()
            if due and not running:
                new_future = self._executor.submit(self._collect_sample)
                self._sample_future = new_future
        # add_done_callback must run outside the lock: if _collect_sample
        # already finished on the executor thread by the time we get here,
        # the callback (_on_sample_done, which also takes self._lock) fires
        # synchronously on THIS thread right inside add_done_callback --
        # re-entering a plain (non-reentrant) Lock and deadlocking the Qt UI
        # thread against itself. This raced rarely enough in testing to look
        # like an intermittent multi-second-to-forever freeze right at
        # startup (first telemetry tick), which is exactly what it was.
        if new_future is not None:
            new_future.add_done_callback(self._on_sample_done)
        return cached

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _collect_sample(self) -> SystemTelemetry:
        return SystemTelemetry(
            cpu_temp_c=self._read_cpu_temp(),
            cpu_load_pct=self._safe_psutil(lambda: psutil.cpu_percent(interval=None)),
            ram_used_pct=self._safe_psutil(lambda: psutil.virtual_memory().percent),
            video_fps=self._video_fps,
        )

    def _on_sample_done(self, future: Future) -> None:
        try:
            sample = future.result()
        except Exception as exc:  # noqa: BLE001 -- background health must never take down UI
            logger.warning("health sample failed (%s)", exc)
            with self._lock:
                if self._sample_future is future:
                    self._sample_future = None
            return
        with self._lock:
            self._last_sample = sample
            self._last_sample_at = time.monotonic()
            if self._sample_future is future:
                self._sample_future = None

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
