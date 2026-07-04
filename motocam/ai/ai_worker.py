"""Runs object detection off the Qt UI thread (stability, design doc 10.4).

The Hailo inference call is a *blocking* wait -- normally tens of
milliseconds, but up to a full timeout on a stall. It used to be called
straight from the video frame callback, which runs on the Qt UI thread,
so any accelerator hiccup froze the entire interface (touch included)
for as long as the wait lasted.

This worker owns the inference loop on its own thread. The UI thread
hands it the latest frame via `submit()` -- any previous un-processed
frame is dropped, so a slow detector never builds a backlog -- and
detections come back through the `detections_ready` signal, delivered to
the UI thread as a queued connection. The UI thread now only ever does
light work per frame (draw + overlay); inference can be as slow as it
likes without ever stalling the display.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from motocam.ai.ai_engine import AiEngine

logger = logging.getLogger("motocam.ai.worker")
UTIL_WINDOW_S = 2.0


class AiWorker(QThread):
    detections_ready = pyqtSignal(object)  # list[Detection]

    def __init__(self, ai_engine: AiEngine, max_fps: float = 0.0, parent=None):
        super().__init__(parent)
        self._engine = ai_engine
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._wake = threading.Event()
        self._running = True
        self._max_fps = 0.0
        self._min_submit_interval_s = 0.0
        self._last_accept_at = 0.0
        self._submitted_frames = 0
        self._accepted_frames = 0
        self._dropped_frames = 0
        self._drop_samples: list[float] = []
        self._completed_frames = 0
        self._failed_frames = 0
        self._last_inference_ms: float | None = None
        self._busy_samples: list[tuple[float, float]] = []  # (end_monotonic, duration_s)
        self.set_max_fps(max_fps)

    def submit(self, frame: np.ndarray) -> bool:
        """Hand the newest frame to the worker, dropping any previous
        un-processed one -- realtime tracking wants the freshest frame, not
        a backlog of stale ones.

        Returns True when the frame was accepted. False means it was
        deliberately dropped by the max-FPS limiter.
        """
        now = time.monotonic()
        with self._lock:
            self._submitted_frames += 1
            if self._min_submit_interval_s > 0 and now - self._last_accept_at < self._min_submit_interval_s:
                self._record_drop_locked(now)
                return False
            if self._latest is not None:
                self._record_drop_locked(now)
            self._latest = frame
            self._accepted_frames += 1
            self._last_accept_at = now
        self._wake.set()
        return True

    def _take(self) -> np.ndarray | None:
        """Atomically fetch and clear the pending frame (the drop step:
        only ever the most recent submitted frame survives)."""
        with self._lock:
            frame = self._latest
            self._latest = None
        return frame

    def run(self) -> None:  # noqa: D401 -- QThread entry point
        while self._running:
            self._wake.wait()
            if not self._running:
                break
            self._wake.clear()
            frame = self._take()
            if frame is None:
                continue
            start = time.monotonic()
            failed = False
            try:
                detections = self._engine.process(frame)
            except Exception as exc:  # noqa: BLE001 -- inference must never kill the loop
                logger.warning("AI inference failed: %s", exc)
                detections = []
                failed = True
            self._record_inference(time.monotonic() - start, failed)
            self.detections_ready.emit(detections)

    def set_max_fps(self, max_fps: float) -> None:
        max_fps = max(0.0, float(max_fps or 0.0))
        with self._lock:
            self._max_fps = max_fps
            self._min_submit_interval_s = 1.0 / max_fps if max_fps > 0 else 0.0

    @property
    def max_fps(self) -> float:
        with self._lock:
            return self._max_fps

    def stats(self) -> dict[str, float | int | None]:
        now = time.monotonic()
        with self._lock:
            self._trim_busy_samples(now)
            self._trim_drop_samples(now)
            busy_s = sum(duration for _, duration in self._busy_samples)
            util = min(100.0, (busy_s / UTIL_WINDOW_S) * 100.0)
            return {
                "max_fps": self._max_fps,
                "submitted_frames": self._submitted_frames,
                "accepted_frames": self._accepted_frames,
                "dropped_frames": len(self._drop_samples),
                "dropped_frames_total": self._dropped_frames,
                "completed_frames": self._completed_frames,
                "failed_frames": self._failed_frames,
                "last_inference_ms": self._last_inference_ms,
                "worker_util_pct": util,
            }

    def _record_inference(self, duration_s: float, failed: bool) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_inference_ms = duration_s * 1000.0
            self._busy_samples.append((now, duration_s))
            self._trim_busy_samples(now)
            self._completed_frames += 1
            if failed:
                self._failed_frames += 1

    def _trim_busy_samples(self, now: float) -> None:
        cutoff = now - UTIL_WINDOW_S
        self._busy_samples = [(end, duration) for end, duration in self._busy_samples if end >= cutoff]

    def _record_drop_locked(self, now: float) -> None:
        self._dropped_frames += 1
        self._drop_samples.append(now)
        self._trim_drop_samples(now)

    def _trim_drop_samples(self, now: float) -> None:
        cutoff = now - UTIL_WINDOW_S
        self._drop_samples = [dropped_at for dropped_at in self._drop_samples if dropped_at >= cutoff]

    def stop(self) -> None:
        """Signal the loop to exit and wait briefly for it to unwind. Bounded
        wait so app shutdown can't hang on a stuck inference."""
        self._running = False
        self._wake.set()
        if not self.wait(2000):
            logger.warning("AI worker did not stop within 2s")
