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
from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from motocam.ai.ai_engine import AiEngine, Detector

logger = logging.getLogger("motocam.ai.worker")
UTIL_WINDOW_S = 2.0

try:
    import cv2
except Exception:  # noqa: BLE001 -- pure numpy fallback keeps tests importable without OpenCV
    cv2 = None


def _select_resize_interpolation(src_w: int, src_h: int, dst_w: int, dst_h: int):
    if cv2 is None:
        return None
    if dst_w >= src_w and dst_h >= src_h:
        return cv2.INTER_LINEAR
    return cv2.INTER_AREA


def _resize_for_inference(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Return a smaller inference frame plus the scale back to source pixels.

    Hailo models letterbox to their own input size internally. Feeding them
    a full 1080p frame first just costs CPU/memory bandwidth on the Pi.
    Downscaling here keeps detector boxes equivalent after remapping while
    making the UI-to-worker handoff and detector preprocessing cheaper.
    """
    if max_width <= 0:
        return frame, 1.0
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / float(width)
    target_size = (max_width, max(1, int(round(height * scale))))
    if cv2 is not None:
        interpolation = _select_resize_interpolation(width, height, target_size[0], target_size[1])
        resized = cv2.resize(frame, target_size, interpolation=interpolation)
    else:
        ys = np.linspace(0, height - 1, target_size[1]).astype(int)
        xs = np.linspace(0, width - 1, target_size[0]).astype(int)
        resized = frame[ys][:, xs]
    return resized, 1.0 / scale


def _scale_detections(detections: list, scale_to_source: float, frame_shape: tuple[int, ...]) -> list:
    if scale_to_source == 1.0:
        return detections
    height, width = frame_shape[:2]
    scaled = []
    for det in detections:
        x = max(0, min(width - 1, int(round(det.x * scale_to_source))))
        y = max(0, min(height - 1, int(round(det.y * scale_to_source))))
        w = max(1, min(width - x, int(round(det.w * scale_to_source))))
        h = max(1, min(height - y, int(round(det.h * scale_to_source))))
        scaled.append(type(det)(x=x, y=y, w=w, h=h, confidence=det.confidence, class_name=det.class_name))
    return scaled


class AiWorker(QThread):
    """Background inference loop (see module docstring); owns frame
    submission, drop/backpressure bookkeeping, and adaptive rate limiting
    so a slow detector self-throttles instead of saturating the CPU."""

    detections_ready = pyqtSignal(object)  # list[Detection]

    def __init__(
        self,
        ai_engine: AiEngine,
        max_fps: float = 0.0,
        max_input_width: int = 960,
        performance_budget_pct: float = 35.0,
        detector_factory: Callable[[], Detector] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._engine = ai_engine
        self._detector_factory = detector_factory
        self._detector_factory_done = detector_factory is None
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._wake = threading.Event()
        self._running = True
        self._max_fps = 0.0
        self._max_input_width = max(0, int(max_input_width or 0))
        self._min_submit_interval_s = 0.0
        self._last_accept_at = 0.0
        self._submitted_frames = 0
        self._accepted_frames = 0
        self._dropped_frames = 0
        self._drop_samples: list[float] = []
        self._completed_frames = 0
        self._failed_frames = 0
        self._last_inference_ms: float | None = None
        self._busy_started_at: float | None = None
        self._performance_budget_pct = max(5.0, min(100.0, float(performance_budget_pct or 35.0)))
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
            min_interval_s = self._effective_min_submit_interval_locked()
            if min_interval_s > 0 and now - self._last_accept_at < min_interval_s:
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
        self._ensure_detector_ready()
        while self._running:
            self._wake.wait()
            if not self._running:
                break
            self._wake.clear()
            item = self._take()
            if item is None:
                continue
            frame = item
            start = time.monotonic()
            failed = False
            with self._lock:
                self._busy_started_at = start
            try:
                inference_frame, scale_to_source = _resize_for_inference(frame, self._max_input_width)
                detections = self._engine.process(inference_frame)
                source_shape = frame.shape
                detections = _scale_detections(detections, scale_to_source, source_shape)
            except Exception as exc:  # noqa: BLE001 -- inference must never kill the loop
                logger.warning("AI inference failed: %s", exc)
                detections = []
                failed = True
            self._record_inference(time.monotonic() - start, failed)
            self.detections_ready.emit(detections)

    def _ensure_detector_ready(self) -> None:
        if self._detector_factory_done or self._detector_factory is None:
            return
        try:
            start = time.monotonic()
            detector = self._detector_factory()
            self._engine.detector = detector
            logger.info(
                "AI detector initialised on worker thread in %.0f ms (source=%s)",
                (time.monotonic() - start) * 1000.0,
                self._engine.source,
            )
        except Exception as exc:  # noqa: BLE001 -- AI init must never prevent manual operation
            logger.warning("AI detector initialisation failed on worker thread: %s", exc)
        finally:
            self._detector_factory_done = True

    def set_max_fps(self, max_fps: float) -> None:
        max_fps = max(0.0, float(max_fps or 0.0))
        with self._lock:
            self._max_fps = max_fps
            self._min_submit_interval_s = 1.0 / max_fps if max_fps > 0 else 0.0

    @property
    def max_fps(self) -> float:
        with self._lock:
            return self._max_fps

    def set_max_input_width(self, max_input_width: int) -> None:
        with self._lock:
            self._max_input_width = max(0, int(max_input_width or 0))

    @property
    def max_input_width(self) -> int:
        with self._lock:
            return self._max_input_width

    def set_performance_budget_pct(self, budget_pct: float) -> None:
        with self._lock:
            self._performance_budget_pct = max(5.0, min(100.0, float(budget_pct or 35.0)))

    @property
    def performance_budget_pct(self) -> float:
        with self._lock:
            return self._performance_budget_pct

    def stats(self) -> dict[str, float | int | None]:
        now = time.monotonic()
        with self._lock:
            self._trim_busy_samples(now)
            self._trim_drop_samples(now)
            busy_s = sum(duration for _, duration in self._busy_samples)
            util = min(100.0, (busy_s / UTIL_WINDOW_S) * 100.0)
            effective_min_interval = self._effective_min_submit_interval_locked()
            current_busy_ms = ((now - self._busy_started_at) * 1000.0) if self._busy_started_at is not None else None
            return {
                "max_fps": self._max_fps,
                "effective_max_fps": (1.0 / effective_min_interval) if effective_min_interval > 0 else 0.0,
                "submitted_frames": self._submitted_frames,
                "accepted_frames": self._accepted_frames,
                "dropped_frames": len(self._drop_samples),
                "dropped_frames_total": self._dropped_frames,
                "completed_frames": self._completed_frames,
                "failed_frames": self._failed_frames,
                "last_inference_ms": self._last_inference_ms,
                "current_busy_ms": current_busy_ms,
                "worker_util_pct": util,
                "max_input_width": self._max_input_width,
                "performance_budget_pct": self._performance_budget_pct,
            }

    def _effective_min_submit_interval_locked(self) -> float:
        """Static max_fps plus an adaptive CPU budget.

        Hailo accelerates the neural net, but the Pi CPU still pays for
        frame handoff, resize/letterbox and post-processing. If a real
        inference takes e.g. 80 ms and the budget is 35%, accepting frames
        faster than about 4.3 fps just builds heat and UI jitter. Latest
        frame wins, so dropping over-budget frames is the right realtime
        behaviour.
        """
        min_interval = self._min_submit_interval_s
        if self._last_inference_ms is None or self._last_inference_ms <= 0:
            return min_interval
        budget_ratio = self._performance_budget_pct / 100.0
        adaptive_interval = (self._last_inference_ms / 1000.0) / budget_ratio
        return max(min_interval, adaptive_interval)

    def _record_inference(self, duration_s: float, failed: bool) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_inference_ms = duration_s * 1000.0
            self._busy_samples.append((now, duration_s))
            self._trim_busy_samples(now)
            self._completed_frames += 1
            self._busy_started_at = None
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
