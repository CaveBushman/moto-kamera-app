"""Tracking Engine (design doc 8.2-8.4, 10.5).

Owns the active target: operator taps a rider in the live preview, the
engine starts an OpenCV correlation tracker (CSRT) on that bounding box
and reports its center every frame so the PID loop can compute pan/tilt
error. A real deployment feeds YOLO detections from ai/ai_engine.py into
re-acquire the tracker after a short loss (ByteTrack/Kalman per section
8.3) -- this MVP tracker instead relies on CSRT's own short-term
robustness and falls back to "lost" if confidence drops.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from motocam.core.protocol import TargetState

logger = logging.getLogger("motocam.tracking")

WEAK_AFTER_S = 0.8
LOST_AFTER_S = 2.5


class TrackingEngine:
    """CSRT tracker with the heavy per-frame update() driven off the UI
    thread.

    update() on a 1080p frame is tens of milliseconds -- far too much to
    run on the Qt frame callback ~30x/s. Instead the UI thread calls
    submit() (cheap: stash the latest frame, drop any older one) and an
    internal worker thread runs update(). The cv2 tracker object is NOT
    thread-safe, so all access to it (init/update/clear) is serialized by
    `_lock`; the *published* result (`_bbox`, `_state`) is read lock-free
    by the UI/PID via atomic attribute reads. `_lock` is deliberately NOT
    the same lock submit() uses -- submit() must never block behind a slow
    update(), or we'd just move the stall back onto the UI thread.

    update() also still works when called directly and synchronously (as
    the unit tests do); the worker is optional and started via start().
    """

    def __init__(self):
        self._tracker: cv2.Tracker | None = None
        self._bbox: tuple[int, int, int, int] | None = None
        self._state: TargetState = TargetState.IDLE
        self._last_good_time: float | None = None
        self._lock = threading.Lock()  # guards the cv2 tracker + published state
        self._frame_lock = threading.Lock()  # guards only the latest-frame handoff
        self._frame_latest: np.ndarray | None = None
        self._wake = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._running = False

    @property
    def state(self) -> TargetState:
        return self._state

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._bbox

    def select_at(self, frame: np.ndarray, x: int, y: int, box_size: int = 120) -> None:
        """Operator tap-to-select: start a fresh tracker centered on (x, y)."""
        h, w = frame.shape[:2]
        half = box_size // 2
        bx = max(0, min(w - box_size, x - half))
        by = max(0, min(h - box_size, y - half))
        bw = min(box_size, w - bx)
        bh = min(box_size, h - by)
        self._start(frame, (bx, by, bw, bh))
        logger.info("Target selected at (%d, %d)", x, y)

    def select_box(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
        """Auto-acquire from an AI detection box (design doc 8.3): seed the
        CSRT tracker with the detector's actual bounding box, which is a
        far better initialisation than a fixed square around a point."""
        fh, fw = frame.shape[:2]
        bx, by, bw, bh = box
        bx = max(0, min(fw - 1, bx))
        by = max(0, min(fh - 1, by))
        bw = max(1, min(fw - bx, bw))
        bh = max(1, min(fh - by, bh))
        self._start(frame, (bx, by, bw, bh))
        logger.info("Target auto-acquired from detection at (%d, %d, %d, %d)", bx, by, bw, bh)

    def _start(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
        with self._lock:
            self._tracker = self._new_tracker()
            self._tracker.init(frame, box)
            self._bbox = box
            self._state = TargetState.LOCKED
            self._last_good_time = time.monotonic()

    def clear(self) -> None:
        with self._lock:
            self._tracker = None
            self._bbox = None
            self._state = TargetState.IDLE
            self._last_good_time = None

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._tracker is None:
                return

            ok, box = self._tracker.update(frame)
            now = time.monotonic()
            if ok:
                self._bbox = tuple(int(v) for v in box)
                self._last_good_time = now
                self._state = TargetState.LOCKED
                return

            assert self._last_good_time is not None
            age = now - self._last_good_time
            if age < WEAK_AFTER_S:
                self._state = TargetState.LOCKED
            elif age < LOST_AFTER_S:
                self._state = TargetState.WEAK
            else:
                self._state = TargetState.MANUAL_REQUIRED
                self._bbox = None
                self._tracker = None
                logger.info("Target lost, manual re-selection required")

    # -- off-thread driver ---------------------------------------------------
    def submit(self, frame: np.ndarray) -> None:
        """Hand the newest frame to the worker (drops any older un-processed
        one). Cheap and non-blocking -- safe to call from the UI frame
        callback every frame."""
        with self._frame_lock:
            self._frame_latest = frame
        self._wake.set()

    def _take(self) -> np.ndarray | None:
        with self._frame_lock:
            frame = self._frame_latest
            self._frame_latest = None
        return frame

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, name="tracker", daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while self._running:
            self._wake.wait()
            if not self._running:
                break
            self._wake.clear()
            frame = self._take()
            if frame is None:
                continue
            try:
                self.update(frame)
            except Exception as exc:  # noqa: BLE001 -- a bad frame must not kill tracking
                logger.warning("tracker update failed: %s", exc)

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        thread = self._worker_thread
        self._worker_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def error_from_center(self, frame_shape: tuple[int, int]) -> tuple[float, float] | None:
        """Pixel error of target center vs. frame center (desired composition point)."""
        if self._bbox is None:
            return None
        h, w = frame_shape[:2]
        bx, by, bw, bh = self._bbox
        target_cx = bx + bw / 2
        target_cy = by + bh / 2
        return target_cx - w / 2, target_cy - h / 2

    @staticmethod
    def _new_tracker() -> cv2.Tracker:
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        return cv2.TrackerKCF_create()
