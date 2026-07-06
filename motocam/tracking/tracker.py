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

from motocam.ai.ai_engine import Detection
from motocam.core.protocol import TargetState
from motocam.tracking.byte_tracker import ByteTracker

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

    Detection-driven mode (the peloton path): when the AI engine is
    producing detections, update_detections() drives a ByteTracker that
    holds a stable id per rider. Tapping (or FULL-AI auto-acquire) locks a
    track *id*, and the target box then comes from that track -- coasting
    through occlusion via the Kalman filter and surviving a rider crossing
    in front, which CSRT cannot. When there are no detections (no HEF /
    AI NULL) or the tap hits no track, it falls back to CSRT appearance
    tracking. The ByteTracker is only ever touched from the UI thread
    (update_detections / taps / control tick), so it needs no locking;
    while a track id is locked the CSRT worker is idle and does not touch
    the published target.
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
        self._stats_lock = threading.Lock()
        self._submitted_frames = 0
        self._dropped_frames = 0
        self._completed_updates = 0
        self._failed_updates = 0
        self._last_update_ms: float | None = None
        self._busy_started_at: float | None = None
        self._last_slow_log_at = 0.0
        # detection-driven peloton tracking (UI-thread only, no lock)
        self._byte = ByteTracker()
        self._locked_id: int | None = None

    @property
    def state(self) -> TargetState:
        return self._state

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._bbox

    # -- detection-driven peloton tracking (UI thread) ----------------------
    def update_detections(self, detections: list[Detection]) -> None:
        """Advance the ByteTracker with the latest AI detections and, if a
        rider id is locked, refresh the published target from that track
        (coasting through occlusion). Cheap; called on the UI thread when
        the AI worker delivers detections."""
        self._byte.update(detections)
        if self._locked_id is None:
            return
        track = self._byte.get(self._locked_id)
        if track is None:
            # the locked rider left tracking (dropped after max_age)
            self._locked_id = None
            self._bbox = None
            self._state = TargetState.MANUAL_REQUIRED
            logger.info("Locked track lost, manual re-selection required")
        else:
            self._publish_from_track(track)

    def _publish_from_track(self, track) -> None:
        bx, by, bw, bh = track.box
        self._bbox = (int(bx), int(by), int(bw), int(bh))
        self._state = TargetState.LOCKED if track.time_since_update == 0 else TargetState.WEAK
        self._last_good_time = time.monotonic()

    def _lock_track(self, track_id: int) -> None:
        """Lock onto a ByteTracker id; the CSRT worker goes idle so it can't
        overwrite the detection-driven target."""
        self._locked_id = track_id
        with self._lock:
            self._tracker = None
        track = self._byte.get(track_id)
        if track is not None:
            self._publish_from_track(track)
        logger.info("Locked onto track id %d", track_id)

    def select_at(self, frame: np.ndarray, x: int, y: int, box_size: int = 120) -> None:
        """Operator tap-to-select. Prefer locking the AI track under the tap
        (stable id, occlusion-robust); fall back to a fresh CSRT box when no
        detection is there (AI NULL / tapped between riders)."""
        track_id = self._byte.track_at(x, y)
        if track_id is not None:
            self._lock_track(track_id)
            return
        self._locked_id = None
        h, w = frame.shape[:2]
        half = box_size // 2
        bx = max(0, min(w - box_size, x - half))
        by = max(0, min(h - box_size, y - half))
        bw = min(box_size, w - bx)
        bh = min(box_size, h - by)
        self._start(frame, (bx, by, bw, bh))
        logger.info("Target selected at (%d, %d) (CSRT fallback)", x, y)

    def select_box(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
        """Auto-acquire from an AI detection box (design doc 8.3, FULL AI).
        Prefer locking the ByteTracker id at the detection's centre (stable,
        occlusion-robust); fall back to seeding CSRT with the box when no
        track is there yet."""
        bx, by, bw, bh = box
        track_id = self._byte.track_at(bx + bw / 2.0, by + bh / 2.0)
        if track_id is not None:
            self._lock_track(track_id)
            return
        self._locked_id = None
        fh, fw = frame.shape[:2]
        bx = max(0, min(fw - 1, bx))
        by = max(0, min(fh - 1, by))
        bw = max(1, min(fw - bx, bw))
        bh = max(1, min(fh - by, bh))
        self._start(frame, (bx, by, bw, bh))
        logger.info("Target auto-acquired from detection at (%d, %d, %d, %d) (CSRT)", bx, by, bw, bh)

    def _start(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
        with self._lock:
            self._tracker = self._new_tracker()
            self._tracker.init(frame, box)
            self._bbox = box
            self._state = TargetState.LOCKED
            self._last_good_time = time.monotonic()

    def clear(self) -> None:
        self._locked_id = None
        with self._lock:
            self._tracker = None
            self._bbox = None
            self._state = TargetState.IDLE
            self._last_good_time = None

    def update(self, frame: np.ndarray) -> None:
        # In detection-driven (byte-locked) mode the target comes from the
        # ByteTracker via update_detections(); CSRT stays idle so it can't
        # fight over the published target.
        if self._locked_id is not None:
            return
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
        if not self.needs_frame_updates:
            return
        with self._frame_lock:
            had_pending = self._frame_latest is not None
            self._frame_latest = frame
        with self._stats_lock:
            self._submitted_frames += 1
            if had_pending:
                self._dropped_frames += 1
        self._wake.set()

    @property
    def needs_frame_updates(self) -> bool:
        if self._locked_id is not None:
            return False
        with self._lock:
            return self._tracker is not None

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
                started = time.monotonic()
                with self._stats_lock:
                    self._busy_started_at = started
                self.update(frame)
                elapsed_ms = (time.monotonic() - started) * 1000.0
                with self._stats_lock:
                    self._busy_started_at = None
                    self._last_update_ms = elapsed_ms
                    self._completed_updates += 1
                self._log_slow_update(elapsed_ms)
            except Exception as exc:  # noqa: BLE001 -- a bad frame must not kill tracking
                with self._stats_lock:
                    self._busy_started_at = None
                    self._failed_updates += 1
                logger.warning("tracker update failed: %s", exc)

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        thread = self._worker_thread
        self._worker_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def stats(self) -> dict[str, float | int | str | None]:
        now = time.monotonic()
        with self._stats_lock:
            busy_ms = ((now - self._busy_started_at) * 1000.0) if self._busy_started_at is not None else None
            mode = "byte" if self._locked_id is not None else ("csrt" if self._tracker is not None else "idle")
            return {
                "mode": mode,
                "submitted_frames": self._submitted_frames,
                "dropped_frames": self._dropped_frames,
                "completed_updates": self._completed_updates,
                "failed_updates": self._failed_updates,
                "last_update_ms": self._last_update_ms,
                "current_busy_ms": busy_ms,
            }

    def _log_slow_update(self, elapsed_ms: float) -> None:
        if elapsed_ms < 80.0:
            return
        now = time.monotonic()
        if now - self._last_slow_log_at < 2.0:
            return
        self._last_slow_log_at = now
        logger.warning(
            "CSRT tracker update slow: %.0f ms state=%s bbox=%s",
            elapsed_ms,
            self._state.value,
            self._bbox,
        )

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
