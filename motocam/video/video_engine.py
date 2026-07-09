"""Video Engine (design doc 10.3).

Opens a UVC/V4L2 capture device with OpenCV on a dedicated background
thread and emits frames to both the UI and the AI engine. If no capture
device is available (e.g. running on a dev laptop with no Magewell
grabber attached) it falls back to a synthetic test-pattern generator so
the rest of the app still runs end to end.

The read loop MUST stay off the Qt UI thread: cv2.VideoCapture.read() is
a blocking call and a hiccuping USB grabber can stall it for a long time,
which -- if it ran on the UI thread -- would freeze the whole interface.
Frames cross back to the UI thread through the frame_ready signal (a
queued connection). The capture object is owned exclusively by the loop
thread; set_device only *requests* a reopen (via a flag) so it never
touches the device from the UI thread while a read is in flight.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger("motocam.video")

FPS_EMIT_INTERVAL_S = 0.5
# How often the loop retries opening the capture device while it has none
# (never found one, or a reopen after a failed read also came up empty).
# Retrying every frame interval would hammer the driver and can stall the
# synthetic-frame cadence noticeably (cv2.VideoCapture(device) is a
# blocking call); this throttles it to a background heartbeat instead.
REOPEN_RETRY_INTERVAL_S = 3.0

if sys.platform == "darwin":
    # OpenCV's AVFoundation backend tries to request camera permission by
    # spinning the macOS main run loop. Our capture opens in a worker thread,
    # so that permission path can hang or print "can not spin main run loop
    # from other thread". We prefer an immediate synthetic fallback on dev
    # Macs; real capture permissions should be granted by launching from a
    # normal app shell before race-day testing.
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")


class VideoEngine(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    status_changed = pyqtSignal(str)  # "connected" | "reconnecting" | "lost" | "synthetic"

    def __init__(
        self,
        device: str | int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        allow_macos_capture: bool = True,
    ):
        super().__init__()
        self._device = device
        self._width = width
        self._height = height
        self._target_fps = fps
        self._allow_macos_capture = allow_macos_capture
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._reopen = False
        self._synthetic = False
        self._frame_in_flight = False
        self._dropped_frames = 0
        self._frame_times: deque[float] = deque()
        self._last_fps_emit_at: float | None = None
        self._last_reopen_attempt_at = 0.0
        self._t = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Capture is opened inside the loop thread (cv2.VideoCapture can
        # block for seconds on a real grabber) so start() never stalls the
        # caller / UI thread.
        self._thread = threading.Thread(target=self._loop, name="video-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            # Loop thread is still blocked inside a native cv2 call (e.g. a
            # stalled UVC driver) after the join timeout -- releasing the
            # capture out from under it here would race a concurrent
            # cap.read()/cap.release() on the same handle, which OpenCV/V4L2
            # gives no safety guarantee for (can crash or hang the process
            # on shutdown). Leaving the handle for the OS to reclaim at
            # process exit is a far smaller risk than that.
            logger.warning("Video capture thread did not stop within timeout; leaving capture handle for process exit")
            return
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def set_device(self, device: str | int) -> None:
        """Hot-swap the capture device without a full app restart -- same
        idea as PttEngine.set_input_device for the audio side. Only flips a
        flag; the loop thread does the actual (blocking) reopen so this stays
        instant and can't race with an in-flight read().

        An explicit device choice also overrides the macOS synthetic-only
        default (allow_macos_capture=False, set in main.py for dev laptops):
        that default exists so the app doesn't hang in AVFoundation's
        permission machinery *unprompted* at startup -- but an operator
        deliberately picking a camera in Settings is precisely the moment
        capture is wanted, and silently staying synthetic here made the
        Capture-device selector appear completely non-functional on a Mac."""
        if device == self._device and self._allow_macos_capture:
            return
        with self._lock:
            self._device = device
            self._allow_macos_capture = True
            self._reopen = True
        if not self._running:
            # Not streaming yet: just record the new device for next start().
            self._reopen = False

    @property
    def device(self) -> str | int:
        return self._device

    @property
    def source(self) -> str:
        return "synthetic" if self._synthetic else "real"

    def mark_frame_delivered(self) -> None:
        """Called by the UI slot after it has accepted a frame.

        PyQt queued signals copy only references, but the event queue can
        still accumulate many full-resolution numpy frames when capture is
        faster than rendering/tracking. Keep at most one frame in flight so
        live video drops stale frames instead of growing memory until the
        app appears frozen.
        """
        with self._lock:
            self._frame_in_flight = False

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames

    def _open_capture(self) -> None:
        """Open self._device into self._cap. Called only from the loop
        thread (directly or via a requested reopen)."""
        with self._lock:
            device = self._device
        if sys.platform == "darwin" and not self._allow_macos_capture:
            with self._lock:
                self._cap = None
            self._synthetic = True
            self.status_changed.emit("synthetic")
            logger.warning(
                "macOS UVC capture disabled by config, using synthetic test pattern "
                "(set video.allow_macos_capture=true to use AVFoundation)"
            )
            return

        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            with self._lock:
                self._cap = cap
            self._synthetic = False
            self.status_changed.emit("connected")
            logger.info("UVC capture opened on device %s", device)
        else:
            cap.release()
            with self._lock:
                self._cap = None
            # This now runs every REOPEN_RETRY_INTERVAL_S while the device
            # is absent -- log the fallback only on the transition into
            # synthetic, not on every retry heartbeat.
            if not self._synthetic:
                logger.warning("No capture device at %s, using synthetic test pattern", device)
            self._synthetic = True
            self.status_changed.emit("synthetic")

    def _loop(self) -> None:
        # Same contract as main_window._on_frame on the consuming side: one
        # bad iteration (a cv2/V4L2 exception during a reconnect handshake
        # after a device power blip, say) must cost one frame, never the
        # thread -- an unguarded exception here would kill this daemon
        # thread silently, freezing the feed with the status chip stuck on
        # its last state and nothing ever restarting capture.
        interval = 1.0 / max(1, self._target_fps)
        while self._running:
            try:
                self._loop_once(interval)
            except Exception as exc:  # noqa: BLE001 -- capture must degrade per-frame, not die
                logger.warning("Video capture iteration failed (%s), continuing", exc)
                time.sleep(interval)

    def _loop_once(self, interval: float) -> None:
        frame_start = time.monotonic()

        if self._reopen:
            with self._lock:
                self._reopen = False
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None

        with self._lock:
            cap = self._cap
        if cap is None and self._should_retry_open(frame_start):
            self._open_capture()
            with self._lock:
                cap = self._cap

        frame = None
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Frame read failed, attempting reconnect")
                self.status_changed.emit("reconnecting")
                with self._lock:
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                # Next iterations keep retrying via _should_retry_open --
                # the old code retried exactly once, right now, which on a
                # jostled USB grabber usually fires before the OS has
                # re-enumerated the device, permanently stranding the app
                # on the synthetic pattern even after the grabber came back.
                self._sleep_remaining(frame_start, interval)
                return

        if frame is None:
            frame = self._synthetic_frame()

        self._record_fps()
        if self._running:
            with self._lock:
                if self._frame_in_flight:
                    self._dropped_frames += 1
                    should_emit = False
                else:
                    self._frame_in_flight = True
                    should_emit = True
            if should_emit:
                self.frame_ready.emit(frame)
        self._sleep_remaining(frame_start, interval)

    def _should_retry_open(self, now: float) -> bool:
        """Throttled periodic reopen while no capture device is held -- so
        a grabber that appears later (booted before it enumerated, or
        replugged mid-ride) gets picked up automatically instead of only on
        a manual Settings change. cv2.VideoCapture() blocks, so this must
        not run every frame interval."""
        if now - self._last_reopen_attempt_at < REOPEN_RETRY_INTERVAL_S:
            return False
        self._last_reopen_attempt_at = now
        return True

    @staticmethod
    def _sleep_remaining(frame_start: float, interval: float) -> None:
        elapsed = time.monotonic() - frame_start
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _synthetic_frame(self) -> np.ndarray:
        self._t += 1 / self._target_fps
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        cx = int(self._width / 2 + 300 * np.sin(self._t))
        cy = int(self._height / 2 + 150 * np.cos(self._t * 0.7))
        cv2.circle(frame, (cx, cy), 40, (60, 180, 255), -1)
        cv2.putText(
            frame, "NO CAPTURE DEVICE - SYNTHETIC PREVIEW", (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
        )
        cv2.putText(
            frame, time.strftime("%H:%M:%S"), (40, self._height - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2,
        )
        return frame

    def _record_fps(self) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        cutoff = now - 2.0
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()
        if len(self._frame_times) >= 2:
            if self._last_fps_emit_at is not None and now - self._last_fps_emit_at < FPS_EMIT_INTERVAL_S:
                return
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                fps = (len(self._frame_times) - 1) / span
                self._last_fps_emit_at = now
                self.fps_updated.emit(fps)
