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
import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger("motocam.video")


class VideoEngine(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    status_changed = pyqtSignal(str)  # "connected" | "reconnecting" | "lost" | "synthetic"

    def __init__(self, device: str | int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        super().__init__()
        self._device = device
        self._width = width
        self._height = height
        self._target_fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._reopen = False
        self._synthetic = False
        self._frame_times: list[float] = []
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
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def set_device(self, device: str | int) -> None:
        """Hot-swap the capture device without a full app restart -- same
        idea as PttEngine.set_input_device for the audio side. Only flips a
        flag; the loop thread does the actual (blocking) reopen so this stays
        instant and can't race with an in-flight read()."""
        if device == self._device:
            return
        with self._lock:
            self._device = device
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

    def _open_capture(self) -> None:
        """Open self._device into self._cap. Called only from the loop
        thread (directly or via a requested reopen)."""
        with self._lock:
            device = self._device
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
            self._synthetic = True
            self.status_changed.emit("synthetic")
            logger.warning("No capture device at %s, using synthetic test pattern", device)

    def _loop(self) -> None:
        self._open_capture()
        interval = 1.0 / max(1, self._target_fps)
        while self._running:
            frame_start = time.monotonic()

            if self._reopen:
                with self._lock:
                    self._reopen = False
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
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
                    self._open_capture()
                    self._sleep_remaining(frame_start, interval)
                    continue

            if frame is None:
                frame = self._synthetic_frame()

            self._record_fps()
            if self._running:
                self.frame_ready.emit(frame)
            self._sleep_remaining(frame_start, interval)

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
        self._frame_times = [t for t in self._frame_times if t >= cutoff]
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                fps = (len(self._frame_times) - 1) / span
                self.fps_updated.emit(fps)
