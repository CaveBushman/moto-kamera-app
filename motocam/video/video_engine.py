"""Video Engine (design doc 10.3).

Opens a UVC/V4L2 capture device with OpenCV on a dedicated QThread and
emits frames to both the UI and the AI engine. If no capture device is
available (e.g. running on a dev laptop with no Magewell grabber
attached) it falls back to a synthetic test-pattern generator so the
rest of the app still runs end to end.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

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
        self._timer: QTimer | None = None
        self._synthetic = False
        self._frame_times: list[float] = []
        self._t = 0.0

    def start(self) -> None:
        self._open_capture()
        self._running = True
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        interval_ms = max(1, int(1000 / self._target_fps))
        self._timer.start(interval_ms)

    def stop(self) -> None:
        self._running = False
        if self._timer:
            self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def set_device(self, device: str | int) -> None:
        """Hot-swap the capture device without a full app restart -- same
        idea as PttEngine.set_input_device for the audio side."""
        if device == self._device:
            return
        self._device = device
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._running:
            self._open_capture()

    @property
    def device(self) -> str | int:
        return self._device

    def _open_capture(self) -> None:
        cap = cv2.VideoCapture(self._device)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            self._cap = cap
            self._synthetic = False
            self.status_changed.emit("connected")
            logger.info("UVC capture opened on device %s", self._device)
        else:
            self._cap = None
            self._synthetic = True
            self.status_changed.emit("synthetic")
            logger.warning("No capture device at %s, using synthetic test pattern", self._device)

    def _tick(self) -> None:
        if not self._running:
            return

        frame = None
        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("Frame read failed, attempting reconnect")
                self.status_changed.emit("reconnecting")
                self._cap.release()
                self._cap = None
                self._open_capture()
                return

        if frame is None:
            frame = self._synthetic_frame()

        self._record_fps()
        self.frame_ready.emit(frame)

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
