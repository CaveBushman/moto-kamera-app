"""Bandwidth policy for the low-fps JPEG preview relay to control room."""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger("motocam.video.preview_relay")


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


class PreviewRelayEncoder(QThread):
    """Encode low-fps control-room preview JPEGs off the Qt UI thread.

    cv2.imencode() on a scaled 1080p frame is small on a laptop but still
    enough to add touch/joystick jitter on a Raspberry Pi. The worker keeps
    only the newest frame, matching LinkClient's own backpressure policy.
    """

    encoded = pyqtSignal(bytes, int, int, int)  # jpeg, width, height, byte_count

    def __init__(self, *, max_width: int | str | None, jpeg_quality: int, parent=None):
        super().__init__(parent)
        self._max_width = max_width
        self._jpeg_quality = preview_jpeg_quality(jpeg_quality)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = True
        self._latest: np.ndarray | None = None
        self._submitted_frames = 0
        self._encoded_frames = 0
        self._dropped_frames = 0
        self._last_encode_ms: float | None = None

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._latest is not None:
                self._dropped_frames += 1
            self._latest = frame
            self._submitted_frames += 1
        self._wake.set()

    def configure(self, *, max_width: int | str | None, jpeg_quality: int) -> None:
        with self._lock:
            self._max_width = max_width
            self._jpeg_quality = preview_jpeg_quality(jpeg_quality)

    def run(self) -> None:  # noqa: D401 -- QThread entry point
        while self._running:
            self._wake.wait()
            if not self._running:
                break
            self._wake.clear()
            frame, max_width, quality = self._take()
            if frame is None:
                continue
            start = time.monotonic()
            try:
                relay_frame = self._scaled_frame(frame, max_width)
                relay_h, relay_w = relay_frame.shape[:2]
                ok, buf = cv2.imencode(".jpg", relay_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    continue
                byte_count = int(buf.nbytes)
                self._record_encode(time.monotonic() - start)
                self.encoded.emit(buf.tobytes(), relay_w, relay_h, byte_count)
            except Exception as exc:  # noqa: BLE001
                logger.warning("preview relay encode failed: %s", exc)

    def _take(self) -> tuple[np.ndarray | None, int | str | None, int]:
        with self._lock:
            frame = self._latest
            self._latest = None
            return frame, self._max_width, self._jpeg_quality

    @staticmethod
    def _scaled_frame(frame: np.ndarray, max_width: int | str | None) -> np.ndarray:
        h, w = frame.shape[:2]
        target_w, target_h = scaled_preview_size(w, h, max_width)
        if (target_w, target_h) == (w, h):
            return frame
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _record_encode(self, duration_s: float) -> None:
        with self._lock:
            self._encoded_frames += 1
            self._last_encode_ms = duration_s * 1000.0

    def stats(self) -> dict[str, float | int | None]:
        with self._lock:
            return {
                "submitted_frames": self._submitted_frames,
                "encoded_frames": self._encoded_frames,
                "dropped_frames": self._dropped_frames,
                "last_encode_ms": self._last_encode_ms,
            }

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        if not self.wait(1000):
            logger.warning("preview relay encoder did not stop within 1s")
