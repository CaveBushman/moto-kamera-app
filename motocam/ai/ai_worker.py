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

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from motocam.ai.ai_engine import AiEngine

logger = logging.getLogger("motocam.ai.worker")


class AiWorker(QThread):
    detections_ready = pyqtSignal(object)  # list[Detection]

    def __init__(self, ai_engine: AiEngine, parent=None):
        super().__init__(parent)
        self._engine = ai_engine
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._wake = threading.Event()
        self._running = True

    def submit(self, frame: np.ndarray) -> None:
        """Hand the newest frame to the worker, dropping any previous
        un-processed one -- realtime tracking wants the freshest frame, not
        a backlog of stale ones."""
        with self._lock:
            self._latest = frame
        self._wake.set()

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
            try:
                detections = self._engine.process(frame)
            except Exception as exc:  # noqa: BLE001 -- inference must never kill the loop
                logger.warning("AI inference failed: %s", exc)
                detections = []
            self.detections_ready.emit(detections)

    def stop(self) -> None:
        """Signal the loop to exit and wait briefly for it to unwind. Bounded
        wait so app shutdown can't hang on a stuck inference."""
        self._running = False
        self._wake.set()
        if not self.wait(2000):
            logger.warning("AI worker did not stop within 2s")
