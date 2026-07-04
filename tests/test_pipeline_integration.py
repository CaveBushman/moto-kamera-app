"""End-to-end integration of the threaded frame pipeline (stability).

Wires the real VideoEngine -> (TrackingEngine worker + AiWorker) fan-out
the way MainWindow does, drives it with the synthetic capture fallback,
and asserts that: frames flow, both workers actually run off-thread, and
-- the part most likely to regress -- everything shuts down cleanly with
no leftover reader/worker threads. No Qt widgets or async loop needed."""
from __future__ import annotations

import threading
import time

import numpy as np

from motocam.ai.ai_engine import AiEngine, Detection
from motocam.ai.ai_worker import AiWorker
from motocam.core.protocol import TargetState
from motocam.tracking.tracker import TrackingEngine
from motocam.video.video_engine import VideoEngine


class _CountingDetector:
    source = "fake"

    def __init__(self):
        self.calls = 0

    def infer(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [Detection(x=1, y=1, w=10, h=10, confidence=0.9, class_name="cyclist")]

    @property
    def fps(self) -> float:
        return 0.0


def _pump(qapp, predicate, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    qapp.processEvents()


def test_frame_fanout_runs_off_thread_and_shuts_down_clean(qapp):
    threads_before = threading.active_count()

    video = VideoEngine(device="/dev/motocam-none", width=160, height=120, fps=60)
    tracker = TrackingEngine()
    detector = _CountingDetector()
    engine = AiEngine(detector=detector)
    engine.enabled = True
    ai = AiWorker(engine)

    seen = {"frames": 0}

    def on_frame(frame: np.ndarray) -> None:
        seen["frames"] += 1
        if seen["frames"] == 1:
            tracker.select_at(frame, 80, 60)  # lock a target on the first frame
        tracker.submit(frame)
        ai.submit(frame)

    video.frame_ready.connect(on_frame)

    tracker.start()
    ai.start()
    video.start()
    try:
        _pump(
            qapp,
            lambda: seen["frames"] >= 5 and detector.calls >= 1 and tracker.bbox is not None,
            timeout_s=3.0,
        )
    finally:
        video.stop()
        ai.stop()
        tracker.stop()
        qapp.processEvents()

    # frames flowed and both workers actually did work off the calling thread
    assert seen["frames"] >= 5
    assert detector.calls >= 1
    assert tracker.state == TargetState.LOCKED
    assert tracker.bbox is not None

    # the part that matters for a long-running kiosk: no thread leak
    _pump(qapp, lambda: threading.active_count() <= threads_before, timeout_s=2.0)
    assert threading.active_count() <= threads_before


import pytest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
