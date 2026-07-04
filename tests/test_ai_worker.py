"""Tests for the off-UI-thread AI worker (stability): the drop-to-latest
semantics that stop a slow detector building a backlog, that disabled
inference produces nothing, and that a running worker actually delivers
detections and never lets an inference exception escape the loop."""
from __future__ import annotations

import numpy as np
import pytest

from motocam.ai.ai_engine import AiEngine, Detection
from motocam.ai.ai_worker import AiWorker


class _CountingDetector:
    """Records how many frames it was asked to infer and returns one
    detection tagged with the frame's marker value."""

    source = "fake"

    def __init__(self):
        self.calls = 0

    def infer(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        marker = int(frame[0, 0, 0])
        return [Detection(x=marker, y=0, w=10, h=10, confidence=0.9, class_name="cyclist")]

    @property
    def fps(self) -> float:
        return 0.0


class _BoomDetector:
    source = "fake"

    def infer(self, frame: np.ndarray) -> list[Detection]:
        raise RuntimeError("accelerator on fire")

    @property
    def fps(self) -> float:
        return 0.0


def _frame(marker: int) -> np.ndarray:
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    f[:] = marker
    return f


def test_take_drops_all_but_the_latest_submitted_frame():
    engine = AiEngine(detector=_CountingDetector())
    worker = AiWorker(engine)
    worker.submit(_frame(1))
    worker.submit(_frame(2))
    worker.submit(_frame(3))
    got = worker._take()
    assert got is not None and int(got[0, 0, 0]) == 3  # only the newest survives
    assert worker._take() is None  # and it is consumed exactly once


def test_max_fps_limiter_drops_over_budget_frames():
    engine = AiEngine(detector=_CountingDetector())
    worker = AiWorker(engine, max_fps=1.0)

    assert worker.submit(_frame(1)) is True
    assert worker.submit(_frame(2)) is False

    stats = worker.stats()
    assert stats["accepted_frames"] == 1
    assert stats["dropped_frames"] == 1
    assert worker.max_fps == 1.0


def test_disabled_engine_yields_no_detections():
    engine = AiEngine(detector=_CountingDetector())
    engine.enabled = False
    assert engine.process(_frame(5)) == []


def test_running_worker_emits_detections(qapp):
    engine = AiEngine(detector=_CountingDetector())
    engine.enabled = True
    worker = AiWorker(engine)
    received: list = []
    worker.detections_ready.connect(received.append)
    worker.start()
    try:
        worker.submit(_frame(7))
        _spin_until(qapp, lambda: len(received) >= 1, timeout_s=2.0)
    finally:
        worker.stop()
    assert received and received[-1][0].x == 7
    stats = worker.stats()
    assert stats["completed_frames"] >= 1
    assert stats["last_inference_ms"] is not None
    assert stats["worker_util_pct"] is not None


def test_inference_exception_is_swallowed(qapp):
    engine = AiEngine(detector=_BoomDetector())
    engine.enabled = True
    worker = AiWorker(engine)
    received: list = []
    worker.detections_ready.connect(received.append)
    worker.start()
    try:
        worker.submit(_frame(1))
        _spin_until(qapp, lambda: len(received) >= 1, timeout_s=2.0)
    finally:
        worker.stop()
    assert received == [[]]  # a failed inference emits an empty list, loop survives


# -- helpers --------------------------------------------------------------
def _spin_until(qapp, predicate, timeout_s: float) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    qapp.processEvents()


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
