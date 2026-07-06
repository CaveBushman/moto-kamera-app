"""Tests for the off-UI-thread AI worker (stability): the drop-to-latest
semantics that stop a slow detector building a backlog, that disabled
inference produces nothing, and that a running worker actually delivers
detections and never lets an inference exception escape the loop."""
from __future__ import annotations

import numpy as np
import pytest
import threading

from motocam.ai.ai_engine import AiEngine, Detection, NullDetector
from motocam.ai import ai_worker as ai_worker_module
from motocam.ai.ai_worker import AiWorker


def test_resize_interpolation_prefers_area_for_downscaling():
    interpolation = ai_worker_module._select_resize_interpolation(1920, 1080, 320, 180)
    if ai_worker_module.cv2 is not None:
        assert interpolation == ai_worker_module.cv2.INTER_AREA
    else:
        assert interpolation is None


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


def _large_frame(marker: int) -> np.ndarray:
    f = np.zeros((80, 80, 3), dtype=np.uint8)
    f[:] = marker
    return f


def test_take_drops_all_but_the_latest_submitted_frame():
    engine = AiEngine(detector=_CountingDetector())
    worker = AiWorker(engine)
    worker.submit(_frame(1))
    worker.submit(_frame(2))
    worker.submit(_frame(3))
    got = worker._take()
    assert got is not None
    assert int(got[0, 0, 0]) == 3  # only the newest survives
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


def test_detector_factory_runs_on_worker_thread(qapp):
    main_thread = threading.get_ident()
    factory_thread: dict[str, int] = {}
    engine = AiEngine(detector=NullDetector("initializing"))
    worker = AiWorker(engine, detector_factory=lambda: _factory_detector(factory_thread))

    worker.start()
    try:
        _spin_until(qapp, lambda: factory_thread.get("id") is not None, timeout_s=2.0)
    finally:
        worker.stop()

    assert factory_thread["id"] != main_thread
    assert engine.source == "fake"


def test_worker_downscales_inference_frame_and_remaps_detections(qapp):
    detector = _CountingDetector()
    engine = AiEngine(detector=detector)
    engine.enabled = True
    worker = AiWorker(engine, max_input_width=40)
    received: list = []
    worker.detections_ready.connect(received.append)
    worker.start()
    try:
        worker.submit(_large_frame(9))
        _spin_until(qapp, lambda: len(received) >= 1, timeout_s=2.0)
    finally:
        worker.stop()

    assert detector.calls == 1
    det = received[-1][0]
    assert det.x == 18
    assert det.w == 20
    assert worker.stats()["max_input_width"] == 40


def test_worker_adapts_submit_rate_to_inference_budget():
    engine = AiEngine(detector=_CountingDetector())
    worker = AiWorker(engine, max_fps=30.0, performance_budget_pct=25.0)

    worker._last_inference_ms = 100.0

    assert worker.submit(_frame(1)) is True
    assert worker.submit(_frame(2)) is False
    stats = worker.stats()
    assert stats["performance_budget_pct"] == 25.0
    assert stats["effective_max_fps"] == pytest.approx(2.5)


def test_worker_reports_current_busy_inference(monkeypatch):
    engine = AiEngine(detector=_CountingDetector())
    worker = AiWorker(engine)
    monkeypatch.setattr(ai_worker_module.time, "monotonic", lambda: 20.5)

    worker._busy_started_at = 20.0

    assert worker.stats()["current_busy_ms"] == pytest.approx(500.0)


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
def _factory_detector(factory_thread: dict[str, int]) -> _CountingDetector:
    factory_thread["id"] = threading.get_ident()
    return _CountingDetector()


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
