"""VideoEngine stability tests: the capture read loop must run off the Qt
UI thread (a blocking grabber must never freeze the UI), frames must be
delivered back on the UI thread, and stop() must join the loop cleanly.
Uses a non-existent device so the engine takes its synthetic-fallback
path -- no real hardware required."""
from __future__ import annotations

import threading
import time

import pytest

from motocam.video.video_engine import VideoEngine


def _pump(qapp, predicate, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    qapp.processEvents()


def test_synthetic_frames_delivered_on_ui_thread(qapp):
    ve = VideoEngine(device="/dev/motocam-nonexistent", width=160, height=120, fps=30)
    frames: list = []
    delivery_thread: dict = {}
    main_thread = threading.get_ident()
    ve.frame_ready.connect(lambda f: frames.append(f.shape))
    ve.frame_ready.connect(lambda f: delivery_thread.setdefault("id", threading.get_ident()))

    ve.start()
    try:
        _pump(qapp, lambda: len(frames) >= 3, timeout_s=2.0)
    finally:
        ve.stop()
    qapp.processEvents()

    assert len(frames) >= 3
    assert ve.source == "synthetic"
    # frame_ready must be a queued delivery onto the UI thread, not the
    # capture thread -- that is the whole point of the off-thread design.
    assert delivery_thread.get("id") == main_thread


def test_stop_joins_capture_thread(qapp):
    ve = VideoEngine(device="/dev/motocam-nonexistent", width=160, height=120, fps=60)
    ve.start()
    _pump(qapp, lambda: False, timeout_s=0.2)  # let the loop spin up
    ve.stop()
    assert ve._thread is None or not ve._thread.is_alive()


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
