from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from motocam.core.protocol import TargetState
from motocam.tracking import tracker as tracker_module
from motocam.tracking.tracker import TrackingEngine

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_initial_state_is_idle_with_no_bbox():
    engine = TrackingEngine()
    assert engine.state == TargetState.IDLE
    assert engine.bbox is None


def test_select_at_locks_a_target():
    engine = TrackingEngine()
    engine.select_at(FRAME, 320, 240)
    assert engine.state == TargetState.LOCKED
    assert engine.bbox is not None
    _, _, w, h = engine.bbox
    assert (w, h) == (120, 120)  # default box_size


def test_select_at_clamps_box_to_frame_bounds_near_edges():
    engine = TrackingEngine()
    engine.select_at(FRAME, 2, 2)  # tap right at the top-left corner
    bx, by, bw, bh = engine.bbox
    assert bx >= 0 and by >= 0
    assert bx + bw <= FRAME.shape[1]
    assert by + bh <= FRAME.shape[0]


def test_clear_resets_to_idle():
    engine = TrackingEngine()
    engine.select_at(FRAME, 320, 240)
    engine.clear()
    assert engine.state == TargetState.IDLE
    assert engine.bbox is None


def test_error_from_center_is_none_without_a_target():
    engine = TrackingEngine()
    assert engine.error_from_center(FRAME.shape) is None


def test_error_from_center_near_zero_for_a_centered_target():
    engine = TrackingEngine()
    engine.select_at(FRAME, 320, 240)  # dead center of a 640x480 frame
    error = engine.error_from_center(FRAME.shape)
    assert error is not None
    ex, ey = error
    assert abs(ex) < 5 and abs(ey) < 5


def test_lost_tracking_degrades_locked_then_weak_then_manual_required(monkeypatch):
    """A dropped target isn't reported as lost immediately -- there's a
    grace period (WEAK_AFTER_S) so a single bad frame doesn't force the
    rider to re-tap, then a WEAK state, then MANUAL_REQUIRED once it's
    genuinely gone (LOST_AFTER_S)."""
    base_time = 1_000.0
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: base_time)

    engine = TrackingEngine()
    engine.select_at(FRAME, 320, 240)  # _last_good_time pinned to base_time

    engine._tracker = MagicMock()
    engine._tracker.update.return_value = (False, None)

    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: base_time + 0.1)
    engine.update(FRAME)
    assert engine.state == TargetState.LOCKED  # still within the grace period

    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: base_time + 1.0)
    engine.update(FRAME)
    assert engine.state == TargetState.WEAK  # past WEAK_AFTER_S, before LOST_AFTER_S

    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: base_time + 3.0)
    engine.update(FRAME)
    assert engine.state == TargetState.MANUAL_REQUIRED
    assert engine.bbox is None
    assert engine._tracker is None


def test_reacquired_target_after_weak_returns_to_locked(monkeypatch):
    base_time = 2_000.0
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: base_time)
    engine = TrackingEngine()
    engine.select_at(FRAME, 320, 240)

    fake_tracker = MagicMock()
    engine._tracker = fake_tracker
    fake_tracker.update.return_value = (False, None)
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: base_time + 1.0)
    engine.update(FRAME)
    assert engine.state == TargetState.WEAK

    fake_tracker.update.return_value = (True, (300, 220, 120, 120))
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: base_time + 1.1)
    engine.update(FRAME)
    assert engine.state == TargetState.LOCKED
    assert engine.bbox == (300, 220, 120, 120)


# -- off-thread worker (stability: CSRT update() must not run on the UI thread) --
import time as _time  # noqa: E402


def _wait(predicate, timeout_s: float = 1.0) -> None:
    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline and not predicate():
        _time.sleep(0.01)


def test_submit_keeps_only_the_latest_frame():
    engine = TrackingEngine()
    f1 = np.zeros((4, 4, 3), np.uint8)
    f2 = np.ones((4, 4, 3), np.uint8)
    engine.submit(f1)
    engine.submit(f2)
    assert engine._take() is f2  # older frame dropped
    assert engine._take() is None  # consumed once


def test_worker_thread_runs_update_and_publishes_bbox():
    engine = TrackingEngine()
    engine.select_at(FRAME, 320, 240)
    fake = MagicMock()
    fake.update.return_value = (True, (100, 110, 120, 120))
    engine._tracker = fake
    engine.start()
    try:
        engine.submit(FRAME)
        _wait(lambda: engine.bbox == (100, 110, 120, 120))
    finally:
        engine.stop()
    assert engine.bbox == (100, 110, 120, 120)
    assert engine.state == TargetState.LOCKED


def test_stop_joins_the_worker_thread():
    engine = TrackingEngine()
    engine.start()
    engine.stop()
    assert engine._worker_thread is None
