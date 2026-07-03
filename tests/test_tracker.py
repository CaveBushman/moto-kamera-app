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
