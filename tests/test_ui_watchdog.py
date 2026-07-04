"""Tests for the UI event-loop stall watchdog. The stall arithmetic is
pure; a live run confirms an on-time timer stays silent while a blocked
UI thread is detected and logged."""
from __future__ import annotations

import logging
import time

from motocam.watchdog.ui_watchdog import UiLatencyWatchdog, stall_ms


def test_on_time_tick_reports_no_stall():
    assert stall_ms(expected_interval_ms=250, actual_delta_ms=260, tolerance_ms=250) is None


def test_late_tick_reports_overshoot():
    # 250ms tick that actually took 900ms -> ~650ms overshoot beyond interval
    assert stall_ms(250, 900, 250) == 650


def test_small_jitter_within_tolerance_is_ignored():
    assert stall_ms(250, 480, 250) is None  # 230ms overshoot < tolerance
    assert stall_ms(250, 520, 250) == 270  # 270ms overshoot > tolerance


def test_watchdog_detects_a_blocked_ui_thread(qapp, caplog):
    wd = UiLatencyWatchdog(interval_ms=50, tolerance_ms=150)
    wd.start()
    # Simulate the UI thread being blocked: process events, then hog the
    # thread so the timer can't fire on schedule.
    with caplog.at_level(logging.WARNING, logger="motocam.watchdog.ui"):
        qapp.processEvents()
        time.sleep(0.5)  # block ~500ms -- far beyond the 50ms tick
        qapp.processEvents()
    wd.stop()
    assert wd.stall_count >= 1
    assert wd.max_stall_ms > 150


def test_watchdog_stays_silent_when_responsive(qapp):
    wd = UiLatencyWatchdog(interval_ms=50, tolerance_ms=150)
    wd.start()
    deadline = time.monotonic() + 0.4
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)  # keep servicing the loop -> no stalls
    wd.stop()
    assert wd.stall_count == 0


import pytest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
