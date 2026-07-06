"""UI event-loop stall watchdog (stability observability).

After moving capture, AI inference, GPS reads and CSRT tracking off the
Qt UI thread, this is the canary that proves it -- and that catches any
future regression in the field. It runs a short self-timer on the UI
thread and measures how late each tick actually fires: if the gap far
exceeds the interval, the event loop was blocked by something running on
the UI thread, and we log how long the stall was. On a healthy unit it
stays silent.

The stall arithmetic is a pure function so it can be tested without a
running event loop; the QObject just wires it to a QTimer.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from PyQt6.QtCore import QObject, QTimer

logger = logging.getLogger("motocam.watchdog.ui")


def stall_ms(expected_interval_ms: float, actual_delta_ms: float, tolerance_ms: float) -> float | None:
    """How long the UI thread was blocked, or None if the tick was on time.

    A tick firing `expected_interval_ms` apart is healthy; anything beyond
    that plus `tolerance_ms` means the event loop couldn't service the
    timer -- the overshoot is (roughly) how long it was blocked."""
    overshoot = actual_delta_ms - expected_interval_ms
    return overshoot if overshoot > tolerance_ms else None


class UiLatencyWatchdog(QObject):
    def __init__(
        self,
        interval_ms: int = 250,
        tolerance_ms: int = 250,
        parent: QObject | None = None,
        context_provider: Callable[[], str] | None = None,
    ):
        super().__init__(parent)
        self._interval_ms = interval_ms
        self._tolerance_ms = tolerance_ms
        self._context_provider = context_provider
        self._last: float | None = None
        self.stall_count = 0
        self.max_stall_ms = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._last = time.monotonic()
        self._timer.start(self._interval_ms)

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        now = time.monotonic()
        if self._last is not None:
            delta_ms = (now - self._last) * 1000.0
            stalled = stall_ms(self._interval_ms, delta_ms, self._tolerance_ms)
            if stalled is not None:
                self.stall_count += 1
                self.max_stall_ms = max(self.max_stall_ms, stalled)
                context = self._context()
                logger.warning(
                    "UI event loop stalled for %.0f ms (expected %d ms tick) "
                    "-- something is blocking the UI thread [stall #%d, max %.0f ms]%s",
                    stalled, self._interval_ms, self.stall_count, self.max_stall_ms, context,
                )
        self._last = now

    def _context(self) -> str:
        if self._context_provider is None:
            return ""
        try:
            value = self._context_provider()
        except Exception as exc:  # noqa: BLE001 -- watchdog logging must never break the timer
            return f" context_error={exc!r}"
        return f" context={value}" if value else ""
