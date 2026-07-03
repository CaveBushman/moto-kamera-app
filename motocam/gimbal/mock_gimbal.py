"""Software-only gimbal backend for development without an RS 4 Pro attached.

Integrates velocity commands over time so manual/AI move commands are
visible in the UI orientation readout even with no hardware connected.
"""
from __future__ import annotations

import asyncio
import time

from motocam.gimbal.base import GimbalBackend


class MockGimbalBackend(GimbalBackend):
    source = "mock"

    def __init__(self):
        self._connected = False
        self._pan = 0.0
        self._tilt = 0.0
        self._roll = 0.0
        self._vel_pan = 0.0
        self._vel_tilt = 0.0
        self._last_update = time.monotonic()

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        await asyncio.sleep(0.2)
        self._connected = True
        self._last_update = time.monotonic()

    async def disconnect(self) -> None:
        self._connected = False

    async def set_velocity(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        self._integrate()
        self._vel_pan = pan_deg_s
        self._vel_tilt = tilt_deg_s

    async def go_home(self) -> None:
        self._pan = 0.0
        self._tilt = 0.0
        self._roll = 0.0
        self._vel_pan = 0.0
        self._vel_tilt = 0.0

    async def get_orientation(self) -> tuple[float, float, float]:
        self._integrate()
        return self._pan, self._tilt, self._roll

    def _integrate(self) -> None:
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now
        self._pan += self._vel_pan * dt
        self._tilt += self._vel_tilt * dt
