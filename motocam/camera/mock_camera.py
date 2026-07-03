"""Software-only Blackmagic PYXIS backend for development without the camera."""
from __future__ import annotations

import asyncio

from motocam.camera.base import CameraBackend, CameraState


class MockCameraBackend(CameraBackend):
    source = "mock"

    def __init__(self):
        self._connected = False
        self._state = CameraState(
            connected=False, recording=False, iso=400, white_balance=5600,
            shutter="1/50", iris="f/2.8", fps=50.0, media_remaining_min=180.0,
            battery_pct=87.0,
        )
        self.zoom_speed = 0.0
        self.autofocus_count = 0

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        await asyncio.sleep(0.2)
        self._connected = True
        self._state.connected = True

    async def disconnect(self) -> None:
        self._connected = False
        self._state.connected = False

    async def get_state(self) -> CameraState:
        return self._state

    async def start_record(self) -> None:
        self._state.recording = True

    async def stop_record(self) -> None:
        self._state.recording = False

    async def set_iso(self, iso: int) -> None:
        self._state.iso = iso

    async def set_white_balance(self, kelvin: int) -> None:
        self._state.white_balance = kelvin

    async def set_shutter(self, shutter: str) -> None:
        self._state.shutter = shutter

    async def set_iris(self, iris: str) -> None:
        self._state.iris = iris

    async def trigger_autofocus(self) -> None:
        self.autofocus_count += 1

    async def set_zoom_speed(self, speed: float) -> None:
        self.zoom_speed = max(-1.0, min(1.0, speed))
