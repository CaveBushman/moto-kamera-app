"""Gimbal Controller interface (design doc 7, 10.6)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from motocam.core.protocol import OperatingMode


class GimbalBackend(ABC):
    """Backend talks to the physical gimbal (BLE, serial, ...). Controller (below)
    owns mode/failsafe logic and stays backend-agnostic."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...

    @abstractmethod
    async def set_velocity(self, pan_deg_s: float, tilt_deg_s: float) -> None: ...

    @abstractmethod
    async def go_home(self) -> None: ...

    @abstractmethod
    async def get_orientation(self) -> tuple[float, float, float]:
        """Returns (pan, tilt, roll) in degrees."""


class GimbalController:
    def __init__(self, backend: GimbalBackend, max_pan_speed: float = 20.0, max_tilt_speed: float = 12.0):
        self.backend = backend
        self.max_pan_speed = max_pan_speed
        self.max_tilt_speed = max_tilt_speed
        self.mode: OperatingMode = OperatingMode.MANUAL
        self.pan_deg = 0.0
        self.tilt_deg = 0.0
        self.roll_deg = 0.0

    @property
    def connected(self) -> bool:
        return self.backend.connected

    @property
    def source(self) -> str:
        return str(getattr(self.backend, "source", self.backend.__class__.__name__))

    async def connect(self) -> None:
        await self.backend.connect()

    async def disconnect(self) -> None:
        await self.backend.disconnect()

    async def set_mode(self, mode: OperatingMode) -> None:
        self.mode = mode
        if mode == OperatingMode.HOME:
            await self.backend.go_home()
        elif mode in (OperatingMode.LOCK, OperatingMode.RESET):
            await self.backend.set_velocity(0.0, 0.0)

    async def manual_move(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        if self.mode != OperatingMode.MANUAL:
            return
        await self._send_clamped(pan_deg_s, tilt_deg_s)

    async def ai_move(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        if self.mode not in (OperatingMode.AI_ASSIST, OperatingMode.FULL_AI):
            return
        await self._send_clamped(pan_deg_s, tilt_deg_s)

    async def stop(self) -> None:
        await self.backend.set_velocity(0.0, 0.0)

    async def refresh_orientation(self) -> None:
        if self.backend.connected:
            self.pan_deg, self.tilt_deg, self.roll_deg = await self.backend.get_orientation()

    async def _send_clamped(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        pan = max(-self.max_pan_speed, min(self.max_pan_speed, pan_deg_s))
        tilt = max(-self.max_tilt_speed, min(self.max_tilt_speed, tilt_deg_s))
        await self.backend.set_velocity(pan, tilt)
