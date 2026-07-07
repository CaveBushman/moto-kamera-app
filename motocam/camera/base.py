"""Camera Controller interface (design doc 6.3, 10.7)."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger("motocam.camera")


@dataclass
class CameraState:
    connected: bool = False
    recording: bool = False
    iso: int | None = None
    white_balance: int | None = None
    shutter: str | None = None
    iris: str | None = None
    fps: float | None = None
    media_remaining_min: float | None = None
    battery_pct: float | None = None


class CameraBackend(ABC):
    """Talks to the physical camera (e.g. camera/bmd_rest_camera.py's
    Blackmagic REST API backend, or camera/mock_camera.py for dev without
    hardware). CameraController (below) is the backend-agnostic front the
    rest of the app uses."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...

    @abstractmethod
    async def get_state(self) -> CameraState: ...

    @abstractmethod
    async def start_record(self) -> None: ...

    @abstractmethod
    async def stop_record(self) -> None: ...

    @abstractmethod
    async def set_iso(self, iso: int) -> None: ...

    @abstractmethod
    async def set_white_balance(self, kelvin: int) -> None: ...

    @abstractmethod
    async def set_shutter(self, shutter: str) -> None: ...

    @abstractmethod
    async def set_iris(self, iris: str) -> None: ...

    @abstractmethod
    async def trigger_autofocus(self) -> None: ...

    @abstractmethod
    async def set_zoom_speed(self, speed: float) -> None: ...


class CameraController:
    """Owns the published CameraState and reconnect/error handling so UI
    code can fire commands without try/except around every button press."""

    def __init__(self, backend: CameraBackend):
        self.backend = backend
        self.state = CameraState()

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

    async def refresh(self) -> None:
        # Auto-reconnect: a camera that comes online mid-session (powered
        # up late, IP fixed in Settings, cable replugged) gets picked up
        # by the regular refresh tick -- backends throttle their own
        # probe rate, so this stays cheap while disconnected.
        if not self.backend.connected:
            try:
                await self.backend.connect()
            except Exception as exc:  # noqa: BLE001 -- refresh must never take the UI down
                logger.debug("camera reconnect attempt failed: %s", exc)
        if self.backend.connected:
            try:
                self.state = await self.backend.get_state()
            except Exception as exc:  # noqa: BLE001
                logger.warning("camera state refresh failed: %s", exc)
                self.state = CameraState(connected=False)
        else:
            self.state = CameraState(connected=False)

    async def _command(self, name: str, coro) -> None:
        """Commands are fired via ensure_future from UI slots -- an
        exception here would otherwise only surface as an unretrieved
        task error long after the button press it belongs to."""
        try:
            await coro
        except Exception as exc:  # noqa: BLE001
            logger.warning("camera command %s failed: %s", name, exc)

    async def start_record(self) -> None:
        await self._command("start_record", self.backend.start_record())

    async def stop_record(self) -> None:
        await self._command("stop_record", self.backend.stop_record())

    async def set_iso(self, iso: int) -> None:
        await self._command("set_iso", self.backend.set_iso(iso))

    async def set_white_balance(self, kelvin: int) -> None:
        await self._command("set_white_balance", self.backend.set_white_balance(kelvin))

    async def set_shutter(self, shutter: str) -> None:
        await self._command("set_shutter", self.backend.set_shutter(shutter))

    async def set_iris(self, iris: str) -> None:
        await self._command("set_iris", self.backend.set_iris(iris))

    async def trigger_autofocus(self) -> None:
        await self._command("trigger_autofocus", self.backend.trigger_autofocus())

    async def set_zoom_speed(self, speed: float) -> None:
        await self._command("set_zoom_speed", self.backend.set_zoom_speed(max(-1.0, min(1.0, speed))))
