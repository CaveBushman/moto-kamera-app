import asyncio

from motocam.core.protocol import OperatingMode
from motocam.gimbal.base import GimbalController


class RecordingBackend:
    connected = True

    def __init__(self):
        self.home_calls = 0
        self.velocity_calls: list[tuple[float, float]] = []

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def set_velocity(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        self.velocity_calls.append((pan_deg_s, tilt_deg_s))

    async def go_home(self) -> None:
        self.home_calls += 1

    async def get_orientation(self) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0


def test_recenter_preserves_ai_assist_mode_and_tracking_output():
    backend = RecordingBackend()
    controller = GimbalController(backend)
    asyncio.run(controller.set_mode(OperatingMode.AI_ASSIST))

    asyncio.run(controller.recenter())
    asyncio.run(controller.ai_move(5.0, -2.0))

    assert controller.mode == OperatingMode.AI_ASSIST
    assert backend.home_calls == 1
    assert backend.velocity_calls == [(5.0, -2.0)]


def test_home_mode_still_blocks_ai_output_when_selected_as_mode():
    backend = RecordingBackend()
    controller = GimbalController(backend)

    asyncio.run(controller.set_mode(OperatingMode.HOME))
    asyncio.run(controller.ai_move(5.0, -2.0))

    assert controller.mode == OperatingMode.HOME
    assert backend.home_calls == 1
    assert backend.velocity_calls == []
