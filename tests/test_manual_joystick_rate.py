import asyncio
from types import SimpleNamespace

from motocam.core.protocol import OperatingMode
from motocam.ui.main_window import MainWindow


def _window_stub():
    window = MainWindow.__new__(MainWindow)
    window.gimbal = SimpleNamespace(
        mode=OperatingMode.MANUAL,
        max_pan_speed=20.0,
        max_tilt_speed=12.0,
    )
    window._manual_pan_v = 0.0
    window._manual_tilt_v = 0.0
    window.preview = SimpleNamespace(
        joystick=SimpleNamespace(is_dragging=True),
        zoom_rocker=SimpleNamespace(is_dragging=False),
    )
    sent = []
    window._request_gimbal_velocity = lambda *args: sent.append(args)
    window._on_zoom_drag = lambda _speed: None
    return window, sent


def test_manual_drag_updates_desired_velocity_without_immediate_send():
    window, sent = _window_stub()
    MainWindow._on_manual_drag(window, 0.5, -0.25)

    assert sent == []
    assert window._manual_pan_v == 10.0
    assert window._manual_tilt_v == 3.0


def test_control_tick_is_the_only_manual_velocity_sender():
    window, sent = _window_stub()
    MainWindow._on_manual_drag(window, 0.5, -0.25)
    MainWindow._control_tick(window)

    assert sent == [(True, 10.0, 3.0)]


def test_manual_release_sends_stop_immediately():
    window, sent = _window_stub()
    MainWindow._on_manual_drag(window, 0.5, -0.25)
    MainWindow._on_manual_drag_end(window)

    assert window._manual_pan_v == 0.0
    assert window._manual_tilt_v == 0.0
    assert sent == [(True, 0.0, 0.0)]


def test_velocity_worker_drops_stale_commands_while_write_is_busy():
    class SlowGimbal:
        def __init__(self):
            self.calls: list[tuple[float, float]] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def manual_move(self, pan: float, tilt: float) -> None:
            self.calls.append((pan, tilt))
            if len(self.calls) == 1:
                self.first_started.set()
                await self.release_first.wait()

        async def ai_move(self, pan: float, tilt: float) -> None:
            raise AssertionError("manual test should not use AI movement")

    async def run_case():
        window = MainWindow.__new__(MainWindow)
        window.gimbal = SlowGimbal()
        window._gimbal_velocity_task = None
        window._latest_gimbal_velocity = None
        window._gimbal_velocity_event = None
        window._consume_task_result = lambda task, label: task.result()

        MainWindow._request_gimbal_velocity(window, True, 1.0, 1.0)
        await window.gimbal.first_started.wait()
        MainWindow._request_gimbal_velocity(window, True, 2.0, 2.0)
        MainWindow._request_gimbal_velocity(window, True, 3.0, 3.0)
        window.gimbal.release_first.set()
        await window._gimbal_velocity_task

        assert window.gimbal.calls == [(1.0, 1.0), (3.0, 3.0)]

    asyncio.run(run_case())
