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
