from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from motocam.ui.widgets.settings_dialog import SettingsDialog

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        _APP = QApplication([])
        app = _APP
    return app


def test_settings_dialog_emits_gps_device_and_baudrate():
    app = _app()
    dialog = SettingsDialog()
    emitted: list[tuple[str, object]] = []
    dialog.gps_apply_requested.connect(lambda device, baudrate: emitted.append((device, baudrate)))

    dialog.set_gps_values("/dev/serial/by-id/navilock", 115200)
    dialog._emit_gps_apply()

    assert emitted == [("/dev/serial/by-id/navilock", 115200)]


def test_settings_dialog_supports_auto_gps_baudrate():
    app = _app()
    dialog = SettingsDialog()
    emitted: list[tuple[str, object]] = []
    dialog.gps_apply_requested.connect(lambda device, baudrate: emitted.append((device, baudrate)))

    dialog.set_gps_values("auto", "auto")
    dialog._emit_gps_apply()

    assert emitted == [("auto", "auto")]


def test_settings_dialog_emits_ble_joystick_timeout():
    app = _app()
    dialog = SettingsDialog()
    emitted: list[dict] = []
    dialog.gimbal_apply_requested.connect(emitted.append)

    dialog.set_gimbal_values({"connection": "ble", "ble_velocity_timeout_s": 0.09, "ble_notify_stale_s": 4.5})
    dialog._emit_gimbal_apply()

    assert emitted[-1]["ble_velocity_timeout_s"] == 0.09
    assert emitted[-1]["ble_notify_stale_s"] == 4.5


def test_settings_dialog_uses_full_screen_scroll_layout():
    app = _app()
    dialog = SettingsDialog()

    dialog.show()
    app.processEvents()

    available = dialog.screen().availableGeometry()
    assert dialog.width() <= available.width()
    assert dialog.height() <= available.height()
    assert dialog.width() >= int(available.width() * 0.95)
    assert dialog.height() >= int(available.height() * 0.95)
    assert dialog.scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert dialog.scroll_area.verticalScrollBar().singleStep() == 32

    dialog.close()
