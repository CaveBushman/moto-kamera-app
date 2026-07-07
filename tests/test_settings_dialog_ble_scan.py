"""BLE scan results in Settings must never show unnamed peripherals -- the
gimbal always advertises a real name, so an unnamed entry can never be
the one the operator wants and only adds clutter (often outnumbering the
real, named devices in a crowded RF environment). The real source
(dji_rs4pro.py's _discover_ble_device_infos) already drops unnamed
devices entirely rather than labelling them "Unknown BLE device", so
this only needs to check the dialog's own defensive filter against
whatever it's handed."""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from motocam.ui.widgets.settings_dialog import SettingsDialog

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        _APP = QApplication([])
        app = _APP
    return app


@dataclass(frozen=True)
class _FakeBleDevice:
    name: str
    address: str
    rssi: int | None = None


def test_unnamed_devices_are_filtered_out_of_scan_results():
    _app()
    dialog = SettingsDialog()

    devices = [
        _FakeBleDevice(name="DJI RS4 PRO-094PP0", address="AA:BB:CC:DD:EE:FF", rssi=-60),
        _FakeBleDevice(name="", address="11:22:33:44:55:66", rssi=-70),
        _FakeBleDevice(name="   ", address="22:33:44:55:66:77", rssi=-80),
    ]

    dialog.set_ble_scan_results(devices)

    labels = [dialog.ble_device_combo.itemText(i) for i in range(dialog.ble_device_combo.count())]
    assert any("DJI RS4 PRO" in label for label in labels)
    assert not any("11:22:33" in label or "22:33:44" in label for label in labels)
    assert "1 named BLE device" in dialog.ble_scan_status_label.text()


def test_scan_with_only_unnamed_devices_reports_none_found():
    _app()
    dialog = SettingsDialog()

    devices = [
        _FakeBleDevice(name="", address="11:22:33:44:55:66"),
        _FakeBleDevice(name="   ", address="22:33:44:55:66:77"),
    ]

    dialog.set_ble_scan_results(devices)

    assert dialog.ble_device_combo.count() == 1
    assert dialog.ble_device_combo.itemData(0) is None
    assert "No named BLE devices found" in dialog.ble_scan_status_label.text()
