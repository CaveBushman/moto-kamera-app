"""Ranking logic for the BLE device picker (Settings -> GIMBAL CONTROL ->
SCAN BLE DEVICES). The scan itself needs a real BT radio, but which
device lands at the top of the combo -- the one the operator taps -- is
pure logic and worth pinning: a name match must beat a non-match, and
among equals the stronger signal wins (physically closest gimbal)."""
from __future__ import annotations

from motocam.gimbal.dji_rs4pro import BleDeviceInfo, _sort_ble_device_infos


def test_name_match_ranks_above_non_match_even_with_weaker_signal():
    devices = [
        BleDeviceInfo("Random Speaker", "AA", rssi=-40),  # strong but wrong
        BleDeviceInfo("RS 4 Pro", "BB", rssi=-80),        # weak but the gimbal
    ]
    ranked = _sort_ble_device_infos(devices, name_filter="RS 4 Pro")
    assert ranked[0].name == "RS 4 Pro"


def test_among_name_matches_stronger_signal_wins():
    devices = [
        BleDeviceInfo("RS 4 Pro", "BB", rssi=-70),
        BleDeviceInfo("RS 4 Pro Gimbal", "CC", rssi=-55),
    ]
    ranked = _sort_ble_device_infos(devices, name_filter="RS 4 Pro")
    assert ranked[0].address == "CC"


def test_no_filter_sorts_by_signal_then_name():
    devices = [
        BleDeviceInfo("Beta", "B", rssi=-60),
        BleDeviceInfo("Alpha", "A", rssi=-30),
        BleDeviceInfo("Gamma", "G", rssi=None),  # unknown signal ranks last
    ]
    ranked = _sort_ble_device_infos(devices, name_filter="")
    assert [d.name for d in ranked] == ["Alpha", "Beta", "Gamma"]


def test_missing_rssi_does_not_crash_ranking():
    devices = [BleDeviceInfo("RS 4 Pro", "BB", rssi=None)]
    ranked = _sort_ble_device_infos(devices, name_filter="RS 4 Pro")
    assert ranked[0].address == "BB"


def test_label_includes_signal_when_present():
    assert "-55 dBm" in BleDeviceInfo("RS 4 Pro", "BB", rssi=-55).label
    assert "dBm" not in BleDeviceInfo("RS 4 Pro", "BB", rssi=None).label
