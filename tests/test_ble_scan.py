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


def test_rs4_without_space_matches_rs_4_filter():
    devices = [
        BleDeviceInfo("Random Speaker", "AA", rssi=-40),
        BleDeviceInfo("DJI RS4 PRO-094PP0", "BB", rssi=-80),
    ]
    ranked = _sort_ble_device_infos(devices, name_filter="RS 4 Pro")
    assert ranked[0].address == "BB"


def test_label_includes_signal_when_present():
    assert "-55 dBm" in BleDeviceInfo("RS 4 Pro", "BB", rssi=-55).label
    assert "dBm" not in BleDeviceInfo("RS 4 Pro", "BB", rssi=None).label


# -- BlueZ adapter contention (org.bluez.Error.InProgress) -----------------
# BlueZ allows only one discovery/connect session per adapter; a second
# concurrent one raises "org.bluez.Error.InProgress" rather than queuing.
# Live-hit on the Pi when the periodic auto-reconnect and the operator's
# manual "SCAN BLE DEVICES" button raced. Fixed with a shared adapter lock
# (_BLE_ADAPTER_LOCK) plus a retry for any residual external race.
import asyncio

import pytest

from motocam.gimbal import dji_rs4pro
from motocam.gimbal.dji_rs4pro import _retry_bluez_in_progress


def test_retry_bluez_in_progress_retries_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("[org.bluez.Error.InProgress] Operation already in progress")
        return "ok"

    result = asyncio.run(_retry_bluez_in_progress(flaky))
    assert result == "ok"
    assert calls["n"] == 3


def test_retry_bluez_in_progress_gives_up_after_max_retries():
    async def always_busy():
        raise RuntimeError("[org.bluez.Error.InProgress] Operation already in progress")

    with pytest.raises(RuntimeError, match="InProgress"):
        asyncio.run(_retry_bluez_in_progress(always_busy))


def test_retry_bluez_in_progress_does_not_retry_unrelated_errors():
    calls = {"n": 0}

    async def other_error():
        calls["n"] += 1
        raise RuntimeError("some unrelated failure")

    with pytest.raises(RuntimeError, match="unrelated"):
        asyncio.run(_retry_bluez_in_progress(other_error))
    assert calls["n"] == 1  # no retry attempted for a non-InProgress error


def test_locked_scan_for_device_does_not_deadlock_on_the_adapter_lock():
    # Regression: BleTransport.open() holds _BLE_ADAPTER_LOCK while calling
    # _scan_for_device(locked=True); that path must NOT try to reacquire
    # the same (non-reentrant) lock via the public scan_ble_devices(), or
    # every BLE connect that needs to scan (no configured address) hangs
    # forever. wait_for() turns a real deadlock into a fast test failure
    # instead of hanging the suite.
    from motocam.gimbal.dji_rs4pro import BleDeviceInfo, BleTransport

    async def fake_discover(timeout_s):
        return [BleDeviceInfo(name="DJI RS4 PRO-094PP0", address="AA:BB", rssi=-50)]

    async def run():
        transport = BleTransport(name="DJI RS4 PRO-094PP0")
        async with dji_rs4pro._BLE_ADAPTER_LOCK:  # simulate open() already holding it
            orig = dji_rs4pro._discover_ble_device_infos
            dji_rs4pro._discover_ble_device_infos = fake_discover
            try:
                return await asyncio.wait_for(transport._scan_for_device(locked=True), timeout=2.0)
            finally:
                dji_rs4pro._discover_ble_device_infos = orig

    address = asyncio.run(run())
    assert address == "AA:BB"
