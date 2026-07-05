"""Proof-of-life behaviour of the DJI R SDK backend: `connected` may only
turn True after the gimbal answers GET_POSITION with a CRC-valid reply.
Exercised with fake transports -- no hardware, no python-can needed."""
from __future__ import annotations

import asyncio

from motocam.gimbal.dji_rs4pro import BleDeviceInfo, BleTransport, DjiRs4ProBackend, _sort_ble_device_infos
from motocam.gimbal.rsdk_protocol import (
    CMD_GET_POSITION,
    CMD_SET_GIMBAL,
    RSdkFrame,
    build_frame,
    parse_frame,
)
import struct

REPLY_CMD_TYPE = 0x01


class SilentTransport:
    """Opens fine (adapter present) but the gimbal never answers --
    wrong wiring, wrong constants, gimbal off."""

    def open(self):
        pass

    def close(self):
        pass

    def send(self, frame: bytes):
        pass

    def receive_chunk(self, timeout_s: float) -> bytes:
        return b""


class EchoPositionTransport:
    """Answers every GET_POSITION with a valid reply frame, chunked to
    CAN-sized pieces to exercise reassembly on the receive path too."""

    def __init__(self):
        self._pending = b""
        self.sent_frames: list[RSdkFrame] = []

    def open(self):
        pass

    def close(self):
        pass

    def send(self, frame: bytes):
        parsed = parse_frame(frame)
        self.sent_frames.append(parsed)
        if parsed.cmd_set == CMD_SET_GIMBAL and parsed.cmd_id == CMD_GET_POSITION:
            payload = struct.pack("<hhh", 155, -20, -305)
            reply = RSdkFrame(REPLY_CMD_TYPE, parsed.sequence, CMD_SET_GIMBAL, CMD_GET_POSITION, payload)
            self._pending += build_frame(reply)

    def receive_chunk(self, timeout_s: float) -> bytes:
        chunk, self._pending = self._pending[:8], self._pending[8:]
        return chunk


class AsyncEchoPositionTransport(EchoPositionTransport):
    async_transport = True

    async def open(self):
        pass

    async def close(self):
        pass

    async def send(self, frame: bytes):
        super().send(frame)

    async def receive_chunk(self, timeout_s: float) -> bytes:
        return super().receive_chunk(timeout_s)


class FakeGattCharacteristic:
    def __init__(self, uuid: str, properties: list[str]):
        self.uuid = uuid
        self.properties = properties


class FakeGattService:
    def __init__(self, uuid: str, characteristics: list[FakeGattCharacteristic]):
        self.uuid = uuid
        self.characteristics = characteristics


def test_no_reply_means_not_connected():
    backend = DjiRs4ProBackend(SilentTransport())
    asyncio.run(backend.connect())
    assert backend.connected is False


def test_valid_position_reply_connects_and_seeds_orientation():
    backend = DjiRs4ProBackend(EchoPositionTransport())
    asyncio.run(backend.connect())
    assert backend.connected is True
    pan, tilt, roll = asyncio.run(backend.get_orientation())
    assert (pan, tilt, roll) == (15.5, -30.5, -2.0)


def test_velocity_commands_are_not_sent_while_disconnected():
    transport = EchoPositionTransport()
    backend = DjiRs4ProBackend(transport)
    asyncio.run(backend.set_velocity(10.0, 5.0))  # before connect
    assert transport.sent_frames == []

    asyncio.run(backend.connect())
    asyncio.run(backend.set_velocity(10.0, 5.0))
    assert any(f.cmd_id == 0x01 for f in transport.sent_frames)


def test_async_transport_is_supported_for_ble_style_io():
    backend = DjiRs4ProBackend(AsyncEchoPositionTransport())
    asyncio.run(backend.connect())
    assert backend.connected is True


def test_ble_autodiscovers_write_notify_pair_on_same_service():
    endpoint = BleTransport._select_characteristics(
        [
            FakeGattService(
                "service-a",
                [
                    FakeGattCharacteristic("notify-a", ["notify"]),
                    FakeGattCharacteristic("write-a", ["write-without-response"]),
                ],
            ),
            FakeGattService(
                "service-b",
                [
                    FakeGattCharacteristic("notify-b", ["notify"]),
                    FakeGattCharacteristic("write-b", ["write"]),
                ],
            ),
        ],
        service_uuid=None,
        tx_char_uuid=None,
        rx_char_uuid=None,
    )

    assert endpoint.tx_uuid == "write-a"
    assert endpoint.rx_uuid == "notify-a"
    assert endpoint.write_with_response is False


def test_ble_accepts_explicit_write_and_notify_uuids():
    endpoint = BleTransport._select_characteristics(
        [
            FakeGattService(
                "service-a",
                [
                    FakeGattCharacteristic("notify-a", ["notify"]),
                    FakeGattCharacteristic("write-a", ["write"]),
                ],
            )
        ],
        service_uuid="service-a",
        tx_char_uuid="write-a",
        rx_char_uuid="notify-a",
    )

    assert endpoint.tx_uuid == "write-a"
    assert endpoint.rx_uuid == "notify-a"
    assert endpoint.write_with_response is True


def test_ble_scan_results_prioritize_name_match_then_signal():
    devices = _sort_ble_device_infos(
        [
            BleDeviceInfo("Keyboard", "aa", -30),
            BleDeviceInfo("DJI RS 4 Pro", "bb", -82),
            BleDeviceInfo("RS 4 Pro Backup", "cc", -50),
        ],
        name_filter="RS 4 Pro",
    )

    assert [device.address for device in devices] == ["cc", "bb", "aa"]


# -- BLE DUML transport: connect on telemetry, control is an honest no-op --
class DumlTelemetryTransport:
    """RS 4 Pro BLE: DUML, and pushes telemetry without being asked. Records
    any writes so we can prove control is NOT sent over BLE yet."""

    protocol = "duml"
    async_transport = True

    def __init__(self):
        # a real captured gimbal DUML frame (cmd_set 0x04, cmd_id 0x27)
        self._pending = bytes.fromhex("551204c70402719500042700800000002014")
        self.sent: list[bytes] = []

    async def open(self):
        pass

    async def close(self):
        pass

    async def send(self, frame: bytes):
        self.sent.append(frame)

    async def receive_chunk(self, timeout_s: float) -> bytes:
        chunk, self._pending = self._pending[:8], self._pending[8:]
        return chunk


def test_ble_duml_connects_from_telemetry_without_sending():
    transport = DumlTelemetryTransport()
    backend = DjiRs4ProBackend(transport)
    asyncio.run(backend.connect())
    assert backend.connected is True
    assert transport.sent == []  # DUML proof-of-life needs no outgoing frame


def test_ble_duml_control_is_a_noop_not_wrong_frames():
    transport = DumlTelemetryTransport()
    backend = DjiRs4ProBackend(transport)
    asyncio.run(backend.connect())
    asyncio.run(backend.set_velocity(15.0, 8.0))
    asyncio.run(backend.go_home())
    # control isn't decoded over BLE -> we must NOT send 0xAA frames the
    # gimbal would ignore; stays a no-op until reverse-engineered.
    assert transport.sent == []


def test_ble_duml_no_telemetry_stays_disconnected():
    transport = DumlTelemetryTransport()
    transport._pending = b""  # gimbal silent
    backend = DjiRs4ProBackend(transport)
    asyncio.run(backend.connect())
    assert backend.connected is False
