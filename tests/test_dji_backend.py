"""Proof-of-life behaviour of the DJI R SDK backend: `connected` may only
turn True after the gimbal answers GET_POSITION with a CRC-valid reply.
Exercised with fake transports -- no hardware, no python-can needed."""
from __future__ import annotations

import asyncio

from motocam.gimbal.dji_duml import JOYSTICK_CENTER, DjiDumlFrame
from motocam.gimbal.dji_rs4pro import (
    VELOCITY_SEND_TIMEOUT_S,
    BleDeviceInfo,
    BleTransport,
    DjiRs4ProBackend,
    _sort_ble_device_infos,
)
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
        self.mtu_payload_bytes = 244

    async def open(self):
        pass

    async def close(self):
        pass

    async def send(self, frame: bytes):
        self.sent.append(frame)

    async def receive_chunk(self, timeout_s: float) -> bytes:
        chunk, self._pending = self._pending[:8], self._pending[8:]
        return chunk

    async def drain(self) -> bytes:
        rest, self._pending = self._pending, b""
        return rest


def test_ble_duml_connects_from_telemetry_without_sending():
    transport = DumlTelemetryTransport()
    backend = DjiRs4ProBackend(transport)
    asyncio.run(backend.connect())
    assert backend.connected is True
    assert transport.sent == []  # DUML proof-of-life needs no outgoing frame


def test_ble_duml_set_velocity_sends_a_joystick_frame():
    # Velocity control IS decoded (live-confirmed against real hardware,
    # see docs/RS4_BLE_FINDINGS.md): must send a real DUML joystick frame,
    # not stay a no-op.
    transport = DumlTelemetryTransport()
    backend = DjiRs4ProBackend(transport, max_pan_speed=20.0, max_tilt_speed=12.0)
    asyncio.run(backend.connect())
    asyncio.run(backend.set_velocity(10.0, -6.0))  # half deflection each axis
    assert len(transport.sent) == 1
    frame = DjiDumlFrame.parse(transport.sent[0])
    assert frame.cmd_set == 0x04 and frame.cmd_id == 0x01
    a, b, c = struct.unpack_from("<HHH", frame.payload, 0)
    # pan_deg_s=10 (half of max 20) -> chC (pan) ratio +0.5, above center.
    # tilt_deg_s=-6 (half of max 12, negative) -> chA (tilt) ratio -0.5,
    # below center. Third channel (chB) stays neutral (untouched axis).
    assert a < JOYSTICK_CENTER
    assert b == JOYSTICK_CENTER
    assert c > JOYSTICK_CENTER


class SlowDumlWriteTransport(DumlTelemetryTransport):
    async def send(self, frame: bytes):
        await asyncio.sleep(VELOCITY_SEND_TIMEOUT_S + 0.05)
        self.sent.append(frame)


def test_ble_duml_slow_velocity_write_is_dropped_without_disconnect():
    transport = SlowDumlWriteTransport()
    backend = DjiRs4ProBackend(transport, max_pan_speed=20.0, max_tilt_speed=12.0)
    asyncio.run(backend.connect())
    asyncio.run(backend.set_velocity(10.0, -6.0))

    assert backend.connected is True
    assert transport.sent == []


def test_ble_duml_velocity_timeout_is_configurable():
    transport = SlowDumlWriteTransport()
    backend = DjiRs4ProBackend(
        transport,
        max_pan_speed=20.0,
        max_tilt_speed=12.0,
        velocity_send_timeout_s=0.05,
    )

    assert backend.velocity_send_timeout_s == 0.05


def test_ble_duml_velocity_stats_report_write_gap_and_timeouts():
    transport = DumlTelemetryTransport()
    backend = DjiRs4ProBackend(transport, max_pan_speed=20.0, max_tilt_speed=12.0)
    asyncio.run(backend.connect())
    asyncio.run(backend.set_velocity(10.0, -6.0))
    asyncio.run(backend.set_velocity(0.0, 0.0))

    stats = backend.velocity_stats()

    assert stats["velocity_write_ms_avg"] is not None
    assert stats["velocity_write_ms_max"] is not None
    assert stats["velocity_call_gap_ms_avg"] is not None
    assert stats["velocity_call_gap_ms_max"] is not None
    assert stats["velocity_timeouts"] == 0
    assert stats["ble_mtu_payload_bytes"] == 244
    assert stats["ble_degraded"] is False


def test_ble_duml_reports_degraded_low_mtu_and_throttles_non_stop_velocity():
    transport = DumlTelemetryTransport()
    transport.mtu_payload_bytes = 20
    backend = DjiRs4ProBackend(transport, max_pan_speed=20.0, max_tilt_speed=12.0)
    asyncio.run(backend.connect())

    asyncio.run(backend.set_velocity(10.0, -6.0))
    asyncio.run(backend.set_velocity(12.0, -6.0))
    asyncio.run(backend.set_velocity(0.0, 0.0))
    stats = backend.velocity_stats()

    assert len(transport.sent) == 2
    assert stats["ble_mtu_payload_bytes"] == 20
    assert stats["ble_degraded"] is True
    assert stats["velocity_throttle_drops"] == 1


def test_ble_duml_health_check_marks_disconnected_transport_down():
    transport = DumlTelemetryTransport()
    transport.is_connected = lambda: False
    backend = DjiRs4ProBackend(transport)
    asyncio.run(backend.connect())

    assert asyncio.run(backend.check_connection()) is False
    assert backend.connected is False


def test_ble_duml_health_check_keeps_control_alive_on_stale_notifications():
    transport = DumlTelemetryTransport()
    transport.is_connected = lambda: True
    transport.notification_age_s = lambda: 9.0
    backend = DjiRs4ProBackend(transport, notify_stale_s=3.0)
    asyncio.run(backend.connect())

    assert asyncio.run(backend.check_connection()) is True
    assert backend.connected is True


def test_ble_duml_go_home_sends_the_recenter_frame():
    # Recenter IS decoded (live-confirmed against real hardware -- gimbal
    # visibly returned to center, see docs/RS4_BLE_FINDINGS.md).
    transport = DumlTelemetryTransport()
    backend = DjiRs4ProBackend(transport)
    asyncio.run(backend.connect())
    asyncio.run(backend.go_home())
    assert len(transport.sent) == 1
    frame = DjiDumlFrame.parse(transport.sent[0])
    assert frame.cmd_set == 0x04 and frame.cmd_id == 0x4C
    assert frame.sender == 0x02 and frame.receiver == 0x04
    assert frame.payload == bytes.fromhex("fe01")


def test_ble_duml_no_telemetry_stays_disconnected():
    transport = DumlTelemetryTransport()
    transport._pending = b""  # gimbal silent
    backend = DjiRs4ProBackend(transport)
    asyncio.run(backend.connect())
    assert backend.connected is False


def test_ble_duml_get_orientation_drains_and_counts_frames():
    transport = DumlTelemetryTransport()
    backend = DjiRs4ProBackend(transport)
    asyncio.run(backend.connect())
    frame = "551204c70402719500042700800000002014"
    transport._pending = bytes.fromhex(frame) * 3  # three telemetry frames queued
    asyncio.run(backend.get_orientation())
    assert backend._duml_frame_count >= 3
    # the notify stream was drained (no unbounded growth)
    assert asyncio.run(transport.drain()) == b""


def test_ble_duml_capture_writes_frames_when_env_set(tmp_path, monkeypatch):
    out = tmp_path / "duml_capture.jsonl"
    monkeypatch.setenv("MOTOCAM_DUML_CAPTURE", str(out))
    transport = DumlTelemetryTransport()
    backend = DjiRs4ProBackend(transport)  # reads env at construction
    asyncio.run(backend.connect())
    transport._pending = bytes.fromhex("551204c70402719500042700800000002014") * 2
    asyncio.run(backend.get_orientation())
    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 2
    import json as _json
    row = _json.loads(lines[0])
    assert row["cmd_set"] == 0x04 and row["cmd_id"] == 0x27 and "hex" in row
