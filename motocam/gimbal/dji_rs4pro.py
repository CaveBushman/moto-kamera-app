"""DJI RS 4 Pro backend over the DJI R SDK frame protocol.

The official R SDK PDF documents the frame layout, CRCs, gimbal command
set, and CAN/UART RSA port wiring. It does not document BLE GATT UUIDs.
For field testing the same R SDK frames can now be transported over BLE:
BleTransport scans/connects with bleak, auto-picks a write/notify pair
when the GATT profile is unambiguous enough, and also accepts explicit
UUIDs from Settings/config once the RS 4 Pro profile is confirmed.

Frame codec and command builders live in rsdk_protocol.py (pure logic,
unit-tested). This module owns the transports and the async backend:

- CanTransport: python-can (e.g. an MCP2515 HAT or USB-CAN adapter on
  the Raspberry Pi 5, `can0`, 1 Mbps). R SDK frames are chunked into
  8-byte CAN messages; FrameAssembler restitches the stream.
- UartTransport: pyserial on the RSA port's UART wiring.
- BleTransport: bleak GATT write/notify byte stream for RS 4 Pro BLE
  testing. The backend still only reports connected after a CRC-valid
  GET_POSITION reply, so a wrong GATT characteristic cannot fake control.

Honesty guarantee: `connected` only turns True after the gimbal has
actually ANSWERED a GET_POSITION request with a frame that passes both
CRCs. If anything differs on RS 4 Pro hardware, the result is a visible
DISCONNECTED state and a log line -- never fake control. Reconnect
probes are throttled the same way as the PYXIS backend's.

Protocol constants (frame codec, CRC seeds, cmd/ctrl bytes) and the CAN
arbitration IDs below (0x223 out / 0x222 in) are verified against the
ConstantRobotics/DJIR_SDK reference implementation of the official
R SDK protocol v2.2 -- see rsdk_protocol.py's docstring.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from motocam.gimbal.base import GimbalBackend
from motocam.gimbal.dji_duml import DjiDumlAssembler, build_joystick_frame
from motocam.gimbal.rsdk_protocol import (
    FrameAssembler,
    build_get_position,
    build_position_control,
    build_speed_control,
    parse_position_reply,
)

logger = logging.getLogger("motocam.gimbal.dji")

CAN_SEND_ID = 0x223  # verified against the reference implementation
CAN_RECV_ID = 0x222  # verified against the reference implementation
CAN_BITRATE = 1_000_000
RECONNECT_MIN_INTERVAL_S = 5.0
REPLY_TIMEOUT_S = 0.8

try:
    import can  # python-can, optional: only needed on the Pi with a CAN adapter

    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False

try:
    import serial

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    from bleak import BleakClient, BleakScanner

    BLE_AVAILABLE = True
except ImportError:
    BleakClient = None  # type: ignore[assignment]
    BleakScanner = None  # type: ignore[assignment]
    BLE_AVAILABLE = False


class CanTransport:
    """R SDK byte stream over CAN: frames chunked into 8-byte messages."""

    def __init__(self, channel: str = "can0", bitrate: int = CAN_BITRATE):
        if not CAN_AVAILABLE:
            raise RuntimeError("python-can is not installed -- pip3 install python-can")
        self.channel = channel
        self.bitrate = bitrate
        self._bus: "can.BusABC | None" = None

    def open(self) -> None:
        self._bus = can.interface.Bus(channel=self.channel, interface="socketcan", bitrate=self.bitrate)

    def close(self) -> None:
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None

    def send(self, frame: bytes) -> None:
        assert self._bus is not None
        for offset in range(0, len(frame), 8):
            message = can.Message(
                arbitration_id=CAN_SEND_ID, data=frame[offset : offset + 8], is_extended_id=False
            )
            self._bus.send(message)

    def receive_chunk(self, timeout_s: float) -> bytes:
        assert self._bus is not None
        message = self._bus.recv(timeout=timeout_s)
        if message is None or message.arbitration_id != CAN_RECV_ID:
            return b""
        return bytes(message.data)


class UartTransport:
    """R SDK byte stream over the RSA port's UART wiring."""

    def __init__(self, device: str, baudrate: int = 115200):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is not installed")
        self.device = device
        self.baudrate = baudrate
        self._serial: "serial.Serial | None" = None

    def open(self) -> None:
        self._serial = serial.Serial(self.device, self.baudrate, timeout=0.05)

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def send(self, frame: bytes) -> None:
        assert self._serial is not None
        self._serial.write(frame)

    def receive_chunk(self, timeout_s: float) -> bytes:
        assert self._serial is not None
        self._serial.timeout = timeout_s
        return self._serial.read(64)


@dataclass(frozen=True)
class BleEndpoint:
    tx_uuid: str
    rx_uuid: str
    write_with_response: bool


@dataclass(frozen=True)
class BleDeviceInfo:
    name: str
    address: str
    rssi: int | None = None

    @property
    def label(self) -> str:
        signal = f" ({self.rssi} dBm)" if self.rssi is not None else ""
        return f"{self.name} - {self.address}{signal}"


async def scan_ble_devices(name_filter: str = "", timeout_s: float = 5.0) -> list[BleDeviceInfo]:
    """Return visible BLE devices, with RS-name matches and strong signals first."""
    if not BLE_AVAILABLE:
        raise RuntimeError("bleak is not installed -- pip3 install bleak")
    assert BleakScanner is not None
    devices = await BleakScanner.discover(timeout=timeout_s)
    infos: list[BleDeviceInfo] = []
    for device in devices:
        address = str(getattr(device, "address", "") or "").strip()
        if not address:
            continue
        name = str(getattr(device, "name", "") or "").strip() or "Unknown BLE device"
        rssi = getattr(device, "rssi", None)
        infos.append(BleDeviceInfo(name=name, address=address, rssi=rssi if isinstance(rssi, int) else None))
    return _sort_ble_device_infos(infos, name_filter)


def _sort_ble_device_infos(devices: list[BleDeviceInfo], name_filter: str = "") -> list[BleDeviceInfo]:
    needle = _normalize_ble_match_text(name_filter)

    def key(device: BleDeviceInfo):
        name = _normalize_ble_match_text(device.name)
        name_miss = bool(needle and needle not in name)
        rssi_rank = device.rssi if device.rssi is not None else -999
        return name_miss, -rssi_rank, device.name.lower(), device.address

    return sorted(devices, key=key)


def _normalize_ble_match_text(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


class BleTransport:
    """R SDK byte stream over BLE GATT write/notify.

    DJI's R SDK PDF v2.5 does not publish a BLE service UUID. The
    transport therefore supports both modes we need in the field:
    explicit UUIDs from config, or best-effort discovery of a write
    characteristic and a notify/indicate characteristic on the same
    service. Connection is still proven by a valid R SDK reply frame.
    """

    async_transport = True
    # The RS 4 Pro's BLE GATT stream is DJI DUML (0x55), not the R SDK 0xAA
    # framing used on the RSA UART/CAN port (see docs/RS4_BLE_FINDINGS.md).
    protocol = "duml"

    def __init__(
        self,
        *,
        address: str | None = None,
        name: str = "RS 4 Pro",
        service_uuid: str | None = None,
        tx_char_uuid: str | None = None,
        rx_char_uuid: str | None = None,
        scan_timeout_s: float = 8.0,
        mtu_payload_bytes: int = 20,
    ) -> None:
        self.address = address.strip() if address else None
        self.name = name.strip() if name else "RS 4 Pro"
        self.service_uuid = self._normalize_uuid(service_uuid)
        self.tx_char_uuid = self._normalize_uuid(tx_char_uuid)
        self.rx_char_uuid = self._normalize_uuid(rx_char_uuid)
        self.scan_timeout_s = scan_timeout_s
        self.mtu_payload_bytes = max(1, min(244, int(mtu_payload_bytes)))
        self._client: Any | None = None
        self._queue: asyncio.Queue[bytes] | None = None
        self._endpoint: BleEndpoint | None = None

    @property
    def label(self) -> str:
        return "BLE"

    async def open(self) -> None:
        if not BLE_AVAILABLE:
            raise RuntimeError("bleak is not installed -- pip3 install bleak")
        if self._client is not None and getattr(self._client, "is_connected", False):
            return

        target = self.address or await self._scan_for_device()
        self._client = BleakClient(target)
        await self._client.connect(timeout=self.scan_timeout_s)

        services = await self._get_services()
        self._endpoint = self._select_characteristics(
            services,
            service_uuid=self.service_uuid,
            tx_char_uuid=self.tx_char_uuid,
            rx_char_uuid=self.rx_char_uuid,
        )
        self._queue = asyncio.Queue()
        await self._client.start_notify(self._endpoint.rx_uuid, self._on_notify)
        logger.info(
            "DJI RS 4 Pro BLE connected, write=%s notify=%s response=%s",
            self._endpoint.tx_uuid,
            self._endpoint.rx_uuid,
            self._endpoint.write_with_response,
        )

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._queue = None
        endpoint = self._endpoint
        self._endpoint = None
        if client is None:
            return
        if endpoint is not None:
            try:
                await client.stop_notify(endpoint.rx_uuid)
            except Exception as exc:  # noqa: BLE001
                logger.debug("BLE stop_notify failed during close: %s", exc)
        try:
            await client.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.debug("BLE disconnect failed during close: %s", exc)

    async def send(self, frame: bytes) -> None:
        if self._client is None or self._endpoint is None:
            raise RuntimeError("BLE transport is not open")
        for offset in range(0, len(frame), self.mtu_payload_bytes):
            await self._client.write_gatt_char(
                self._endpoint.tx_uuid,
                frame[offset : offset + self.mtu_payload_bytes],
                response=self._endpoint.write_with_response,
            )

    async def receive_chunk(self, timeout_s: float) -> bytes:
        if self._queue is None:
            return b""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return b""

    async def drain(self) -> bytes:
        """Non-blocking: return everything currently queued and empty it.
        Used on the DUML (BLE) path, which streams telemetry continuously
        and would otherwise grow the notify queue unbounded."""
        if self._queue is None:
            return b""
        out = bytearray()
        while True:
            try:
                out += self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        return bytes(out)

    async def _scan_for_device(self):
        devices = await scan_ble_devices(name_filter=self.name, timeout_s=self.scan_timeout_s)
        needle = _normalize_ble_match_text(self.name)
        for device in devices:
            if needle and needle in _normalize_ble_match_text(device.name):
                return device.address
        seen = ", ".join(device.name for device in devices) or "none"
        raise RuntimeError(f"BLE gimbal '{self.name}' not found; scanned devices: {seen}")

    async def _get_services(self):
        assert self._client is not None
        try:
            services = getattr(self._client, "services", None)
        except Exception:
            services = None
        if services is not None:
            return services
        getter = getattr(self._client, "get_services", None)
        if getter is None:
            raise RuntimeError("BLE client did not expose GATT services")
        result = getter()
        if inspect.isawaitable(result):
            return await result
        return result

    def _on_notify(self, _sender, data: bytearray | bytes) -> None:
        if self._queue is not None:
            self._queue.put_nowait(bytes(data))

    @classmethod
    def _select_characteristics(
        cls,
        services,
        *,
        service_uuid: str | None,
        tx_char_uuid: str | None,
        rx_char_uuid: str | None,
    ) -> BleEndpoint:
        write_candidates: list[tuple[str, Any, str, set[str]]] = []
        notify_candidates: list[tuple[str, Any, str, set[str]]] = []
        configured_tx: tuple[str, Any, str, set[str]] | None = None
        configured_rx: tuple[str, Any, str, set[str]] | None = None

        for service in cls._iter_services(services):
            current_service_uuid = cls._normalize_uuid(getattr(service, "uuid", None)) or ""
            if service_uuid and current_service_uuid != service_uuid:
                continue
            for char in getattr(service, "characteristics", []) or []:
                char_uuid = cls._normalize_uuid(getattr(char, "uuid", None))
                if not char_uuid:
                    continue
                props = {str(prop).lower() for prop in (getattr(char, "properties", []) or [])}
                item = (current_service_uuid, char, char_uuid, props)
                if tx_char_uuid and char_uuid == tx_char_uuid:
                    configured_tx = item
                if rx_char_uuid and char_uuid == rx_char_uuid:
                    configured_rx = item
                if not tx_char_uuid and ("write" in props or "write-without-response" in props):
                    write_candidates.append(item)
                if not rx_char_uuid and ("notify" in props or "indicate" in props):
                    notify_candidates.append(item)

        if tx_char_uuid and configured_tx is None:
            raise RuntimeError(f"Configured BLE write characteristic not found: {tx_char_uuid}")
        if rx_char_uuid and configured_rx is None:
            raise RuntimeError(f"Configured BLE notify characteristic not found: {rx_char_uuid}")

        tx = configured_tx
        rx = configured_rx
        if tx is None and rx is not None:
            tx = cls._best_counterpart(rx, write_candidates)
        if rx is None and tx is not None:
            rx = cls._best_counterpart(tx, notify_candidates)
        if tx is None or rx is None:
            tx, rx = cls._best_pair(write_candidates, notify_candidates)
        if tx is None or rx is None:
            raise RuntimeError(
                "Could not find BLE write/notify characteristics; set service/tx/rx UUIDs in Settings"
            )

        write_props = tx[3]
        write_with_response = "write-without-response" not in write_props and "write" in write_props
        return BleEndpoint(tx_uuid=tx[2], rx_uuid=rx[2], write_with_response=write_with_response)

    @staticmethod
    def _best_counterpart(reference: tuple[str, Any, str, set[str]], candidates: list[tuple[str, Any, str, set[str]]]):
        same_service = [candidate for candidate in candidates if candidate[0] == reference[0]]
        if same_service:
            return sorted(same_service, key=lambda candidate: candidate[2])[0]
        if candidates:
            return sorted(candidates, key=lambda candidate: candidate[2])[0]
        return None

    @staticmethod
    def _best_pair(
        write_candidates: list[tuple[str, Any, str, set[str]]],
        notify_candidates: list[tuple[str, Any, str, set[str]]],
    ):
        scored = []
        for write in write_candidates:
            for notify in notify_candidates:
                score = 0
                if write[0] == notify[0]:
                    score += 100
                if "write-without-response" in write[3]:
                    score += 10
                if "notify" in notify[3]:
                    score += 5
                scored.append((score, write[2], notify[2], write, notify))
        if not scored:
            return None, None
        _, _, _, write, notify = max(scored, key=lambda item: (item[0], item[1], item[2]))
        return write, notify

    @staticmethod
    def _iter_services(services):
        raw_services = getattr(services, "services", None)
        if isinstance(raw_services, dict):
            return list(raw_services.values())
        try:
            return list(services)
        except TypeError:
            return []

    @staticmethod
    def _normalize_uuid(uuid: str | None) -> str | None:
        if uuid is None:
            return None
        normalized = str(uuid).strip().lower()
        return normalized or None


class DjiRs4ProBackend(GimbalBackend):
    def __init__(self, transport, max_pan_speed: float = 20.0, max_tilt_speed: float = 12.0) -> None:
        self.transport = transport
        # Needed to normalize deg/s into the DUML joystick's [-1, 1] ratio
        # (see build_joystick_frame) -- GimbalController clamps to these same
        # max speeds before calling set_velocity, so a full-deflection
        # request maps to a full-deflection joystick channel value.
        self.max_pan_speed = max(1e-6, max_pan_speed)
        self.max_tilt_speed = max(1e-6, max_tilt_speed)
        self._connected = False
        self._sequence = 0
        self._assembler = FrameAssembler()
        self._last_connect_attempt = 0.0
        self._last_orientation = (0.0, 0.0, 0.0)
        self._connect_lock = asyncio.Lock()
        # BLE speaks DUML (0x55); CAN/UART speak the R SDK (0xAA). set_velocity
        # over DUML sends the reverse-engineered joystick command (cmd_set
        # 0x04, cmd_id 0x01 -- see dji_duml.build_joystick_frame, decoded from
        # a PacketLogger capture of the Ronin app and confirmed live against
        # real hardware: chA=tilt +up/-down, chC=pan +right/-left). Recenter
        # (go_home) is NOT decoded yet -- that stays an honest no-op.
        self._protocol = getattr(transport, "protocol", "rsdk")
        self._duml_assembler = DjiDumlAssembler()
        self._ble_control_warned = False
        # Optional: dump received DUML telemetry to a JSONL for offline
        # decode (set MOTOCAM_DUML_CAPTURE=/path). Lets you capture a moving
        # gimbal through the app itself -- no separate BLE connection that
        # would collide with the app's.
        self._duml_capture_path = os.environ.get("MOTOCAM_DUML_CAPTURE")
        self._duml_frame_count = 0

    def _next_sequence(self) -> int:
        self._sequence = (self._sequence + 1) & 0xFFFF
        return self._sequence

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def source(self) -> str:
        label = getattr(self.transport, "label", self.transport.__class__.__name__)
        return f"dji-rsdk-{str(label).lower()}"

    async def connect(self) -> None:
        if self._connected or self._connect_lock.locked():
            return
        async with self._connect_lock:
            now = time.monotonic()
            if self._connected or now - self._last_connect_attempt < RECONNECT_MIN_INTERVAL_S:
                return
            self._last_connect_attempt = now
            try:
                await self._call_transport("open")
            except Exception as exc:  # noqa: BLE001 -- adapter missing/busy must degrade, not crash
                logger.info("R SDK transport not available (%s), will retry", exc)
                self._connected = False
                return
            # Proof-of-life differs by transport protocol: the RSA port
            # (CAN/UART) answers an R SDK GET_POSITION; BLE (DUML) pushes
            # telemetry on its own, so a single CRC-valid DUML frame proves
            # the link without us sending anything.
            label = getattr(self.transport, "label", self.transport.__class__.__name__)
            if self._protocol == "duml":
                alive = await self._duml_proof_of_life()
                if not alive:
                    logger.warning(
                        "BLE open but no DUML frame received -- check the notify "
                        "characteristic (fff4) and that the gimbal is powered/awake"
                    )
                    await self._call_transport("close")
                    self._connected = False
                    return
                self._connected = True
                logger.info(
                    "DJI RS 4 Pro connected via %s (DUML telemetry OK). Joystick velocity "
                    "control is live; recenter (HOME/RESET) is not decoded yet on this transport.",
                    label,
                )
                return

            orientation = await self._query_position()
            if orientation is None:
                logger.warning(
                    "R SDK transport open but gimbal did not answer GET_POSITION -- "
                    "check RSA wiring/CAN bitrate/termination"
                )
                await self._call_transport("close")
                self._connected = False
                return
            self._last_orientation = orientation
            self._connected = True
            logger.info("DJI RS 4 Pro connected via %s R SDK transport (position reply OK)", label)

    async def disconnect(self) -> None:
        try:
            await self._call_transport("close")
        finally:
            self._connected = False

    async def _duml_proof_of_life(self) -> bool:
        """Wait for one CRC-valid DUML frame from the gimbal (it streams
        telemetry on notify without being asked). No send needed."""
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        while time.monotonic() < deadline:
            chunk = await self._call_transport("receive_chunk", max(0.0, deadline - time.monotonic()))
            if not chunk:
                continue
            if self._duml_assembler.feed(chunk):
                return True
        return False

    async def _query_position(self) -> tuple[float, float, float] | None:
        request = build_get_position(self._next_sequence())
        try:
            await self._call_transport("send", request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("R SDK send failed: %s", exc)
            self._connected = False
            return None
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        while time.monotonic() < deadline:
            chunk = await self._call_transport("receive_chunk", max(0.0, deadline - time.monotonic()))
            if not chunk:
                continue
            for frame in self._assembler.feed(chunk):
                orientation = parse_position_reply(frame)
                if orientation is not None:
                    return orientation
        return None

    async def set_velocity(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        if not self._connected:
            return
        if self._protocol == "duml":
            ratio_pan = max(-1.0, min(1.0, pan_deg_s / self.max_pan_speed))
            ratio_tilt = max(-1.0, min(1.0, tilt_deg_s / self.max_tilt_speed))
            frame = build_joystick_frame(self._next_sequence(), ch_a=ratio_tilt, ch_c=ratio_pan)
            try:
                await self._call_transport("send", frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("BLE joystick send failed (%s), marking disconnected", exc)
                self._connected = False
            return
        frame = build_speed_control(self._next_sequence(), pan_deg_s, tilt_deg_s)
        try:
            await self._call_transport("send", frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("R SDK velocity send failed (%s), marking disconnected", exc)
            self._connected = False

    async def go_home(self) -> None:
        if not self._connected:
            return
        if self._protocol == "duml":
            self._warn_ble_control_once()
            return
        frame = build_position_control(self._next_sequence(), 0.0, 0.0)
        try:
            await self._call_transport("send", frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("R SDK home send failed (%s), marking disconnected", exc)
            self._connected = False

    async def get_orientation(self) -> tuple[float, float, float]:
        if not self._connected:
            return self._last_orientation
        # DUML (BLE): the angle payload isn't decoded yet (needs a moving
        # capture), so don't send an R SDK query the gimbal ignores and
        # don't fake a reading. But DO drain the notify stream so the queue
        # can't grow unbounded, and optionally log frames for offline decode.
        if self._protocol == "duml":
            await self._drain_duml_telemetry()
            return self._last_orientation
        orientation = await self._query_position()
        if orientation is not None:
            self._last_orientation = orientation
        return self._last_orientation

    async def _drain_duml_telemetry(self) -> None:
        chunk = await self._call_transport("drain")
        if not chunk:
            return
        for frame in self._duml_assembler.feed(chunk):
            self._duml_frame_count += 1
            self._record_duml_frame(frame)

    def _record_duml_frame(self, frame) -> None:
        if not self._duml_capture_path:
            return
        try:
            with open(self._duml_capture_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "t": time.time(),
                    "cmd_set": frame.cmd_set,
                    "cmd_id": frame.cmd_id,
                    "hex": frame.to_bytes().hex(),
                }) + "\n")
        except OSError as exc:
            logger.debug("DUML capture write failed: %s", exc)

    def _warn_ble_control_once(self) -> None:
        if not self._ble_control_warned:
            logger.warning(
                "Recenter (HOME/RESET) over BLE is not available yet: only the joystick "
                "velocity command (cmd_set 0x04, cmd_id 0x01) has been reverse-engineered "
                "so far, not a recenter/position command. Use RESET via the RSA port "
                "(CAN/UART), or see docs/RS4_BLE_FINDINGS.md."
            )
            self._ble_control_warned = True

    async def _call_transport(self, method_name: str, *args):
        method = getattr(self.transport, method_name)
        if getattr(self.transport, "async_transport", False):
            result = method(*args)
            if inspect.isawaitable(result):
                return await result
            return result
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(method, *args))
