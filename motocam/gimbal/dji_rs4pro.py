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
probes are throttled the same way as the Blackmagic REST camera backend's.

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
from motocam.gimbal.dji_duml import DjiDumlAssembler, build_joystick_frame, build_recenter_frame
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
BLE_NOTIFY_STALE_S = 10.0
BLE_DEGRADED_MTU_PAYLOAD_BYTES = 20
BLE_FAST_SEND_INTERVAL_S = 0.05
BLE_SLOW_SEND_INTERVAL_S = 0.10
BLE_DEGRADED_SEND_INTERVAL_S = 0.15
# Bounds a single joystick write's wait. A command older than a few hundred
# milliseconds is already harmful for hand control: waiting behind it is the
# "rubber band" lag the rider feels. Timeout drops the stale write so the
# next control tick can send the freshest joystick state instead.
VELOCITY_SEND_TIMEOUT_S = 0.15
BLE_NOTIFY_QUEUE_MAX = 256
SERVICE_DISCOVERY_RETRY_S = 0.05
SERVICE_DISCOVERY_TIMEOUT_S = 1.5

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


# BlueZ (Linux/Raspberry Pi) allows only one discovery/connect session per
# adapter at a time; a second one raises org.bluez.Error.InProgress rather
# than queuing. The app has two independent sources of adapter activity --
# the periodic auto-reconnect (BleTransport.open(), which even a
# direct-by-address connect can trigger BlueZ-side discovery for) and the
# operator's manual "SCAN BLE DEVICES" button in Settings -- so without a
# shared gate they can collide. This lock serializes every adapter
# operation across the whole process.
_BLE_ADAPTER_LOCK = asyncio.Lock()
_BLUEZ_IN_PROGRESS_RETRIES = 2
_BLUEZ_IN_PROGRESS_BACKOFF_S = 0.6


async def _retry_bluez_in_progress(coro_factory):
    """Run an awaitable, retrying a few times if BlueZ reports another
    adapter operation is already in progress (defense-in-depth alongside
    _BLE_ADAPTER_LOCK, e.g. for activity outside our own lock's control)."""
    last_exc: Exception | None = None
    for attempt in range(_BLUEZ_IN_PROGRESS_RETRIES + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 -- match on message, bleak's exception type varies
            if "InProgress" not in str(exc) or attempt == _BLUEZ_IN_PROGRESS_RETRIES:
                raise
            last_exc = exc
            logger.debug("BlueZ adapter busy (attempt %d), retrying: %s", attempt + 1, exc)
            await asyncio.sleep(_BLUEZ_IN_PROGRESS_BACKOFF_S)
    raise last_exc  # pragma: no cover -- loop always returns or raises above


async def _discover_ble_device_infos(timeout_s: float) -> list[BleDeviceInfo]:
    """Raw discovery -> BleDeviceInfo list, no locking/retry. Callers that
    already hold _BLE_ADAPTER_LOCK (e.g. BleTransport._scan_for_device, from
    inside BleTransport.open()'s own locked section) must call this
    directly rather than the public scan_ble_devices() below, or they would
    deadlock re-acquiring the (non-reentrant) lock.

    Devices with no advertised name are dropped here, not just cosmetically
    hidden by the Settings UI: the gimbal always advertises a real name, so
    an unnamed peripheral can never be a match for _scan_for_device's
    name-filter logic either -- keeping it around only clutters the manual
    "SCAN BLE DEVICES" list (unnamed devices routinely outnumber named ones
    several to one in a crowded RF environment)."""
    assert BleakScanner is not None
    devices = await BleakScanner.discover(timeout=timeout_s)
    infos: list[BleDeviceInfo] = []
    for device in devices:
        address = str(getattr(device, "address", "") or "").strip()
        if not address:
            continue
        name = str(getattr(device, "name", "") or "").strip()
        if not name:
            continue
        rssi = getattr(device, "rssi", None)
        infos.append(BleDeviceInfo(name=name, address=address, rssi=rssi if isinstance(rssi, int) else None))
    return infos


async def scan_ble_devices(name_filter: str = "", timeout_s: float = 5.0) -> list[BleDeviceInfo]:
    """Return visible BLE devices, with RS-name matches and strong signals
    first. Public entry point (e.g. the Settings "SCAN BLE DEVICES" button)
    -- takes the shared adapter lock and retries a transient BlueZ
    "InProgress" busy error."""
    if not BLE_AVAILABLE:
        raise RuntimeError("bleak is not installed -- pip3 install bleak")

    async def _discover():
        async with _BLE_ADAPTER_LOCK:
            return await _discover_ble_device_infos(timeout_s)

    infos = await _retry_bluez_in_progress(_discover)
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
        # Floor, not the final word: open() raises this to the ACTUALLY
        # negotiated ATT MTU once connected (commonly 247 on modern
        # BLE, vs. this conservative un-negotiated default of 20). Without
        # that, every joystick frame (22 bytes) was needlessly split into 2
        # GATT writes -- doubling write traffic at the ~20 Hz control-tick
        # rate, which is exactly the kind of load that produces the jerky,
        # multi-second-lagging motion seen live on real hardware.
        self.mtu_payload_bytes = max(1, min(244, int(mtu_payload_bytes)))
        self._client: Any | None = None
        self._queue: asyncio.Queue[bytes] | None = None
        self._endpoint: BleEndpoint | None = None
        self._last_notify_at: float | None = None
        self._notify_loop: asyncio.AbstractEventLoop | None = None

    @property
    def label(self) -> str:
        return "BLE"

    def is_connected(self) -> bool:
        """Used by DjiRs4ProBackend.check_connection() to detect a BLE
        drop between control ticks, independent of DUML telemetry age."""
        return bool(self._client is not None and getattr(self._client, "is_connected", False))

    async def open(self) -> None:
        if not BLE_AVAILABLE:
            raise RuntimeError("bleak is not installed -- pip3 install bleak")
        if self._client is not None and getattr(self._client, "is_connected", False):
            return

        async def _connect() -> Any:
            # Scanning (if no explicit address) and the connect itself both
            # share _BLE_ADAPTER_LOCK with scan_ble_devices() -- on BlueZ a
            # direct-by-address connect can still trigger adapter-side
            # discovery, so it must not overlap with a manual scan either.
            return await self._connect_client_with_retry()

        self._client = await _retry_bluez_in_progress(_connect)
        await self._adopt_negotiated_mtu()

        services = await self._get_services()
        self._endpoint = self._select_characteristics(
            services,
            service_uuid=self.service_uuid,
            tx_char_uuid=self.tx_char_uuid,
            rx_char_uuid=self.rx_char_uuid,
        )
        self._queue = asyncio.Queue(maxsize=BLE_NOTIFY_QUEUE_MAX)
        self._last_notify_at = None
        self._notify_loop = asyncio.get_running_loop()
        await self._client.start_notify(self._endpoint.rx_uuid, self._on_notify)
        logger.info(
            "DJI RS 4 Pro BLE connected, write=%s notify=%s response=%s mtu_payload=%d",
            self._endpoint.tx_uuid,
            self._endpoint.rx_uuid,
            self._endpoint.write_with_response,
            self.mtu_payload_bytes,
        )

    async def _connect_client_with_retry(self) -> Any:
        """Attempt BLE connection, retrying with a fresh scan if the first
        direct-by-address connect fails and the device is still becoming
        visible after an app restart.

        A configured `self.address` can be permanently stale (not just
        "not visible yet"): macOS's CoreBluetooth address for a peripheral
        is a per-machine UUID, not the peripheral's real MAC, so a
        ble_address saved from one machine's config is never valid on
        another -- every single reconnect would otherwise burn a full
        scan_timeout_s on a doomed attempt 0 before falling back. Once a
        scan-fallback connect succeeds, remember that address so the
        *next* reconnect (e.g. after a mid-ride BLE drop) tries it
        directly first instead of repeating the failed guess forever."""
        async with _BLE_ADAPTER_LOCK:
            for attempt in range(2):
                target = self.address if attempt == 0 else await self._scan_for_device(locked=True)
                client = BleakClient(target)
                try:
                    await client.connect(timeout=self.scan_timeout_s)
                    if attempt == 1:
                        self.address = target
                    return client
                except Exception as exc:  # noqa: BLE001
                    if attempt == 1:
                        raise
                    logger.info("BLE connect to %s failed (%s); retrying after a fresh scan", target, exc)
                    await asyncio.sleep(0.25)
        raise RuntimeError("BLE connect target could not be resolved")

    async def _adopt_negotiated_mtu(self) -> None:
        """Raise mtu_payload_bytes to the connection's actual negotiated ATT
        MTU (bleak exposes this post-connect as client.mtu_size) if it's
        bigger than our conservative floor -- never lowers it, so an
        explicit smaller config value or a peripheral without MTU exchange
        support both still work.

        On bleak's CoreBluetooth (macOS) backend mtu_size is populated by
        the time connect() returns. On the BlueZ (Linux/Pi) backend the MTU
        exchange can complete a little *after* connect() returns (it's a
        separate D-Bus property update), so a single immediate read can
        still see the un-negotiated default (23) and silently keep every
        joystick frame split into 2 GATT writes -- exactly the jerky,
        stepped motion seen live on the Pi but not on macOS. Poll briefly
        instead of reading once."""
        for attempt in range(6):  # ~0.6s total, generous for a D-Bus round trip
            mtu_size = getattr(self._client, "mtu_size", None)
            # 23 is BLE's un-negotiated default ATT MTU (protocol-defined,
            # not a guess) -- keep polling while we're still seeing it, only
            # adopt once it's genuinely negotiated above that.
            if isinstance(mtu_size, int) and mtu_size > 23:
                usable = min(244, mtu_size - 3)  # 3 bytes of ATT write-command overhead
                if usable > self.mtu_payload_bytes:
                    logger.info(
                        "BLE MTU negotiated at %d (after %d check%s) -> raising write chunk size to %d bytes",
                        mtu_size, attempt + 1, "" if attempt == 0 else "s", usable,
                    )
                    self.mtu_payload_bytes = usable
                return
            await asyncio.sleep(0.1)
        logger.info(
            "BLE MTU not reported as negotiated after connect (mtu_size=%r) -- "
            "keeping write chunk size at %d bytes",
            getattr(self._client, "mtu_size", None), self.mtu_payload_bytes,
        )

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._queue = None
        self._last_notify_at = None
        self._notify_loop = None
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
            chunk = frame[offset : offset + self.mtu_payload_bytes]
            await self._write_chunk(chunk)

    async def _write_chunk(self, chunk: bytes) -> None:
        assert self._client is not None and self._endpoint is not None
        try:
            await self._client.write_gatt_char(
                self._endpoint.tx_uuid,
                chunk,
                response=self._endpoint.write_with_response,
            )
        except Exception as exc:  # noqa: BLE001 -- bleak raises backend-specific errors here
            if "Service Discovery has not been performed yet" not in str(exc):
                raise
            logger.debug("BLE write hit service-discovery race; waiting and retrying once")
            await self._get_services()
            await self._client.write_gatt_char(
                self._endpoint.tx_uuid,
                chunk,
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

    def notification_age_s(self) -> float | None:
        """Seconds since the last DUML notify frame arrived, or None
        before the first one. Feeds check_connection()'s staleness check."""
        if self._last_notify_at is None:
            return None
        return time.monotonic() - self._last_notify_at

    async def _scan_for_device(self, *, locked: bool = False):
        """`locked=True` means the caller (BleTransport.open()) already
        holds _BLE_ADAPTER_LOCK -- go straight to the no-lock raw discovery
        instead of the public scan_ble_devices(), which would try to
        re-acquire that same (non-reentrant) lock and deadlock."""
        if locked:
            devices = await _discover_ble_device_infos(self.scan_timeout_s)
        else:
            devices = await scan_ble_devices(name_filter=self.name, timeout_s=self.scan_timeout_s)
        needle = _normalize_ble_match_text(self.name)
        for device in devices:
            if needle and needle in _normalize_ble_match_text(device.name):
                return device.address
        seen = ", ".join(device.name for device in devices) or "none"
        raise RuntimeError(f"BLE gimbal '{self.name}' not found; scanned devices: {seen}")

    async def _get_services(self):
        assert self._client is not None
        deadline = time.monotonic() + SERVICE_DISCOVERY_TIMEOUT_S
        last_exc: Exception | None = None
        while True:
            try:
                services = getattr(self._client, "services", None)
            except Exception as exc:  # noqa: BLE001 -- BleakError on discovery not ready
                services = None
                last_exc = exc
            if services is not None:
                return services
            getter = getattr(self._client, "get_services", None)
            if getter is not None:
                result = getter()
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    return result
            if time.monotonic() >= deadline:
                if last_exc is not None:
                    raise RuntimeError(f"BLE service discovery did not complete: {last_exc}") from last_exc
                raise RuntimeError("BLE client did not expose GATT services")
            await asyncio.sleep(SERVICE_DISCOVERY_RETRY_S)

    def _on_notify(self, _sender, data: bytearray | bytes) -> None:
        loop = self._notify_loop
        if loop is None or loop.is_closed():
            return
        payload = bytes(data)
        timestamp = time.monotonic()
        loop.call_soon_threadsafe(self._enqueue_notify, payload, timestamp)

    def _enqueue_notify(self, data: bytes, timestamp: float) -> None:
        self._last_notify_at = timestamp
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(data)

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
    """GimbalBackend implementation for the DJI RS 4 Pro, talking either R
    SDK (0xAA, over CanTransport/UartTransport on the RSA port) or DUML
    (0x55, over BleTransport) depending on `transport.protocol`.

    Beyond simple send/receive, this class also self-regulates BLE
    joystick traffic so a slow or flaky link degrades gracefully instead
    of freezing the whole control loop (see docs/RS4_BLE_FINDINGS.md):
    - `set_velocity` bounds each write with `velocity_send_timeout_s` and
      abandons (doesn't queue) a stalled one -- see `_should_send_velocity_now`.
    - `_adaptive_velocity_interval` backs off the send rate when writes
      are running slow, instead of hammering an already-struggling link.
    - `check_connection` treats a stale DUML notify stream as a warning,
      not an automatic disconnect -- BLE writes can still succeed even if
      telemetry momentarily stops.
    `velocity_stats()` exposes the rolling write/gap/timeout numbers so
    GimbalTelemetry (core/protocol.py) can surface BLE health in the UI
    and to the control room, not just connected/disconnected.
    """

    def __init__(
        self,
        transport,
        max_pan_speed: float = 20.0,
        max_tilt_speed: float = 12.0,
        velocity_send_timeout_s: float = VELOCITY_SEND_TIMEOUT_S,
        notify_stale_s: float = BLE_NOTIFY_STALE_S,
    ) -> None:
        self.transport = transport
        # Needed to normalize deg/s into the DUML joystick's [-1, 1] ratio
        # (see build_joystick_frame) -- GimbalController clamps to these same
        # max speeds before calling set_velocity, so a full-deflection
        # request maps to a full-deflection joystick channel value.
        self.max_pan_speed = max(1e-6, max_pan_speed)
        self.max_tilt_speed = max(1e-6, max_tilt_speed)
        self.velocity_send_timeout_s = max(0.03, float(velocity_send_timeout_s))
        self.notify_stale_s = max(1.0, float(notify_stale_s))
        self._connected = False
        self._sequence = 0
        self._assembler = FrameAssembler()
        self._last_connect_attempt = 0.0
        self._last_orientation = (0.0, 0.0, 0.0)
        self._connect_lock = asyncio.Lock()
        # BLE speaks DUML (0x55); CAN/UART speak the R SDK (0xAA). set_velocity
        # over DUML sends the reverse-engineered joystick command (cmd_set
        # 0x04, cmd_id 0x01 -- see dji_duml.build_joystick_frame) and
        # go_home() sends the recenter command (cmd_set 0x04, cmd_id 0x4c --
        # see dji_duml.build_recenter_frame). Both decoded from PacketLogger
        # captures of the Ronin app and confirmed live against real
        # hardware: chA=tilt +up/-down, chC=pan +right/-left.
        self._protocol = getattr(transport, "protocol", "rsdk")
        self._duml_assembler = DjiDumlAssembler()
        # Optional: dump received DUML telemetry to a JSONL for offline
        # decode (set MOTOCAM_DUML_CAPTURE=/path). Lets you capture a moving
        # gimbal through the app itself -- no separate BLE connection that
        # would collide with the app's.
        self._duml_capture_path = os.environ.get("MOTOCAM_DUML_CAPTURE")
        self._duml_frame_count = 0
        # Joystick send timing diagnostics -- reported to the Pi, control
        # is smooth on macOS. Rather than guess again, log periodic
        # min/avg/max for (a) the actual write_gatt_char round-trip time
        # and (b) the interval between successive set_velocity() calls, so
        # a real Pi run can show whether the *write* is slow (BlueZ/D-Bus,
        # connection interval) or the *calls themselves* arrive irregularly
        # (event loop contention from video/AI/tracking on a much weaker
        # CPU) -- those point at very different fixes.
        self._velocity_send_durations_ms: list[float] = []
        self._velocity_call_gaps_ms: list[float] = []
        self._velocity_timeout_count = 0
        self._last_velocity_stats: dict[str, float | int | None] = {}
        self._last_velocity_call_at: float | None = None
        self._velocity_stats_logged_at = 0.0
        self._notify_stale_logged = False
        self._next_velocity_send_at = 0.0
        self._velocity_throttle_drop_count = 0

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
                self._notify_stale_logged = False
                logger.info(
                    "DJI RS 4 Pro connected via %s (DUML telemetry OK). Joystick velocity "
                    "and recenter (HOME/RESET) control are both live.",
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
            self._notify_stale_logged = False
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
            now = time.monotonic()
            if self._last_velocity_call_at is not None:
                self._velocity_call_gaps_ms.append((now - self._last_velocity_call_at) * 1000.0)
            self._last_velocity_call_at = now
            if not self._should_send_velocity_now(now, pan_deg_s, tilt_deg_s):
                return
            ratio_pan = max(-1.0, min(1.0, pan_deg_s / self.max_pan_speed))
            ratio_tilt = max(-1.0, min(1.0, tilt_deg_s / self.max_tilt_speed))
            frame = build_joystick_frame(self._next_sequence(), ch_a=ratio_tilt, ch_c=ratio_pan)
            send_start = now
            try:
                await asyncio.wait_for(
                    self._call_transport("send", frame), timeout=self.velocity_send_timeout_s
                )
            except asyncio.TimeoutError:
                self._velocity_timeout_count += 1
                self._next_velocity_send_at = time.monotonic() + BLE_DEGRADED_SEND_INTERVAL_S
                logger.warning(
                    "BLE joystick write stalled past %.2fs, abandoning -- next tick's "
                    "freshest command will be sent instead of waiting behind it",
                    self.velocity_send_timeout_s,
                )
                self._velocity_send_durations_ms.append((time.monotonic() - send_start) * 1000.0)
                self._log_velocity_timing_stats(now)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("BLE joystick send failed (%s), marking disconnected", exc)
                self._connected = False
                return
            elapsed_ms = (time.monotonic() - send_start) * 1000.0
            self._velocity_send_durations_ms.append(elapsed_ms)
            self._next_velocity_send_at = time.monotonic() + self._adaptive_velocity_interval(elapsed_ms)
            self._log_velocity_timing_stats(now)
            return
        frame = build_speed_control(self._next_sequence(), pan_deg_s, tilt_deg_s)
        try:
            await self._call_transport("send", frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("R SDK velocity send failed (%s), marking disconnected", exc)
            self._connected = False

    def _log_velocity_timing_stats(self, now: float) -> None:
        """Every ~2s, log min/avg/max for the write round-trip and the
        inter-call gap -- a slow *write* points at BlueZ/D-Bus/connection
        interval; an irregular *gap* points at event-loop contention on the
        Pi (video/AI/tracking competing for a much weaker CPU than macOS)."""
        if now - self._velocity_stats_logged_at < 2.0 or not self._velocity_send_durations_ms:
            return
        durations = self._velocity_send_durations_ms
        gaps = self._velocity_call_gaps_ms
        stats = self._build_velocity_stats(durations, gaps, self._velocity_timeout_count)
        stats.update(self._ble_transport_stats())
        stats["velocity_throttle_drops"] = self._velocity_throttle_drop_count
        self._last_velocity_stats = stats
        logger.info(
            "BLE joystick timing (n=%d): write_ms min/avg/max=%.1f/%.1f/%.1f  "
            "call_gap_ms min/avg/max=%s  timeouts=%d throttle_drop=%d mtu=%s degraded=%s",
            len(durations),
            min(durations), sum(durations) / len(durations), max(durations),
            f"{min(gaps):.1f}/{sum(gaps) / len(gaps):.1f}/{max(gaps):.1f}" if gaps else "n/a",
            self._velocity_timeout_count,
            self._velocity_throttle_drop_count,
            stats.get("ble_mtu_payload_bytes"),
            int(bool(stats.get("ble_degraded"))),
        )
        self._velocity_send_durations_ms.clear()
        self._velocity_call_gaps_ms.clear()
        self._velocity_timeout_count = 0
        self._velocity_throttle_drop_count = 0
        self._velocity_stats_logged_at = now

    def velocity_stats(self) -> dict[str, float | int | None]:
        if self._velocity_send_durations_ms:
            stats = self._build_velocity_stats(
                self._velocity_send_durations_ms,
                self._velocity_call_gaps_ms,
                self._velocity_timeout_count,
            )
            stats.update(self._ble_transport_stats())
            stats["velocity_throttle_drops"] = self._velocity_throttle_drop_count
            return stats
        stats = dict(self._last_velocity_stats)
        stats.update(self._ble_transport_stats())
        return stats

    def _should_send_velocity_now(self, now: float, pan_deg_s: float, tilt_deg_s: float) -> bool:
        """Rate-gate: reject calls that arrive before `_next_velocity_send_at`
        (set by `_adaptive_velocity_interval` after the previous write) so a
        struggling BLE link gets fewer, more likely to land, writes instead
        of every 50ms tick piling on."""
        # Always try to send an explicit stop; throttling a stop command is
        # worse than dropping an intermediate move command.
        if abs(pan_deg_s) < 1e-6 and abs(tilt_deg_s) < 1e-6:
            self._next_velocity_send_at = now
            return True
        if now < self._next_velocity_send_at:
            self._velocity_throttle_drop_count += 1
            return False
        return True

    def _adaptive_velocity_interval(self, elapsed_ms: float) -> float:
        """How long to wait before the next send is allowed, based on how
        long the write we just finished took. Slow writes mean the link is
        struggling, so back off the send rate (BLE_SLOW/DEGRADED_SEND_INTERVAL_S)
        instead of queuing more traffic behind an already-congested
        connection; a healthy fast link gets the full BLE_FAST_SEND_INTERVAL_S
        (~20Hz) rate."""
        if self._ble_is_degraded() or elapsed_ms >= 140.0:
            return BLE_DEGRADED_SEND_INTERVAL_S
        if elapsed_ms >= 80.0:
            return BLE_SLOW_SEND_INTERVAL_S
        return BLE_FAST_SEND_INTERVAL_S

    def _ble_transport_stats(self) -> dict[str, int | bool | None]:
        """MTU/degraded-state snapshot merged into velocity_stats() -- lets
        the UI/control room distinguish "BLE is slow because the MTU never
        negotiated up" from other causes of poor write_ms."""
        if self._protocol != "duml":
            return {"ble_mtu_payload_bytes": None, "ble_degraded": False}
        mtu_payload = getattr(self.transport, "mtu_payload_bytes", None)
        return {
            "ble_mtu_payload_bytes": int(mtu_payload) if isinstance(mtu_payload, int) else None,
            "ble_degraded": self._ble_is_degraded(),
        }

    def _ble_is_degraded(self) -> bool:
        mtu_payload = getattr(self.transport, "mtu_payload_bytes", None)
        return isinstance(mtu_payload, int) and mtu_payload <= BLE_DEGRADED_MTU_PAYLOAD_BYTES

    @staticmethod
    def _build_velocity_stats(
        durations: list[float],
        gaps: list[float],
        timeouts: int,
    ) -> dict[str, float | int | None]:
        return {
            "velocity_write_ms_avg": sum(durations) / len(durations) if durations else None,
            "velocity_write_ms_max": max(durations) if durations else None,
            "velocity_call_gap_ms_avg": sum(gaps) / len(gaps) if gaps else None,
            "velocity_call_gap_ms_max": max(gaps) if gaps else None,
            "velocity_timeouts": timeouts,
        }

    async def go_home(self) -> None:
        if not self._connected:
            return
        if self._protocol == "duml":
            frame = build_recenter_frame(self._next_sequence())
            try:
                await self._call_transport("send", frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("BLE recenter send failed (%s), marking disconnected", exc)
                self._connected = False
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

    async def check_connection(self) -> bool:
        """Periodic health check (called from the UI's async refresh tick,
        not the 20Hz control loop). Distinguishes two failure modes: the
        BLE client itself dropping (`transport.is_connected()` false --
        triggers an actual disconnect + reconnect) versus DUML telemetry
        merely going stale (logged as a warning only, since joystick
        writes can keep working even if notify has hiccuped)."""
        if not self._connected:
            return False
        checker = getattr(self.transport, "is_connected", None)
        if not callable(checker):
            return True
        try:
            alive = bool(await self._call_transport("is_connected"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gimbal transport health check failed (%s), marking disconnected", exc)
            self._connected = False
            return False
        if not alive:
            label = getattr(self.transport, "label", self.transport.__class__.__name__)
            logger.warning("Gimbal %s transport disconnected, reconnect will be attempted", label)
            self._connected = False
            await self._close_after_health_failure()
            return False
        if self._protocol == "duml":
            notify_age = await self._notification_age_s()
            if notify_age is not None and notify_age > self.notify_stale_s:
                if not self._notify_stale_logged:
                    logger.warning(
                        "Gimbal BLE notify stream stale for %.1fs; keeping joystick control alive "
                        "because BLE transport is still connected",
                        notify_age,
                    )
                    self._notify_stale_logged = True
            else:
                self._notify_stale_logged = False
        return alive

    async def _notification_age_s(self) -> float | None:
        age_getter = getattr(self.transport, "notification_age_s", None)
        if not callable(age_getter):
            return None
        try:
            value = await self._call_transport("notification_age_s")
        except Exception as exc:  # noqa: BLE001
            logger.debug("BLE notify age check failed: %s", exc)
            return None
        return float(value) if value is not None else None

    async def _close_after_health_failure(self) -> None:
        try:
            await asyncio.wait_for(self._call_transport("close"), timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("gimbal transport close after health failure failed: %s", exc)

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

    async def _call_transport(self, method_name: str, *args):
        method = getattr(self.transport, method_name)
        if getattr(self.transport, "async_transport", False):
            result = method(*args)
            if inspect.isawaitable(result):
                return await result
            return result
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(method, *args))
