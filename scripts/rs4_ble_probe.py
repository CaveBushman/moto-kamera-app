#!/usr/bin/env python3
"""Standalone DJI RS 4 Pro BLE probe.

This is a safe hardware diagnostic: it never sends movement commands.
It scans for the RS 4 Pro, prints its GATT profile, then sends only the
R SDK GET_POSITION query across every plausible write/notify pair. A
valid R SDK answer starts with 0xAA and parses into pan/tilt/roll. The
DJI app BLE profile often emits 0x55 vendor frames instead; that proves
BLE is alive, but not raw R SDK control.

Run with the same venv as the moto app:

    PYTHONPATH=. python scripts/rs4_ble_probe.py
    PYTHONPATH=. python scripts/rs4_ble_probe.py --address <BLE-UUID>
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from bleak import BleakClient
except ImportError:  # pragma: no cover - depends on deployment venv
    BleakClient = None  # type: ignore[assignment]

from motocam.gimbal.dji_rs4pro import BleDeviceInfo, scan_ble_devices
from motocam.gimbal.rsdk_protocol import FrameAssembler, build_get_position, parse_position_reply


@dataclass(frozen=True)
class GattChar:
    service_uuid: str
    uuid: str
    properties: frozenset[str]


@dataclass(frozen=True)
class ProbePair:
    tx_uuid: str
    rx_uuid: str
    write_with_response: bool


def normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def short_uuid(uuid: str) -> str:
    normalized = uuid.lower()
    if normalized.startswith("0000") and normalized.endswith("-0000-1000-8000-00805f9b34fb"):
        return normalized[4:8]
    return normalized


async def select_device(name: str, address: str | None, timeout_s: float) -> BleDeviceInfo | None:
    devices = await scan_ble_devices(name_filter=name, timeout_s=timeout_s)
    print(f"BLE scan: {len(devices)} visible devices")
    for device in devices[:12]:
        marker = "*" if address and device.address == address else " "
        print(f" {marker} {device.label}")
    if address:
        return next((device for device in devices if device.address == address), BleDeviceInfo(name, address))
    needle = normalize_name(name)
    return next((device for device in devices if needle and needle in normalize_name(device.name)), None)


def iter_gatt_chars(services: Any) -> list[GattChar]:
    chars: list[GattChar] = []
    for service in services:
        service_uuid = str(getattr(service, "uuid", "") or "")
        for char in getattr(service, "characteristics", []) or []:
            uuid = str(getattr(char, "uuid", "") or "")
            props = frozenset(str(prop).lower() for prop in (getattr(char, "properties", []) or []))
            if uuid:
                chars.append(GattChar(service_uuid=service_uuid, uuid=uuid, properties=props))
    return chars


def build_probe_pairs(chars: list[GattChar]) -> list[ProbePair]:
    write_chars = [char for char in chars if "write" in char.properties or "write-without-response" in char.properties]
    notify_chars = [char for char in chars if "notify" in char.properties or "indicate" in char.properties]
    scored: list[tuple[int, str, str, ProbePair]] = []
    for tx in write_chars:
        for rx in notify_chars:
            score = 0
            if tx.service_uuid == rx.service_uuid:
                score += 100
            if tx.uuid == rx.uuid:
                score += 20
            if "write-without-response" in tx.properties:
                score += 10
            if "notify" in rx.properties:
                score += 5
            scored.append(
                (
                    score,
                    tx.uuid,
                    rx.uuid,
                    ProbePair(
                        tx_uuid=tx.uuid,
                        rx_uuid=rx.uuid,
                        write_with_response="write-without-response" not in tx.properties and "write" in tx.properties,
                    ),
                )
            )
    pairs: list[ProbePair] = []
    seen: set[tuple[str, str, bool]] = set()
    for _score, _tx, _rx, pair in sorted(scored, reverse=True):
        key = (pair.tx_uuid, pair.rx_uuid, pair.write_with_response)
        if key not in seen:
            seen.add(key)
            pairs.append(pair)
    return pairs


async def read_services(address: str, timeout_s: float) -> list[GattChar]:
    assert BleakClient is not None
    async with BleakClient(address, timeout=timeout_s) as client:
        services = getattr(client, "services", None)
        if services is None:
            getter = getattr(client, "get_services", None)
            services = await getter() if getter is not None else []
        chars = iter_gatt_chars(services)
    print("\nGATT profile:")
    current_service = None
    for char in chars:
        if char.service_uuid != current_service:
            current_service = char.service_uuid
            print(f"  service {short_uuid(current_service)}")
        print(f"    char {short_uuid(char.uuid):<8} props={','.join(sorted(char.properties))}")
    return chars


async def probe_pair(address: str, pair: ProbePair, timeout_s: float, listen_s: float) -> tuple[bool, list[bytes]]:
    assert BleakClient is not None
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    assembler = FrameAssembler()

    def on_notify(_sender, data: bytearray | bytes) -> None:
        queue.put_nowait(bytes(data))

    async with BleakClient(address, timeout=timeout_s) as client:
        await client.start_notify(pair.rx_uuid, on_notify)
        try:
            frame = build_get_position(1)
            for offset in range(0, len(frame), 20):
                await client.write_gatt_char(
                    pair.tx_uuid,
                    frame[offset : offset + 20],
                    response=pair.write_with_response,
                )
            deadline = asyncio.get_running_loop().time() + listen_s
            chunks: list[bytes] = []
            while asyncio.get_running_loop().time() < deadline:
                try:
                    chunk = await asyncio.wait_for(
                        queue.get(),
                        timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                    )
                except asyncio.TimeoutError:
                    break
                chunks.append(chunk)
                for parsed in assembler.feed(chunk):
                    orientation = parse_position_reply(parsed)
                    if orientation is not None:
                        print(
                            "    VALID R SDK position reply: "
                            f"pan={orientation[0]:.1f} tilt={orientation[1]:.1f} roll={orientation[2]:.1f}"
                        )
                        return True, chunks
            return False, chunks
        finally:
            try:
                await client.stop_notify(pair.rx_uuid)
            except Exception:
                pass


async def main_async(args: argparse.Namespace) -> int:
    if BleakClient is None:
        print("bleak is not installed in this interpreter")
        return 2

    print(f"== RS 4 Pro BLE probe ==\nname filter: {args.name}\n")
    device = await select_device(args.name, args.address, args.timeout)
    if device is None:
        print("\nNo matching RS 4 Pro BLE device found.")
        return 1
    print(f"\nTarget: {device.label}")

    chars = await read_services(device.address, args.timeout)
    pairs = build_probe_pairs(chars)
    if not pairs:
        print("\nNo write/notify GATT pair found.")
        return 1

    print("\nR SDK GET_POSITION probe (safe, no movement):")
    any_notifications = False
    for pair in pairs[: args.max_pairs]:
        print(
            f"  tx={short_uuid(pair.tx_uuid):<8} rx={short_uuid(pair.rx_uuid):<8} "
            f"response={pair.write_with_response}"
        )
        try:
            ok, chunks = await probe_pair(device.address, pair, args.timeout, args.listen)
        except Exception as exc:  # noqa: BLE001 - diagnostics must print the actual field failure
            print(f"    ERROR {exc}")
            continue
        any_notifications = any_notifications or bool(chunks)
        if ok:
            print("\nVerdict: BLE raw R SDK works for GET_POSITION.")
            return 0
        if chunks:
            sample = " ".join(chunk[:12].hex() for chunk in chunks[:3])
            first = chunks[0][0]
            label = "DJI vendor BLE frames (0x55), not R SDK 0xAA" if first == 0x55 else f"non-RSDK first byte 0x{first:02x}"
            print(f"    notifications={len(chunks)} sample={sample}")
            print(f"    {label}")
        else:
            print("    no notifications")

    print("\nVerdict:")
    if any_notifications:
        print("  BLE profile is alive, but it did not return a raw R SDK 0xAA position reply.")
        print("  Use RSA UART/CAN for deterministic R SDK gimbal control unless DJI's BLE vendor protocol is decoded.")
    else:
        print("  No BLE notifications came back after GET_POSITION.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="RS 4 Pro", help="BLE name filter")
    parser.add_argument("--address", default=None, help="Exact BLE address/UUID")
    parser.add_argument("--timeout", type=float, default=8.0, help="BLE scan/connect timeout in seconds")
    parser.add_argument("--listen", type=float, default=2.0, help="Seconds to listen after each GET_POSITION")
    parser.add_argument("--max-pairs", type=int, default=8, help="Maximum GATT write/notify pairs to test")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
