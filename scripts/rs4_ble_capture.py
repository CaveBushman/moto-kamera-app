#!/usr/bin/env python3
"""Capture DJI RS 4 Pro BLE vendor notifications for reverse engineering.

This records the observed DJI BLE stream, not movement control. It
subscribes to notify characteristics, writes JSONL events to
captures/rs4_ble_*.jsonl, and frames 0x55 vendor packets using the
length byte. Use it to compare captures while pressing hardware buttons,
moving the gimbal by hand, changing mode, or repeating a safe R SDK
GET_POSITION probe.

Examples:

    PYTHONPATH=. python scripts/rs4_ble_capture.py --duration 20
    PYTHONPATH=. python scripts/rs4_ble_capture.py --duration 10 --send-rsdk-probe
    PYTHONPATH=. python scripts/rs4_ble_analyze.py captures/rs4_ble_*.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from bleak import BleakClient
except ImportError:  # pragma: no cover - depends on deployment venv
    BleakClient = None  # type: ignore[assignment]

from motocam.gimbal.dji_ble_vendor import DjiBleVendorAssembler
from motocam.gimbal.dji_rs4pro import BleDeviceInfo, scan_ble_devices
from motocam.gimbal.rsdk_protocol import build_get_position


@dataclass(frozen=True)
class GattChar:
    service_uuid: str
    uuid: str
    properties: frozenset[str]


def normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def short_uuid(uuid: str) -> str:
    normalized = uuid.lower()
    if normalized.startswith("0000") and normalized.endswith("-0000-1000-8000-00805f9b34fb"):
        return normalized[4:8]
    return normalized


def event_time(start: float) -> float:
    return round(time.monotonic() - start, 6)


def write_event(handle, start: float, event: str, **payload: object) -> None:
    row = {"t": event_time(start), "event": event}
    row.update(payload)
    handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    handle.flush()


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


def notify_chars(chars: list[GattChar], requested: str) -> list[GattChar]:
    candidates = [char for char in chars if "notify" in char.properties or "indicate" in char.properties]
    if requested == "all":
        return candidates
    wanted = requested.lower()
    return [char for char in candidates if short_uuid(char.uuid) == wanted or char.uuid.lower() == wanted]


def first_write_char(chars: list[GattChar]) -> GattChar | None:
    write = [char for char in chars if "write" in char.properties or "write-without-response" in char.properties]
    return (
        sorted(
            write,
            key=lambda char: (
                "notify" in char.properties or "indicate" in char.properties,
                "write" not in char.properties,
                "write-without-response" not in char.properties,
                char.uuid,
            ),
        )[0]
        if write
        else None
    )


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO / "captures" / f"rs4_ble_{stamp}.jsonl"


async def capture(args: argparse.Namespace) -> int:
    if BleakClient is None:
        print("bleak is not installed in this interpreter")
        return 2

    device = await select_device(args.name, args.address, args.timeout)
    if device is None:
        print("\nNo matching RS 4 Pro BLE device found.")
        return 1
    print(f"\nTarget: {device.label}")

    out_path = Path(args.output) if args.output else default_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    assemblers: dict[str, DjiBleVendorAssembler] = {}
    notification_count = 0
    frame_count = 0

    with out_path.open("w", encoding="utf-8") as handle:
        write_event(
            handle,
            start,
            "meta",
            tool="rs4_ble_capture",
            device={"name": device.name, "address": device.address, "rssi": device.rssi},
            args={
                "duration": args.duration,
                "notify": args.notify,
                "send_rsdk_probe": args.send_rsdk_probe,
                "probe_interval": args.probe_interval,
            },
        )

        async with BleakClient(device.address, timeout=args.timeout) as client:
            services = getattr(client, "services", None)
            if services is None:
                getter = getattr(client, "get_services", None)
                services = await getter() if getter is not None else []
            chars = iter_gatt_chars(services)
            write_event(
                handle,
                start,
                "gatt",
                chars=[
                    {
                        "service": char.service_uuid,
                        "uuid": char.uuid,
                        "short_uuid": short_uuid(char.uuid),
                        "properties": sorted(char.properties),
                    }
                    for char in chars
                ],
            )
            selected_notify = notify_chars(chars, args.notify)
            if not selected_notify:
                print(f"No notify characteristic matched '{args.notify}'.")
                return 1

            def make_callback(uuid: str):
                assembler = assemblers.setdefault(uuid, DjiBleVendorAssembler())

                def on_notify(_sender, data: bytearray | bytes) -> None:
                    nonlocal notification_count, frame_count
                    chunk = bytes(data)
                    notification_count += 1
                    write_event(
                        handle,
                        start,
                        "notification",
                        char=uuid,
                        short_uuid=short_uuid(uuid),
                        len=len(chunk),
                        hex=chunk.hex(),
                    )
                    for frame in assembler.feed(chunk):
                        frame_count += 1
                        write_event(
                            handle,
                            start,
                            "vendor_frame",
                            char=uuid,
                            short_uuid=short_uuid(uuid),
                            **frame.to_capture_dict(),
                        )

                return on_notify

            active_notify: list[GattChar] = []
            for char in selected_notify:
                try:
                    await client.start_notify(char.uuid, make_callback(char.uuid))
                except Exception as exc:  # noqa: BLE001 - diagnostics need field errors
                    write_event(handle, start, "subscribe_error", char=char.uuid, short_uuid=short_uuid(char.uuid), error=str(exc))
                    print(f"Subscribe failed on {short_uuid(char.uuid)}: {exc}")
                    continue
                active_notify.append(char)
                print(f"Subscribed {short_uuid(char.uuid)}")

            if not active_notify:
                print("No notify subscription succeeded.")
                return 1

            probe_char = first_write_char(chars)
            next_probe = time.monotonic()
            deadline = time.monotonic() + args.duration
            print(f"Capturing {args.duration:.1f}s -> {out_path}")
            try:
                while time.monotonic() < deadline:
                    if args.send_rsdk_probe and probe_char is not None and time.monotonic() >= next_probe:
                        frame = build_get_position(1)
                        for offset in range(0, len(frame), 20):
                            await client.write_gatt_char(
                                probe_char.uuid,
                                frame[offset : offset + 20],
                                response="write-without-response" not in probe_char.properties and "write" in probe_char.properties,
                            )
                        write_event(handle, start, "safe_rsdk_get_position_probe", char=probe_char.uuid, short_uuid=short_uuid(probe_char.uuid))
                        next_probe = time.monotonic() + args.probe_interval
                    await asyncio.sleep(0.05)
            except KeyboardInterrupt:
                write_event(handle, start, "interrupted")
            finally:
                for char in active_notify:
                    try:
                        await client.stop_notify(char.uuid)
                    except Exception:
                        pass

        write_event(handle, start, "summary", notifications=notification_count, vendor_frames=frame_count)

    print(f"Done: {notification_count} notifications, {frame_count} vendor frames")
    print(f"Saved: {out_path}")
    print(f"Analyze: PYTHONPATH=. python scripts/rs4_ble_analyze.py {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="RS 4 Pro", help="BLE name filter")
    parser.add_argument("--address", default=None, help="Exact BLE address/UUID")
    parser.add_argument("--timeout", type=float, default=8.0, help="BLE scan/connect timeout in seconds")
    parser.add_argument("--duration", type=float, default=20.0, help="Capture duration in seconds")
    parser.add_argument("--notify", default="fff4", help="'fff4', full UUID, or 'all'")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--send-rsdk-probe", action="store_true", help="Periodically send safe GET_POSITION probes")
    parser.add_argument("--probe-interval", type=float, default=1.0, help="Seconds between safe GET_POSITION probes")
    return asyncio.run(capture(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
