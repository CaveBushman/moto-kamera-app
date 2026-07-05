#!/usr/bin/env python3
"""Standalone DJI RS 4 Pro RSA-port probe for UART/CAN.

This is the deployment-side counterpart to scripts/rs4_ble_probe.py.
It uses the actual DJI R SDK backend and sends only GET_POSITION, so it
is safe to run on a mounted rig: no pan/tilt/zoom movement commands are
sent. A successful probe proves the RSA wiring, bitrate/baudrate and
protocol framing before the moto app is allowed to drive the gimbal.

Examples:

    PYTHONPATH=. python scripts/rs4_rsa_probe.py --transport uart --device /dev/ttyAMA0
    PYTHONPATH=. python scripts/rs4_rsa_probe.py --transport can --channel can0
    PYTHONPATH=. python scripts/rs4_rsa_probe.py --list
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from motocam.gimbal.dji_rs4pro import CanTransport, DjiRs4ProBackend, UartTransport


SERIAL_PATTERNS = (
    "/dev/serial/by-id/*",
    "/dev/ttyAMA*",
    "/dev/ttyS*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
    "/dev/cu.*",
)


def serial_candidates() -> list[str]:
    devices: set[str] = set()
    for pattern in SERIAL_PATTERNS:
        devices.update(glob.glob(pattern))
    return sorted(devices)


def can_candidates() -> list[str]:
    sys_net = Path("/sys/class/net")
    if not sys_net.is_dir():
        return []
    channels: list[str] = []
    for item in sys_net.iterdir():
        try:
            # Linux ARPHRD_CAN == 280.
            if (item / "type").read_text().strip() == "280":
                channels.append(item.name)
        except OSError:
            continue
    return sorted(channels)


def print_candidates() -> None:
    print("Serial candidates:")
    serial = serial_candidates()
    if serial:
        for device in serial:
            print(f"  {device}")
    else:
        print("  none")

    print("\nCAN candidates:")
    can = can_candidates()
    if can:
        for channel in can:
            print(f"  {channel}")
    else:
        print("  none")


def build_backend(args: argparse.Namespace) -> DjiRs4ProBackend:
    if args.transport == "uart":
        transport = UartTransport(device=args.device, baudrate=args.baudrate)
    else:
        transport = CanTransport(channel=args.channel, bitrate=args.bitrate)
    return DjiRs4ProBackend(transport)


async def probe(args: argparse.Namespace) -> int:
    backend = build_backend(args)
    label = f"UART {args.device} @ {args.baudrate}" if args.transport == "uart" else f"CAN {args.channel} @ {args.bitrate}"
    print(f"== RS 4 Pro RSA probe ==\ntransport: {label}\n")
    try:
        await backend.connect()
        if not backend.connected:
            print("Verdict: no valid R SDK GET_POSITION reply.")
            print("Check RSA cable orientation/pinout, common ground, baudrate/CAN bitrate and CAN termination.")
            return 1
        pan, tilt, roll = await backend.get_orientation()
        print("Verdict: R SDK link OK.")
        print(f"Orientation: pan={pan:.1f} deg  tilt={tilt:.1f} deg  roll={roll:.1f} deg")
        return 0
    except Exception as exc:  # noqa: BLE001 - field diagnostics should print the actual failure
        print(f"Verdict: transport failed before R SDK reply: {exc}")
        return 2
    finally:
        await backend.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List likely UART/CAN devices and exit")
    parser.add_argument("--transport", choices=("uart", "can"), default="uart")
    parser.add_argument("--device", default="/dev/ttyAMA0", help="UART device for the RSA port")
    parser.add_argument("--baudrate", type=int, default=115200, help="UART baudrate")
    parser.add_argument("--channel", default="can0", help="SocketCAN channel")
    parser.add_argument("--bitrate", type=int, default=1_000_000, help="CAN bitrate")
    args = parser.parse_args()

    if args.list:
        print_candidates()
        return 0
    return asyncio.run(probe(args))


if __name__ == "__main__":
    sys.exit(main())
