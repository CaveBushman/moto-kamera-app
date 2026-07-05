#!/usr/bin/env python3
"""Parse an Apple PacketLogger (.pklg) capture and extract ATT Write
Commands to the RS 4 Pro's BLE write characteristic, decoded as DUML
frames -- the write-side capture needed to reverse-engineer gimbal
control commands (see docs/RS4_BLE_FINDINGS.md).

.pklg record format (as documented in Wireshark's wiretap/packetlogger.c
and used by Apple's PacketLogger / BTStack's HCI_DUMP_PACKETLOGGER):
    uint32 length   (big-endian; length of this record AFTER this field)
    uint32 ts_sec    (big-endian)
    uint32 ts_usec   (big-endian)
    uint8  type      (0x00 HCI cmd, 0x01 HCI event, 0x02 ACL sent,
                       0x03 ACL received, ...)
    (length - 9) bytes of raw HCI packet data

We don't need a full HCI/L2CAP/ATT stack here: DUML frames always start
with 0x55 and self-describe their length + CRC (see dji_duml.py), so we
scan every ACL payload for a valid 0x55...CRC16 frame rather than fully
parsing L2CAP/ATT framing.

Usage:
    python scripts/pklg_parse.py capture.pklg
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motocam.gimbal.dji_duml import DjiDumlFrame  # noqa: E402

TYPE_NAMES = {
    0x00: "HCI CMD", 0x01: "HCI EVENT",
    0x02: "ACL SENT", 0x03: "ACL RECV",
    0x04: "SCO SENT", 0x05: "SCO RECV",
}


def iter_records(path: Path):
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    n = len(data)
    while offset + 9 <= n:
        # Empirically verified against a real .pklg (all fields little-endian,
        # unlike some documented descriptions of this format): the 4-byte
        # length does NOT include itself, matches known ASCII note strings
        # ("Product: iPhone14,2" etc.) byte-for-byte.
        length = struct.unpack_from("<I", data, offset)[0]
        record_end = offset + 4 + length
        if length < 9 or record_end > n:
            break  # malformed/truncated trailing record
        ts_sec, ts_usec, rtype = struct.unpack_from("<IIB", data, offset + 4)
        payload = data[offset + 13 : record_end]
        yield (ts_sec + ts_usec / 1e6, rtype, payload)
        offset = record_end


def find_duml_frames(payload: bytes):
    """Scan raw bytes for a 0x55-framed, CRC-valid DUML frame anywhere in
    this ACL payload (skips L2CAP/ATT opcode/handle bytes we don't parse)."""
    frames = []
    i = 0
    n = len(payload)
    while i < n:
        if payload[i] != 0x55:
            i += 1
            continue
        if i + 3 > n:
            break
        length = payload[i + 1] | ((payload[i + 2] & 0x03) << 8)
        if length < 13 or i + length > n:
            i += 1
            continue
        candidate = payload[i : i + length]
        try:
            frames.append(DjiDumlFrame.parse(candidate))
            i += length
        except ValueError:
            i += 1
    return frames


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} capture.pklg")
        return 1
    path = Path(sys.argv[1])

    acl_sent = acl_recv = 0
    duml_sent = []
    duml_recv = []
    for ts, rtype, payload in iter_records(path):
        if rtype == 0x02:  # ACL sent = phone -> gimbal (our write commands)
            acl_sent += 1
            for frame in find_duml_frames(payload):
                duml_sent.append((ts, frame))
        elif rtype == 0x03:  # ACL received = gimbal -> phone (telemetry)
            acl_recv += 1
            for frame in find_duml_frames(payload):
                duml_recv.append((ts, frame))

    print(f"records: ACL sent={acl_sent} ACL received={acl_recv}")
    print(f"DUML frames: sent(app->gimbal)={len(duml_sent)}  received(gimbal->app)={len(duml_recv)}")

    print("\n=== app -> gimbal (write) DUML frames, by (cmd_set, cmd_id) ===")
    from collections import Counter, defaultdict

    groups = defaultdict(list)
    for ts, f in duml_sent:
        groups[(f.cmd_set, f.cmd_id)].append((ts, f))
    for (cs, ci), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  cmd_set=0x{cs:02x} cmd_id=0x{ci:02x}  n={len(items)}")

    if duml_sent:
        print("\n=== timeline of sent frames (t, cmd_set, cmd_id, payload hex) ===")
        t0 = duml_sent[0][0]
        for ts, f in duml_sent:
            print(f"  {ts - t0:7.3f}s  set=0x{f.cmd_set:02x} id=0x{f.cmd_id:02x}  payload={f.payload.hex()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
