"""DJI DUML transport for the RS 4 Pro BLE link.

Reverse-engineered from a real RS 4 Pro BLE capture (see
docs/RS4_BLE_FINDINGS.md). Unlike the public R SDK used on the RSA
UART/CAN port (0xAA framing, see rsdk_protocol.py), the RS 4 Pro's BLE
GATT stream speaks DJI's DUML protocol:

    off  field
    0    SOF = 0x55
    1    length low 8 bits            (total frame length)
    2    length high 2 bits | ver<<2  (version = 1)
    3    CRC-8 of bytes[0:3]          (init 0x77)
    4    sender id
    5    receiver id
    6-7  sequence number (LE)
    8    cmd type  (0x00 request, 0x20 ack, ...)
    9    cmd set   (0x04 = gimbal)
    10   cmd id
    11.. payload
    -2   CRC-16 of bytes[0:-2] (LE, init 0x3692, reflected poly 0x8408)

Both CRCs validated against all 615 frames of the capture, so frames
built here carry checksums the gimbal accepts.

Control command (joystick) reverse-engineered from a PacketLogger capture
of the DJI Ronin app's own BLE writes while dragging its on-screen
joystick (see docs/RS4_BLE_FINDINGS.md for the full analysis). The app
sends, continuously (~15-20 Hz) while any axis is off-center:

    sender=0x00, receiver=0x04 (gimbal), cmd_type=0x00 (request),
    cmd_set=0x04, cmd_id=0x01, 9-byte payload:
        [chA uint16 LE] [chB uint16 LE] [chC uint16 LE] [00 00] [02]

Each channel is centered at 1024 (rest/zero); the capture showed one
channel sweep smoothly between ~386 and ~1683 (held at each extreme),
and a second between ~364 and ~1680 -- symmetric ~1024 +/- 660, so that
660-wide half-range is the empirically *proven safe* deflection (rather
than assuming the theoretical 0..2047 11-bit full range). The third
channel stayed at 1024 (untouched) throughout -- its axis, and which of
chA/chC is pan vs. tilt, still needs live confirmation against real
hardware.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

DUML_SOF = 0x55
DUML_VERSION = 1
CMD_SET_GIMBAL = 0x04
HEADER_LEN = 4          # sof, len_lo, len_hi/ver, crc8
OVERHEAD = 13           # header(4) + sender/recv(2) + seq(2) + cmd(3) + crc16(2)


def _crc8_table() -> list[int]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x8C if c & 1 else c >> 1
        table.append(c)
    return table


def _crc16_table() -> list[int]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x8408 if c & 1 else c >> 1
        table.append(c)
    return table


_CRC8 = _crc8_table()
_CRC16 = _crc16_table()


def duml_crc8(data: bytes, init: int = 0x77) -> int:
    """Header checksum (byte 3) over bytes[0:3]."""
    crc = init
    for b in data:
        crc = _CRC8[(crc ^ b) & 0xFF]
    return crc & 0xFF


def duml_crc16(data: bytes, init: int = 0x3692) -> int:
    """Whole-frame checksum (last 2 bytes, LE) over bytes[0:-2]."""
    crc = init
    for b in data:
        crc = (crc >> 8) ^ _CRC16[(crc ^ b) & 0xFF]
    return crc & 0xFFFF


@dataclass(frozen=True)
class DjiDumlFrame:
    sender: int
    receiver: int
    seq: int
    cmd_type: int
    cmd_set: int
    cmd_id: int
    payload: bytes
    version: int = DUML_VERSION

    @classmethod
    def parse(cls, raw: bytes) -> "DjiDumlFrame":
        if len(raw) < OVERHEAD:
            raise ValueError(f"frame too short: {len(raw)} bytes")
        if raw[0] != DUML_SOF:
            raise ValueError(f"bad SOF: {raw[0]:#x}")
        length = raw[1] | ((raw[2] & 0x03) << 8)
        if length != len(raw):
            raise ValueError(f"length mismatch: header says {length}, got {len(raw)}")
        if duml_crc8(raw[0:3]) != raw[3]:
            raise ValueError("header CRC-8 mismatch")
        if duml_crc16(raw[:-2]) != int.from_bytes(raw[-2:], "little"):
            raise ValueError("frame CRC-16 mismatch")
        return cls(
            sender=raw[4],
            receiver=raw[5],
            seq=int.from_bytes(raw[6:8], "little"),
            cmd_type=raw[8],
            cmd_set=raw[9],
            cmd_id=raw[10],
            payload=bytes(raw[11:-2]),
            version=raw[2] >> 2,
        )

    def to_bytes(self) -> bytes:
        return build_duml_frame(
            self.sender, self.receiver, self.seq,
            self.cmd_type, self.cmd_set, self.cmd_id, self.payload,
            version=self.version,
        )


class DjiDumlAssembler:
    """Reassemble DUML frames from arbitrary BLE notification chunks and
    yield the CRC-valid ones (a corrupt/false-SOF byte is skipped and the
    stream re-synced on the next 0x55)."""

    def __init__(self, max_buffer: int = 4096):
        self._buffer = bytearray()
        self._max_buffer = max_buffer

    def feed(self, chunk: bytes) -> list[DjiDumlFrame]:
        self._buffer.extend(chunk)
        frames: list[DjiDumlFrame] = []
        while True:
            start = self._buffer.find(bytes([DUML_SOF]))
            if start < 0:
                self._buffer.clear()
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 3:
                break
            length = self._buffer[1] | ((self._buffer[2] & 0x03) << 8)
            if length < OVERHEAD or length > 0x3FF:
                del self._buffer[0]  # implausible length -> false SOF
                continue
            if len(self._buffer) < length:
                break
            raw = bytes(self._buffer[:length])
            try:
                frame = DjiDumlFrame.parse(raw)
            except ValueError:
                del self._buffer[0]  # bad CRC -> false SOF, resync
                continue
            frames.append(frame)
            del self._buffer[:length]
        if len(self._buffer) > self._max_buffer:
            del self._buffer[: -self._max_buffer]
        return frames


def build_duml_frame(
    sender: int,
    receiver: int,
    seq: int,
    cmd_type: int,
    cmd_set: int,
    cmd_id: int,
    payload: bytes = b"",
    version: int = DUML_VERSION,
) -> bytes:
    """Assemble a DUML frame with correct CRC-8/CRC-16 (verified against the
    RS 4 Pro's own frames)."""
    length = OVERHEAD + len(payload)
    if length > 0x3FF:
        raise ValueError("payload too long for a DUML frame")
    header = bytes([DUML_SOF, length & 0xFF, ((length >> 8) & 0x03) | (version << 2)])
    header += bytes([duml_crc8(header)])
    body = header + bytes([
        sender & 0xFF, receiver & 0xFF,
        seq & 0xFF, (seq >> 8) & 0xFF,
        cmd_type & 0xFF, cmd_set & 0xFF, cmd_id & 0xFF,
    ]) + bytes(payload)
    return body + duml_crc16(body).to_bytes(2, "little")


# -- BLE joystick control (cmd_set 0x04, cmd_id 0x01) --------------------
CMD_ID_JOYSTICK = 0x01
JOYSTICK_SENDER = 0x00
JOYSTICK_RECEIVER = 0x04
JOYSTICK_CENTER = 1024
# Proven-safe half-range from the capture (channel swept ~364..1683 and
# ~364..1680, i.e. 1024 +/- ~660) -- deliberately not the theoretical
# 0..2047 11-bit full range, which was never observed and could be
# clamped/rejected differently by the gimbal.
JOYSTICK_SPAN = 660
JOYSTICK_MIN = JOYSTICK_CENTER - JOYSTICK_SPAN
JOYSTICK_MAX = JOYSTICK_CENTER + JOYSTICK_SPAN


def joystick_channel_value(ratio: float) -> int:
    """ratio in [-1, 1] -> DUML joystick channel value, centered at 1024,
    clamped to the empirically proven-safe range."""
    ratio = max(-1.0, min(1.0, ratio))
    value = JOYSTICK_CENTER + round(ratio * JOYSTICK_SPAN)
    return max(JOYSTICK_MIN, min(JOYSTICK_MAX, value))


def build_joystick_frame(seq: int, ch_a: float, ch_c: float, ch_b: float = 0.0) -> bytes:
    """RS 4 Pro BLE gimbal joystick control frame. ch_a/ch_b/ch_c are
    ratios in [-1, 1]; axis assignment (which channel is pan/tilt/roll)
    is not yet confirmed against real hardware -- see
    docs/RS4_BLE_FINDINGS.md."""
    payload = struct.pack(
        "<HHH",
        joystick_channel_value(ch_a),
        joystick_channel_value(ch_b),
        joystick_channel_value(ch_c),
    ) + b"\x00\x00\x02"
    return build_duml_frame(
        sender=JOYSTICK_SENDER, receiver=JOYSTICK_RECEIVER, seq=seq,
        cmd_type=0x00, cmd_set=CMD_SET_GIMBAL, cmd_id=CMD_ID_JOYSTICK,
        payload=payload,
    )


# -- BLE recenter/HOME (cmd_set 0x04, cmd_id 0x4c) ------------------------
# Reverse-engineered from a third PacketLogger capture of the Ronin app's
# recenter control (see docs/RS4_BLE_FINDINGS.md). An earlier capture
# session had suggested cmd_set 0x00/cmd_id 0x34 as the recenter command,
# but that live-tested as a no-op against real hardware (gimbal did not
# move) and does not reappear in this capture at all -- it was evidently
# an unrelated command that happened to fire near that button press, not
# the actual action. This one is a much better fit: it lives under
# cmd_set 0x04 (GIMBAL, the same command set as the joystick control),
# fires exactly once (a true one-shot action, not a stream), and the
# surrounding traffic is just the same ambient chatter seen throughout
# every capture -- no other unusual command appears near it.
RECENTER_SENDER = 0x02
RECENTER_RECEIVER = 0x04
CMD_ID_RECENTER = 0x4C
RECENTER_PAYLOAD = bytes.fromhex("fe01")


def build_recenter_frame(seq: int) -> bytes:
    """RS 4 Pro BLE recenter/HOME command. Reproduces the captured frame's
    header and payload exactly (see docs/RS4_BLE_FINDINGS.md); only the
    sequence number and CRCs vary per call. Not yet confirmed live against
    real hardware -- see the log warning in DjiRs4ProBackend.go_home()."""
    return build_duml_frame(
        sender=RECENTER_SENDER, receiver=RECENTER_RECEIVER, seq=seq,
        cmd_type=0x40, cmd_set=CMD_SET_GIMBAL, cmd_id=CMD_ID_RECENTER,
        payload=RECENTER_PAYLOAD,
    )
