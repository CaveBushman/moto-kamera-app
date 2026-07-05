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
built here carry checksums the gimbal accepts. This module owns *framing*
only; the specific command payloads (set speed / position) still need a
write-side capture to pin down -- this capture is telemetry the gimbal
pushes, not the app's commands.
"""
from __future__ import annotations

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
