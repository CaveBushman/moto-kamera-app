"""DJI R SDK frame codec + command builders (pure logic, no I/O).

The R SDK is DJI's official wired-control protocol for the Ronin/RS
gimbal family, spoken over the RSA port (CAN bus or UART). The frame
layout implemented here follows the R SDK documentation as reproduced
by multiple independent community implementations (RS 2 / RS 3 Pro CAN
control projects):

    offset  size  field
    0       1     SOF 0xAA
    1       2     length (10 bits, little-endian) | version << 10
    3       1     cmd_type   (request/reply + ack flags)
    4       1     enc        (0 = no encryption)
    5       3     reserved
    8       2     sequence   (little-endian, incrementing)
    10      2     CRC16 over bytes 0..9 (little-endian)
    12      ...   data: cmd_set, cmd_id, payload...
    end-4   4     CRC32 over everything before it (little-endian)

`length` counts the WHOLE frame including both CRCs. The sequence field
is written high-byte-first (matching the reference implementation), the
CRCs little-endian.

Constants below are VERIFIED against ConstantRobotics/DJIR_SDK -- an
open-source C++ implementation of the official "DJI R SDK Protocol and
User Interface" (v2.2) that its authors run against real RS 2 hardware
over USBCAN. Frame layout, CRC polynomials/seeds, cmd_type, ctrl bytes,
payload layouts and CAN arbitration IDs all match that implementation
1:1. Remaining hardware step: a smoke test on the RS 4 Pro itself
(same protocol family; the backend refuses to report connected until
the gimbal actually answers, so any RS4-specific difference shows up
as DISCONNECTED, never as fake control).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

SOF = 0xAA
FRAME_OVERHEAD = 16  # 12-byte header + 4-byte CRC32

# CRC seeds -- reflected-table algorithms; verified: pycrc-generated
# tables in the reference use XorIn 0xC55C / 0xC55C0000, which in the
# reflected implementation becomes a working init of 0x3AA3 for both.
CRC16_INIT = 0x3AA3
CRC32_INIT = 0x3AA3

# Command set / IDs for the handheld gimbal (R SDK "cmd_set 0x0E").
CMD_SET_GIMBAL = 0x0E
CMD_POSITION_CONTROL = 0x00
CMD_SPEED_CONTROL = 0x01
CMD_GET_POSITION = 0x02

# cmd_type 0x03 is what the reference sends for every gimbal command
# (position, speed, get-position). Replies come back with a different
# cmd_type; parsing carries it through without interpreting.
CMD_TYPE_CTRL = 0x03

# Speed-control ctrl byte: BIT7 (0x40) enables speed control, BIT3
# (0x04) keeps focal control disabled -- 0x44 = "speed control on".
# Position-control ctrl byte: BIT1 (0x01) = absolute move.
SPEED_CTRL_BYTE = 0x44
POSITION_CTRL_ABSOLUTE = 0x01


def crc16(data: bytes, init: int = CRC16_INIT) -> int:
    """Reflected CRC-16, polynomial 0x8005 (table form 0xA001), R SDK seed."""
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def crc32(data: bytes, init: int = CRC32_INIT) -> int:
    """Reflected CRC-32, polynomial 0x04C11DB7 (table form 0xEDB88320), R SDK seed."""
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return crc & 0xFFFFFFFF


@dataclass(frozen=True)
class RSdkFrame:
    """One decoded/pre-encode R SDK command or reply (see the wire layout
    at the top of this module)."""

    cmd_type: int
    sequence: int
    cmd_set: int
    cmd_id: int
    payload: bytes


def build_frame(frame: RSdkFrame, version: int = 0) -> bytes:
    """Encode an RSdkFrame to wire bytes with both CRCs filled in."""
    data = bytes([frame.cmd_set, frame.cmd_id]) + frame.payload
    total_length = FRAME_OVERHEAD + len(data)
    if total_length > 0x3FF:
        raise ValueError(f"frame too long: {total_length}")
    length_field = total_length | (version << 10)

    header = struct.pack(
        "<BHBB3s",
        SOF,
        length_field,
        frame.cmd_type,
        0,  # enc
        b"\x00\x00\x00",
    )
    header += struct.pack(">H", frame.sequence & 0xFFFF)  # seq is high-byte-first on the wire
    header += struct.pack("<H", crc16(header))
    body = header + data
    return body + struct.pack("<I", crc32(body))


class FrameError(ValueError):
    pass


def parse_frame(raw: bytes) -> RSdkFrame:
    """Decode wire bytes to an RSdkFrame, raising FrameError if either
    CRC or the length field doesn't check out."""
    if len(raw) < FRAME_OVERHEAD + 2:
        raise FrameError(f"frame too short: {len(raw)} bytes")
    if raw[0] != SOF:
        raise FrameError(f"bad SOF: {raw[0]:#x}")
    (length_field,) = struct.unpack_from("<H", raw, 1)
    total_length = length_field & 0x3FF
    if total_length != len(raw):
        raise FrameError(f"length field {total_length} != actual {len(raw)}")
    (header_crc,) = struct.unpack_from("<H", raw, 10)
    if crc16(raw[:10]) != header_crc:
        raise FrameError("header CRC16 mismatch")
    (frame_crc,) = struct.unpack_from("<I", raw, len(raw) - 4)
    if crc32(raw[:-4]) != frame_crc:
        raise FrameError("frame CRC32 mismatch")

    cmd_type = raw[3]
    (sequence,) = struct.unpack_from(">H", raw, 8)  # high-byte-first, see build_frame
    data = raw[12:-4]
    if len(data) < 2:
        raise FrameError("frame has no cmd_set/cmd_id")
    return RSdkFrame(
        cmd_type=cmd_type, sequence=sequence, cmd_set=data[0], cmd_id=data[1], payload=bytes(data[2:])
    )


# -- command payloads (angles/speeds in 0.1-degree units, little-endian) ----

def build_speed_control(sequence: int, pan_deg_s: float, tilt_deg_s: float) -> bytes:
    """Continuous velocity command: yaw/roll/pitch in 0.1 deg/s. The app's
    pan maps to yaw, tilt to pitch; roll is never commanded (the gimbal
    self-levels)."""
    payload = struct.pack(
        "<hhhB",
        int(round(pan_deg_s * 10)),
        0,
        int(round(tilt_deg_s * 10)),
        SPEED_CTRL_BYTE,
    )
    return build_frame(RSdkFrame(CMD_TYPE_CTRL, sequence, CMD_SET_GIMBAL, CMD_SPEED_CONTROL, payload))


def build_position_control(
    sequence: int, pan_deg: float, tilt_deg: float, duration_ms: int = 2000
) -> bytes:
    """Absolute move -- used for HOME (0/0). Duration is expressed in
    0.1 s units per the R SDK payload layout."""
    payload = struct.pack(
        "<hhhBB",
        int(round(pan_deg * 10)),
        0,
        int(round(tilt_deg * 10)),
        POSITION_CTRL_ABSOLUTE,
        max(1, min(255, duration_ms // 100)),
    )
    return build_frame(RSdkFrame(CMD_TYPE_CTRL, sequence, CMD_SET_GIMBAL, CMD_POSITION_CONTROL, payload))


def build_get_position(sequence: int) -> bytes:
    """Request the gimbal's current orientation (paired with parse_position_reply)."""
    return build_frame(RSdkFrame(CMD_TYPE_CTRL, sequence, CMD_SET_GIMBAL, CMD_GET_POSITION, b"\x01"))


def parse_position_reply(frame: RSdkFrame) -> tuple[float, float, float] | None:
    """Reply to GET_POSITION: yaw/roll/pitch int16 in 0.1 degrees,
    directly at payload offset 0 (verified layout -- the reference
    implementation reads frame bytes 14/16/18, i.e. the first six
    payload bytes). Returns (pan, tilt, roll) in the app's convention,
    or None when the frame isn't a well-formed position reply."""
    if frame.cmd_set != CMD_SET_GIMBAL or frame.cmd_id != CMD_GET_POSITION:
        return None
    if len(frame.payload) < 6:
        return None
    yaw, roll, pitch = struct.unpack_from("<hhh", frame.payload, 0)
    return yaw / 10.0, pitch / 10.0, roll / 10.0


class FrameAssembler:
    """Reassembles R SDK frames from a chunked byte stream -- CAN delivers
    at most 8 bytes per bus message and UART is an arbitrary stream, so
    both transports feed bytes in here and pull complete frames out."""

    def __init__(self, max_buffer: int = 4096):
        self._buffer = bytearray()
        self._max_buffer = max_buffer

    def feed(self, chunk: bytes) -> list[RSdkFrame]:
        self._buffer.extend(chunk)
        frames: list[RSdkFrame] = []
        while True:
            start = self._buffer.find(bytes([SOF]))
            if start < 0:
                self._buffer.clear()
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 3:
                break
            (length_field,) = struct.unpack_from("<H", self._buffer, 1)
            total_length = length_field & 0x3FF
            if total_length < FRAME_OVERHEAD + 2:
                # Not a real frame start -- skip this SOF byte and rescan.
                del self._buffer[0]
                continue
            if len(self._buffer) < total_length:
                break  # wait for more bytes
            candidate = bytes(self._buffer[:total_length])
            try:
                frames.append(parse_frame(candidate))
                del self._buffer[:total_length]
            except FrameError:
                del self._buffer[0]  # false SOF; resync on the next one
        if len(self._buffer) > self._max_buffer:
            del self._buffer[: -self._max_buffer]
        return frames
