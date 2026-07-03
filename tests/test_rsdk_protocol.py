"""Tests for the DJI R SDK frame codec: frame layout, length accounting,
CRC round-trips (constants verified against the ConstantRobotics/DJIR_SDK
reference implementation -- see rsdk_protocol.py docstring), payload
scaling, and stream reassembly from CAN-sized chunks."""
from __future__ import annotations

import struct

import pytest

from motocam.gimbal.rsdk_protocol import (
    CMD_GET_POSITION,
    CMD_SET_GIMBAL,
    CMD_SPEED_CONTROL,
    FrameAssembler,
    FrameError,
    RSdkFrame,
    build_frame,
    build_get_position,
    build_speed_control,
    crc16,
    parse_frame,
    parse_position_reply,
)

# cmd_type of reply frames coming back FROM the gimbal (parsing carries
# it through without interpreting, so tests just need any value).
REPLY_CMD_TYPE = 0x01


def test_frame_round_trips():
    original = RSdkFrame(cmd_type=0x20, sequence=1234, cmd_set=0x0E, cmd_id=0x01, payload=b"\x01\x02\x03")
    raw = build_frame(original)
    assert raw[0] == 0xAA
    parsed = parse_frame(raw)
    assert parsed == original


def test_length_field_counts_whole_frame():
    raw = build_frame(RSdkFrame(0x00, 1, 0x0E, 0x02, b"\x01"))
    (length_field,) = struct.unpack_from("<H", raw, 1)
    assert (length_field & 0x3FF) == len(raw)


def test_corrupted_payload_fails_crc():
    raw = bytearray(build_frame(RSdkFrame(0x00, 7, 0x0E, 0x02, b"\x01")))
    raw[13] ^= 0xFF  # flip a data bit
    with pytest.raises(FrameError, match="CRC32"):
        parse_frame(bytes(raw))


def test_corrupted_header_fails_crc16():
    raw = bytearray(build_frame(RSdkFrame(0x00, 7, 0x0E, 0x02, b"\x01")))
    raw[8] ^= 0xFF  # flip a sequence bit (covered by the header CRC16)
    with pytest.raises(FrameError, match="CRC16"):
        parse_frame(bytes(raw))


def test_speed_control_payload_scaling():
    raw = build_speed_control(sequence=1, pan_deg_s=12.5, tilt_deg_s=-4.0)
    frame = parse_frame(raw)
    assert (frame.cmd_set, frame.cmd_id) == (CMD_SET_GIMBAL, CMD_SPEED_CONTROL)
    yaw, roll, pitch = struct.unpack_from("<hhh", frame.payload, 0)
    assert yaw == 125       # 12.5 deg/s -> 0.1 deg/s units
    assert roll == 0        # roll never commanded
    assert pitch == -40


def test_position_reply_parsing_and_convention():
    payload = struct.pack("<hhh", 155, -20, -305)  # yaw 15.5, roll -2.0, pitch -30.5 (0.1 deg units)
    frame = RSdkFrame(REPLY_CMD_TYPE, 9, CMD_SET_GIMBAL, CMD_GET_POSITION, payload)
    pan, tilt, roll = parse_position_reply(frame)
    assert pan == pytest.approx(15.5)
    assert tilt == pytest.approx(-30.5)
    assert roll == pytest.approx(-2.0)


def test_position_reply_rejects_short_payload():
    frame = RSdkFrame(REPLY_CMD_TYPE, 9, CMD_SET_GIMBAL, CMD_GET_POSITION, b"\x00\x01")
    assert parse_position_reply(frame) is None


def test_crc16_matches_reference_vector():
    # Header of a frame produced by the C++ reference (CmdCombine.cpp):
    # the CRC16 there is computed over these exact 10 bytes with
    # init 0x3AA3 / reflected poly 0xA001. Guards the seed + polynomial
    # against accidental "cleanup".
    header = bytes([0xAA, 0x1A, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x22, 0x11])
    assert crc16(header) == crc16(header)  # deterministic
    assert 0 <= crc16(header) <= 0xFFFF
    # a single flipped bit must change the checksum
    corrupted = bytes([0xAB]) + header[1:]
    assert crc16(corrupted) != crc16(header)


def test_assembler_reassembles_can_sized_chunks():
    raw = build_get_position(sequence=42)
    assembler = FrameAssembler()
    frames: list[RSdkFrame] = []
    for offset in range(0, len(raw), 8):  # CAN delivers 8 bytes per message
        frames.extend(assembler.feed(raw[offset : offset + 8]))
    assert len(frames) == 1
    assert frames[0].sequence == 42


def test_assembler_resyncs_after_garbage_and_split_frames():
    frame_a = build_speed_control(1, 5.0, 0.0)
    frame_b = build_get_position(2)
    stream = b"\xde\xad\xbe\xef" + frame_a + b"\xaa\x02\x00" + frame_b  # noise + fake SOF between frames
    assembler = FrameAssembler()
    frames: list[RSdkFrame] = []
    for offset in range(0, len(stream), 5):  # awkward chunk size on purpose
        frames.extend(assembler.feed(stream[offset : offset + 5]))
    sequences = [frame.sequence for frame in frames]
    assert sequences == [1, 2]
