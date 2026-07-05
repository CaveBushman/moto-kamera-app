"""Tests for the DJI DUML transport, checked against real RS 4 Pro BLE
frames (captured hex, so they don't depend on the git-ignored capture
files). Every real frame's CRC-8 and CRC-16 validated 615/615 in the
capture; these lock the framing so a future refactor can't drift."""
from __future__ import annotations

import pytest

from motocam.gimbal.dji_duml import (
    CMD_SET_GIMBAL,
    DjiDumlFrame,
    build_duml_frame,
    duml_crc16,
    duml_crc8,
)

# Real frames pulled from the RS 4 Pro BLE capture (cmd_set 0x04 = gimbal).
REAL_FRAMES = [
    "551204c70402719500042700800000002014",
    "551204c70402d5950004270080000000ffaa",
    "551204c704023996000427008000000075fa",
]


def test_crcs_match_a_real_frame():
    raw = bytes.fromhex(REAL_FRAMES[0])
    assert duml_crc8(raw[0:3]) == raw[3]
    assert duml_crc16(raw[:-2]) == int.from_bytes(raw[-2:], "little")


def test_parse_real_gimbal_frame():
    frame = DjiDumlFrame.parse(bytes.fromhex(REAL_FRAMES[0]))
    assert frame.sender == 0x04
    assert frame.receiver == 0x02
    assert frame.cmd_type == 0x00       # request/push
    assert frame.cmd_set == CMD_SET_GIMBAL
    assert frame.cmd_id == 0x27
    assert frame.seq == 0x9571          # little-endian 71 95
    assert frame.version == 1
    assert frame.payload == bytes.fromhex("00800000 00".replace(" ", ""))


@pytest.mark.parametrize("hexstr", REAL_FRAMES)
def test_build_reproduces_real_frames_byte_for_byte(hexstr):
    raw = bytes.fromhex(hexstr)
    frame = DjiDumlFrame.parse(raw)
    assert frame.to_bytes() == raw  # rebuilt frame (incl. CRCs) is identical


def test_build_then_parse_round_trips():
    built = build_duml_frame(
        sender=0x02, receiver=0x04, seq=0x1234,
        cmd_type=0x00, cmd_set=CMD_SET_GIMBAL, cmd_id=0x01,
        payload=bytes([0x10, 0x20, 0x30]),
    )
    frame = DjiDumlFrame.parse(built)
    assert frame.sender == 0x02 and frame.receiver == 0x04
    assert frame.seq == 0x1234
    assert frame.cmd_set == CMD_SET_GIMBAL and frame.cmd_id == 0x01
    assert frame.payload == bytes([0x10, 0x20, 0x30])
    assert len(built) == 13 + 3  # OVERHEAD + payload


def test_parse_rejects_corrupt_crc():
    raw = bytearray(bytes.fromhex(REAL_FRAMES[0]))
    raw[-1] ^= 0xFF  # break the CRC-16
    with pytest.raises(ValueError):
        DjiDumlFrame.parse(bytes(raw))


def test_parse_rejects_bad_sof():
    raw = bytearray(bytes.fromhex(REAL_FRAMES[0]))
    raw[0] = 0xAA
    with pytest.raises(ValueError):
        DjiDumlFrame.parse(bytes(raw))


# -- DUML reassembly from BLE chunks --------------------------------------
from motocam.gimbal.dji_duml import DjiDumlAssembler  # noqa: E402


def test_assembler_reassembles_a_split_frame():
    asm = DjiDumlAssembler()
    raw = bytes.fromhex(REAL_FRAMES[0])
    out = []
    for i in range(0, len(raw), 5):  # dribble it in 5-byte BLE-ish chunks
        out += asm.feed(raw[i : i + 5])
    assert len(out) == 1
    assert out[0].cmd_set == CMD_SET_GIMBAL and out[0].cmd_id == 0x27


def test_assembler_skips_garbage_before_a_frame():
    asm = DjiDumlAssembler()
    frames = asm.feed(b"\x00\x11\x22" + bytes.fromhex(REAL_FRAMES[1]))
    assert len(frames) == 1
    assert frames[0].cmd_set == CMD_SET_GIMBAL


def test_assembler_yields_multiple_frames_in_one_chunk():
    asm = DjiDumlAssembler()
    blob = bytes.fromhex(REAL_FRAMES[0]) + bytes.fromhex(REAL_FRAMES[2])
    frames = asm.feed(blob)
    assert len(frames) == 2
