"""Helpers for recording DJI's RS 4 Pro BLE vendor frames.

The public DJI R SDK frames used by the RSA UART/CAN port start with
0xAA. The RS 4 Pro BLE profile observed on real hardware emits a
different vendor stream whose frames start with 0x55 and carry their
total length in byte 1. This module deliberately does not claim to
decode commands yet; it only frames and labels captures so we can build
evidence without mixing packet-boundary guesses into scripts.
"""
from __future__ import annotations

from dataclasses import dataclass

VENDOR_SOF = 0x55
MIN_VENDOR_FRAME_LEN = 4
MAX_VENDOR_FRAME_LEN = 255


@dataclass(frozen=True)
class DjiBleVendorFrame:
    raw: bytes

    @property
    def length(self) -> int:
        return len(self.raw)

    @property
    def signature(self) -> str:
        """Stable-ish frame grouping key.

        Hardware captures so far show bytes 2..5 distinguishing the
        recurring frame families, while later bytes include counters and
        payloads. If a shorter frame appears, return whatever is present
        after SOF/length.
        """
        return self.raw[2:6].hex()

    @property
    def first_payload_hex(self) -> str:
        return self.raw[6:14].hex()

    def to_capture_dict(self) -> dict[str, object]:
        return {
            "len": self.length,
            "signature": self.signature,
            "first_payload_hex": self.first_payload_hex,
            "hex": self.raw.hex(),
        }


class DjiBleVendorAssembler:
    """Split arbitrary BLE notification chunks into 0x55 vendor frames."""

    def __init__(self, max_buffer: int = 4096):
        self._buffer = bytearray()
        self._max_buffer = max_buffer

    def feed(self, chunk: bytes) -> list[DjiBleVendorFrame]:
        self._buffer.extend(chunk)
        frames: list[DjiBleVendorFrame] = []
        while True:
            start = self._buffer.find(bytes([VENDOR_SOF]))
            if start < 0:
                self._buffer.clear()
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 2:
                break
            frame_len = self._buffer[1]
            if frame_len < MIN_VENDOR_FRAME_LEN or frame_len > MAX_VENDOR_FRAME_LEN:
                del self._buffer[0]
                continue
            if len(self._buffer) < frame_len:
                break
            raw = bytes(self._buffer[:frame_len])
            del self._buffer[:frame_len]
            frames.append(DjiBleVendorFrame(raw=raw))
        if len(self._buffer) > self._max_buffer:
            del self._buffer[: -self._max_buffer]
        return frames

    @property
    def buffered_len(self) -> int:
        return len(self._buffer)
