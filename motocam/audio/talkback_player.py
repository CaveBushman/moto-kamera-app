"""Playback for director talkback audio sent from the control room."""
from __future__ import annotations

import logging
import threading

import numpy as np
from PyQt6.QtCore import QObject

from motocam.audio.ptt_engine import SAMPLE_RATE, SOUNDDEVICE_AVAILABLE

logger = logging.getLogger("motocam.audio")

if SOUNDDEVICE_AVAILABLE:
    import sounddevice as sd

BYTES_PER_FRAME = 2  # int16 mono


class TalkbackPlayer(QObject):
    """Plays director talkback PCM chunks arriving over the WebSocket link
    (see LinkClient) through a small jitter buffer: `feed()` (any thread)
    appends to `_buffer`, the PortAudio callback `_on_output` (its own
    thread) drains it, zero-padding on underrun instead of glitching."""

    def __init__(self, output_device: int | str | None = None):
        super().__init__()
        self.output_device = output_device
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stream: "sd.OutputStream | None" = None
        self.available = self._probe_output_device(output_device)

    @staticmethod
    def _probe_output_device(device: int | str | None = None) -> bool:
        if not SOUNDDEVICE_AVAILABLE:
            return False
        try:
            if device is None:
                return sd.query_devices(kind="output") is not None
            return sd.query_devices(device=device, kind="output") is not None
        except Exception as exc:
            logger.warning("No audio output available for control-room talkback (%s)", exc)
            return False

    def set_output_device(self, device: int | str | None) -> None:
        if device == self.output_device:
            return
        self.stop()
        self.output_device = device
        self.available = self._probe_output_device(device)

    def start(self) -> bool:
        if not self.available:
            return False
        if self._stream is not None:
            return True
        try:
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=self.output_device,
                callback=self._on_output,
            )
            self._stream.start()
            logger.info("Control-room talkback playback opened")
            return True
        except Exception as exc:
            logger.warning("Failed to open audio output for control-room talkback (%s)", exc)
            self.available = False
            self._stream = None
            return False

    def feed(self, pcm_bytes: bytes, sample_rate: int) -> None:
        if sample_rate != SAMPLE_RATE:
            logger.warning("Talkback audio at unexpected sample rate %d (expected %d)", sample_rate, SAMPLE_RATE)
        if not self.start():
            return
        with self._lock:
            self._buffer.extend(pcm_bytes)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._buffer.clear()

    def _on_output(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning("Talkback playback stream status: %s", status)
        needed = frames * BYTES_PER_FRAME
        with self._lock:
            chunk = bytes(self._buffer[:needed])
            del self._buffer[:needed]
        if len(chunk) < needed:
            chunk += b"\x00" * (needed - len(chunk))
        outdata[:] = np.frombuffer(chunk, dtype="int16").reshape(-1, 1)
