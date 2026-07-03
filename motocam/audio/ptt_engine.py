"""Push-to-talk microphone capture (operator -> director, one-way).

Mono 16 kHz PCM16, chunked at 20ms (320 samples) -- small enough for
low perceived latency over the same WebSocket link telemetry already
uses, without needing a separate real-time media transport for what's
fundamentally short bursts of speech, not continuous streaming.

If no input device is available (or PortAudio fails to open one) this
degrades the same way video/GPS/gimbal do elsewhere in this app:
`available` goes False, the PTT button greys out, nothing is fabricated.
"""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger("motocam.audio")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 320  # 20ms @ 16kHz


class PttEngine(QObject):
    audio_chunk = pyqtSignal(bytes)

    def __init__(self, input_device: int | str | None = None):
        super().__init__()
        self.input_device = input_device
        self._stream: "sd.InputStream | None" = None
        self.available = self._probe_input_device(input_device)

    @staticmethod
    def _probe_input_device(device: int | str | None = None) -> bool:
        if not SOUNDDEVICE_AVAILABLE:
            return False
        try:
            if device is None:
                return sd.query_devices(kind="input") is not None
            return sd.query_devices(device=device, kind="input") is not None
        except Exception as exc:  # sounddevice raises plain Exception/PortAudioError
            logger.warning("No microphone available for PTT (%s)", exc)
            return False

    def set_input_device(self, device: int | str | None) -> None:
        if device == self.input_device:
            return
        self.stop()
        self.input_device = device
        self.available = self._probe_input_device(device)

    def start(self) -> bool:
        if not self.available or self._stream is not None:
            return self.available
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SAMPLES,
                device=self.input_device,
                callback=self._on_audio,
            )
            self._stream.start()
            logger.info("PTT capture started")
            return True
        except Exception as exc:
            logger.warning("Failed to open microphone for PTT (%s)", exc)
            self.available = False
            self._stream = None
            return False

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("PTT capture stopped")

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # Runs on PortAudio's own thread, not the Qt thread -- emitting a
        # signal here is safe, PyQt queues delivery to this QObject's
        # (main-thread) owner automatically.
        if status:
            logger.warning("PTT audio stream status: %s", status)
        self.audio_chunk.emit(bytes(indata))
