"""Helpers for listing PortAudio devices in the settings UI."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from motocam.audio.ptt_engine import SOUNDDEVICE_AVAILABLE

logger = logging.getLogger("motocam.audio")

if SOUNDDEVICE_AVAILABLE:
    import sounddevice as sd


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channels: int

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name} ({self.channels} ch)"


def list_audio_devices(kind: Literal["input", "output"]) -> list[AudioDevice]:
    if not SOUNDDEVICE_AVAILABLE:
        return []

    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("Failed to list %s audio devices (%s)", kind, exc)
        return []

    result: list[AudioDevice] = []
    for index, device in enumerate(devices):
        channels = int(device.get(channel_key, 0))
        if channels > 0:
            result.append(AudioDevice(index=index, name=str(device.get("name", "Unknown")), channels=channels))
    return result
