"""Helpers for listing UVC/V4L2 capture devices in the settings UI.

OpenCV has no portable "list devices with names" API, so this probes: on
Linux it reads the real device name out of /sys/class/video4linux (V4L2
exposes that for free) and falls back to opening each /dev/videoN; on
other platforms (dev laptops without a Magewell grabber) it just probes
a handful of numeric indices, since that's all cv2.VideoCapture(int)
gives us to work with.
"""
from __future__ import annotations

import glob
import logging
import platform
from dataclasses import dataclass

import cv2

logger = logging.getLogger("motocam.video")

PROBE_INDEX_COUNT = 6


@dataclass(frozen=True)
class VideoDevice:
    device: str | int
    name: str

    @property
    def label(self) -> str:
        return f"{self.device}: {self.name}"


def list_video_devices() -> list[VideoDevice]:
    if platform.system() == "Linux":
        return _list_linux_devices()
    return _probe_indices()


def _list_linux_devices() -> list[VideoDevice]:
    result: list[VideoDevice] = []
    for path in sorted(glob.glob("/dev/video*")):
        name_path = f"/sys/class/video4linux/{path.rsplit('/', 1)[-1]}/name"
        try:
            with open(name_path) as f:
                name = f.read().strip()
        except OSError:
            name = path
        cap = cv2.VideoCapture(path)
        try:
            if cap.isOpened():
                result.append(VideoDevice(device=path, name=name))
        finally:
            cap.release()
    return result


def _probe_indices() -> list[VideoDevice]:
    result: list[VideoDevice] = []
    for index in range(PROBE_INDEX_COUNT):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                result.append(VideoDevice(device=index, name=f"Camera {index}"))
        finally:
            cap.release()
    return result
