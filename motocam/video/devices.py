"""Helpers for listing UVC/V4L2 capture devices in the settings UI.

OpenCV has no portable "list devices with names" API. On Linux we read
the real device name out of /sys/class/video4linux (V4L2 exposes that for
free) and check whether each node is a *capture* node with a cheap,
non-blocking V4L2 QUERYCAP ioctl -- crucially WITHOUT opening the device
through cv2.VideoCapture, which negotiates formats and can take seconds
per node (and can block outright on the /dev/videoN the running capture
already holds). That blocking probe used to freeze the whole UI every
time Settings was opened on the Pi. On other platforms (dev laptops
without a Magewell grabber) we probe a handful of numeric indices, since
that's all cv2.VideoCapture(int) gives us to work with.
"""
from __future__ import annotations

import fcntl
import glob
import logging
import os
import platform
from dataclasses import dataclass

import cv2

logger = logging.getLogger("motocam.video")

PROBE_INDEX_COUNT = 6

# V4L2 QUERYCAP ioctl (linux/videodev2.h). VIDIOC_QUERYCAP = _IOR('V', 0,
# struct v4l2_capability) where the struct is 104 bytes. We only read the
# capabilities/device_caps words to tell a capture node from an output or
# metadata node.
_IOC_READ = 2
_QUERYCAP_STRUCT_SIZE = 104
_VIDIOC_QUERYCAP = (_IOC_READ << 30) | (_QUERYCAP_STRUCT_SIZE << 16) | (ord("V") << 8) | 0
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_DEVICE_CAPS = 0x80000000


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
        if not _v4l2_is_capture(path):
            continue
        name_path = f"/sys/class/video4linux/{path.rsplit('/', 1)[-1]}/name"
        try:
            with open(name_path) as f:
                name = f.read().strip()
        except OSError:
            name = path
        result.append(VideoDevice(device=path, name=name))
    return result


def _v4l2_is_capture(path: str) -> bool:
    """True if `path` is a V4L2 video-capture node. Uses a non-blocking
    QUERYCAP ioctl so it neither stalls nor disturbs the device the running
    VideoEngine is already streaming from."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        buf = bytearray(_QUERYCAP_STRUCT_SIZE)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf, True)
    except OSError:
        return False
    finally:
        os.close(fd)
    capabilities = int.from_bytes(buf[84:88], "little")
    device_caps = int.from_bytes(buf[88:92], "little")
    return _capture_capable(capabilities, device_caps)


def _capture_capable(capabilities: int, device_caps: int) -> bool:
    """Whether a QUERYCAP result describes a video-capture node. When the
    driver reports per-node device_caps (V4L2_CAP_DEVICE_CAPS set), trust
    those over the device-wide capabilities -- a single physical device can
    expose separate capture and metadata/output nodes."""
    effective = device_caps if (capabilities & _V4L2_CAP_DEVICE_CAPS) else capabilities
    return bool(effective & _V4L2_CAP_VIDEO_CAPTURE)


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
