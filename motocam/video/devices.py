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
that's all cv2.VideoCapture(int) gives us to work with -- bounded by a
wall-clock deadline (PROBE_TIMEOUT_S) since the same class of freeze
turned out to hit macOS too (see PROBE_TIMEOUT_S's comment).
"""
from __future__ import annotations

import fcntl
import glob
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass

import cv2

logger = logging.getLogger("motocam.video")

PROBE_INDEX_COUNT = 6
# cv2.VideoCapture(index)'s constructor can block for a long time -- or,
# per a live macOS crash report, indefinitely inside AVFoundation's
# grabImageUntilDate while validating an index that's out of range or
# permission-gated. That single blocking probe froze the whole app (not
# just Settings) and, worse, left the DeviceScanWorker QThread still
# running when the app tried to exit -- Qt's QThread destructor calls
# qFatal() (a hard abort()) if it's destroyed while still alive, which is
# exactly the SIGABRT the crash report showed. Each probe now gets a
# bounded wall-clock budget via a daemon thread (never a
# ThreadPoolExecutor -- those aren't daemonic and Python's atexit
# machinery would block process exit waiting for an abandoned probe to
# finish, reintroducing the same hang at shutdown instead of at scan
# time); a timed-out probe is simply abandoned (leaked, not killed --
# there's no API to cancel an in-flight VideoCapture open) and treated
# as "no camera there".
#
# macOS gets a much longer budget than the generic case: a *healthy*
# AVFoundation open routinely takes 1-3s (device warm-up, permission
# machinery), so 0.6s classified every real laptop camera as "hung" and
# Settings showed an empty Capture-device list -- nothing to select. The
# scan runs on DeviceScanWorker's own QThread, so a longer wait costs scan
# latency only, never UI freeze; the hang scenario the timeout exists for
# (indefinite grabImageUntilDate) is still bounded, just at 4s.
PROBE_TIMEOUT_S = 4.0 if platform.system() == "Darwin" else 0.6

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


def _probe_index(index: int, results: dict[int, VideoDevice], lock: threading.Lock) -> None:
    try:
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                with lock:
                    results[index] = VideoDevice(device=index, name=f"Camera {index}")
        finally:
            cap.release()
    except Exception as exc:  # noqa: BLE001 -- one bad index must not break the scan
        logger.debug("Camera index %d probe failed: %s", index, exc)


def _probe_indices() -> list[VideoDevice]:
    """Probe indices 0..PROBE_INDEX_COUNT-1 in parallel, each on its own
    daemon thread, against a single shared PROBE_TIMEOUT_S deadline (see
    module-level comment) -- joining against a per-thread timeout instead
    would let N hung indices add up to N*PROBE_TIMEOUT_S, defeating the
    whole point of bounding this scan. Total wall-clock time here is
    bounded by PROBE_TIMEOUT_S regardless of how many indices hang."""
    results: dict[int, VideoDevice] = {}
    lock = threading.Lock()
    threads = [
        threading.Thread(target=_probe_index, args=(index, results, lock), daemon=True)
        for index in range(PROBE_INDEX_COUNT)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + PROBE_TIMEOUT_S
    for index, thread in enumerate(threads):
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)
        if thread.is_alive():
            logger.warning("Camera index %d probe exceeded %.1fs, skipping it", index, PROBE_TIMEOUT_S)
    return [results[index] for index in sorted(results)]
