"""Tests for V4L2 device enumeration used by the Settings dialog. The real
ioctl needs a device, but the capability decoding and the QUERYCAP ioctl
number are pure and are exactly what determines whether Settings freezes
(we must never fall back to opening every /dev/videoN through cv2)."""
from __future__ import annotations

import time

from motocam.video import devices as devices_module
from motocam.video.devices import (
    _V4L2_CAP_DEVICE_CAPS,
    _V4L2_CAP_VIDEO_CAPTURE,
    _VIDIOC_QUERYCAP,
    _capture_capable,
)


def test_querycap_ioctl_number_matches_kernel_constant():
    # VIDIOC_QUERYCAP is a stable UAPI constant; if this changes the ioctl
    # silently fails and enumeration returns nothing.
    assert _VIDIOC_QUERYCAP == 0x80685600


def test_capture_node_detected_from_device_wide_caps():
    assert _capture_capable(_V4L2_CAP_VIDEO_CAPTURE, 0) is True


def test_output_only_node_rejected():
    v4l2_cap_video_output = 0x00000002
    assert _capture_capable(v4l2_cap_video_output, 0) is False


def test_device_caps_preferred_when_flag_set():
    # Device-wide caps advertise capture, but THIS node's device_caps do
    # not (it's the metadata node) -- trust the per-node caps.
    capabilities = _V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_DEVICE_CAPS
    device_caps = 0x00400000  # V4L2_CAP_META_CAPTURE, not video capture
    assert _capture_capable(capabilities, device_caps) is False


def test_device_caps_capture_accepted_when_flag_set():
    capabilities = _V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_DEVICE_CAPS
    device_caps = _V4L2_CAP_VIDEO_CAPTURE
    assert _capture_capable(capabilities, device_caps) is True


# -- macOS fallback probe: bounded wall-clock time -------------------------
class _HangingVideoCapture:
    """Stands in for cv2.VideoCapture: index 0 "hangs" (simulates the live
    macOS crash report -- AVFoundation blocked inside grabImageUntilDate),
    every other index opens instantly."""

    def __init__(self, index: int):
        self.index = index
        if index == 0:
            time.sleep(5.0)

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass


def test_probe_indices_bounded_time_when_a_camera_hangs(monkeypatch):
    """A hung cv2.VideoCapture() must not block the whole scan -- this is
    the exact freeze from the live macOS crash report: DeviceScanWorker
    stuck inside a single index's open, and because that QThread never
    finished, the app aborted (SIGABRT) on exit since Qt's QThread
    destructor fatals if destroyed while still running."""
    monkeypatch.setattr(devices_module, "cv2", type("cv2", (), {"VideoCapture": _HangingVideoCapture}))
    monkeypatch.setattr(devices_module, "PROBE_TIMEOUT_S", 0.3)

    start = time.monotonic()
    result = devices_module._probe_indices()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"scan took {elapsed:.2f}s, should be bounded near PROBE_TIMEOUT_S"
    assert all(device.device != 0 for device in result)  # hung index excluded
    assert any(device.device == 1 for device in result)  # other indices still reported


def test_probe_indices_total_time_bounded_regardless_of_hung_count(monkeypatch):
    """Multiple hung indices must not add up (N * PROBE_TIMEOUT_S) --
    they're joined against one shared deadline, not timed out one at a
    time."""

    class _AllHangingVideoCapture:
        def __init__(self, index: int):
            time.sleep(5.0)

        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(devices_module, "cv2", type("cv2", (), {"VideoCapture": _AllHangingVideoCapture}))
    monkeypatch.setattr(devices_module, "PROBE_TIMEOUT_S", 0.3)

    start = time.monotonic()
    result = devices_module._probe_indices()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"scan took {elapsed:.2f}s -- looks like timeouts summed instead of sharing a deadline"
    assert result == []
