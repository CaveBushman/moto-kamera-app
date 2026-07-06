import time

import numpy as np
import pytest

from motocam.video.preview_relay import PreviewRelayEncoder, preview_interval_s, preview_jpeg_quality, scaled_preview_size


def test_preview_relay_limits_are_clamped():
    assert preview_interval_s(5) == 0.2
    assert preview_interval_s(0) == 2.0
    assert preview_interval_s(120) == 1 / 30.0
    assert preview_jpeg_quality(10) == 20
    assert preview_jpeg_quality(95) == 90


def test_scaled_preview_size_keeps_aspect_ratio():
    assert scaled_preview_size(1920, 1080, 960) == (960, 540)
    assert scaled_preview_size(640, 360, 960) == (640, 360)
    assert scaled_preview_size(1920, 1080, 0) == (1920, 1080)


def test_preview_encoder_emits_scaled_jpeg_off_thread(qapp):
    worker = PreviewRelayEncoder(max_width=80, jpeg_quality=50)
    received: list[tuple[bytes, int, int, int]] = []
    worker.encoded.connect(lambda *args: received.append(args))
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:] = (20, 80, 140)
    worker.start()
    try:
        worker.submit(frame)
        _spin_until(qapp, lambda: bool(received), timeout_s=2.0)
    finally:
        worker.stop()

    jpeg, width, height, byte_count = received[-1]
    assert width == 80
    assert height == 60
    assert byte_count == len(jpeg)
    assert jpeg.startswith(b"\xff\xd8")


def _spin_until(qapp, predicate, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    qapp.processEvents()


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
