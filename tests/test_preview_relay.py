from motocam.video.preview_relay import preview_interval_s, preview_jpeg_quality, scaled_preview_size


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
