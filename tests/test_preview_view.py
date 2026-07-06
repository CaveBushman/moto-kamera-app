"""Smoke tests for the preview draw path. The tracking rectangle is now
drawn on the scaled pixmap (not a full-res copy of every frame); these
guard that the render path survives frames with and without a box, an
odd frame size, and a resize, always producing a label-sized pixmap."""
from __future__ import annotations

import numpy as np
import pytest

from motocam.ui.widgets import preview_view as preview_view_module


def _frame(w: int, h: int) -> np.ndarray:
    return (np.random.rand(h, w, 3) * 255).astype("uint8")


def test_update_frame_without_bbox_sets_pixmap(preview):
    preview.set_bbox(None, "idle")
    preview.update_frame(_frame(1920, 1080))
    pm = preview._image_label.pixmap()
    assert pm is not None and not pm.isNull()
    assert pm.width() <= preview._image_label.width()


def test_update_frame_with_bbox_does_not_crash(preview):
    preview.set_bbox((760, 420, 400, 300), "locked")
    preview.update_frame(_frame(1920, 1080))
    assert not preview._image_label.pixmap().isNull()
    # the clean full-res pixmap is retained for rescaling (no baked-in box)
    assert preview._last_pixmap.width() == 1920


def test_render_survives_resize_and_reuses_last_frame(preview):
    preview.set_bbox((10, 10, 50, 50), "weak")
    preview.update_frame(_frame(640, 480))
    preview.resize(800, 600)
    preview._render_scaled()
    assert not preview._image_label.pixmap().isNull()


def test_render_scaled_noop_before_first_frame(preview_no_frame):
    # Must not raise when there is no pixmap yet (e.g. an early resize).
    preview_no_frame._render_scaled()
    assert preview_no_frame._last_pixmap is None


def test_update_frame_throttles_expensive_qt_render(preview, monkeypatch):
    preview.set_render_options(max_fps=10, smooth_scaling=False)
    times = iter([10.00, 10.05, 10.11])
    monkeypatch.setattr(preview_view_module.time, "monotonic", lambda: next(times))
    renders = 0

    def render_probe() -> None:
        nonlocal renders
        renders += 1

    preview._render_scaled = render_probe  # type: ignore[method-assign]

    preview.update_frame(_frame(640, 480))
    preview.update_frame(_frame(640, 480))
    preview.update_frame(_frame(640, 480))

    assert renders == 2


@pytest.fixture
def preview(qapp):
    from motocam.ui.widgets.preview_view import PreviewView

    pv = PreviewView()
    pv.resize(1280, 720)
    pv.show()
    qapp.processEvents()
    return pv


@pytest.fixture
def preview_no_frame(qapp):
    from motocam.ui.widgets.preview_view import PreviewView

    return PreviewView()


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
