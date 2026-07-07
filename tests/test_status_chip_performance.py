from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from motocam.ui.widgets.top_bar import StatusChip

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        _APP = QApplication([])
        app = _APP
    return app


def test_status_chip_skips_repaint_for_identical_state():
    _app()
    chip = StatusChip("FPS")

    renders = 0

    def render_probe() -> None:
        nonlocal renders
        renders += 1

    chip._render = render_probe  # type: ignore[method-assign]

    chip.set_state("ok", "FPS 30")
    chip.set_state("ok", "FPS 30")
    assert renders == 1

    chip.set_state("warn", "FPS 12")
    assert renders == 2
