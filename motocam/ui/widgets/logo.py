"""Federation logo badge (design doc branding). Always scaled with
`Qt.AspectRatioMode.KeepAspectRatio` -- never stretched to fill a target
size -- since the source is a wide wordmark (~5.9:1) that would look
obviously wrong squashed into a square-ish slot.

The source PNG has a fully transparent background and dark navy/red/cyan
artwork (see assets/logo_ccf/), so it's mounted on a small white plate;
without one it would nearly disappear against this app's dark theme.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel

logger = logging.getLogger("motocam.ui")

LOGO_PATH = Path(__file__).resolve().parents[3] / "assets" / "logo_ccf" / "CSC Logo Horizontal EN RGB.png"


class LogoWidget(QLabel):
    def __init__(self, height: int = 30, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "background-color: white; border-radius: 8px; padding: 5px 12px;"
        )

        pixmap = QPixmap(str(LOGO_PATH))
        if pixmap.isNull():
            logger.warning("Logo asset not found at %s", LOGO_PATH)
            self.hide()
            return

        self.setPixmap(
            pixmap.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation)
        )
