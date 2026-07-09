"""Reflows a set of "card" widgets (QGroupBox sections) into 1-3 columns
depending on available width, so a settings-style dialog makes sense both
on the Pi's narrow touchscreen and stretched across a wide desktop dev
window -- a single vertical stack of full-width forms looks fine at 480px
but wastes most of the row at 1600px, with labels and fields stranded far
apart from each other.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QWidget

# Content-width breakpoints (px) for column count -- measured empirically
# against this dialog's actual widest card (~680px: AI TRACKING/STREAMING
# ENCODER/FINISH ZONE all run long), not a guessed round number. Cards keep
# their full touch/glove-friendly field width (design doc 11.1) rather than
# being squeezed to fit more columns, so a breakpoint is only crossed once
# N columns of the *widest* card actually fit -- e.g. a 1366px laptop still
# gets 1 column (safe default), a 1920px+ desktop monitor gets 2, and only
# a genuinely wide/ultrawide desktop gets 3. The scroll area has no
# horizontal scrollbar (see __init__), so understating a breakpoint clips
# a card off-screen instead of just looking a bit sparse -- verified via
# scratch rendering at 800x480 / 1024x600 / 1366x768 / 1920x1080 / 2560x1440
# before landing on these numbers.
#
# Primary target hardware: Waveshare 13.3" 2K AMOLED (2560x1440 physical,
# capacitive touch, USB-C/HDMI). With config/config.yaml's display.ui_scale
# (QT_SCALE_FACTOR) at 1.5, Qt widget geometry on that panel is ~1707x960
# logical px -- lands in the 2-column bucket below with no overflow
# (re-verified by rendering at exactly that logical size).
_BREAKPOINTS = (1420, 2130)


class ResponsiveGroupGrid(QWidget):
    """Holds cards added via `add_card` and reflows them on resize.

    Placement is a light masonry (greedy shortest-column), not a strict
    row-major grid: settings cards vary a lot in height (CONNECTION is a
    few rows, AI TRACKING is a dozen), and row-major placement would strand
    a short card next to a tall one, leaving one column full of dead space
    for the rest of the dialog.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setSpacing(14)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._cards: list[QWidget] = []
        self._columns = 0

    def add_card(self, widget: QWidget) -> None:
        self._cards.append(widget)
        self._relayout(force=True)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._relayout()

    @staticmethod
    def _columns_for_width(width: int) -> int:
        if width < _BREAKPOINTS[0]:
            return 1
        if width < _BREAKPOINTS[1]:
            return 2
        return 3

    def _relayout(self, force: bool = False) -> None:
        columns = max(1, self._columns_for_width(self.width()))
        if columns == self._columns and not force:
            return  # cheap no-op on every resize tick until a breakpoint is actually crossed
        self._columns = columns
        while self._grid.count():
            self._grid.takeAt(0)
        column_heights = [0] * columns
        column_rows = [0] * columns
        for widget in self._cards:
            col = column_heights.index(min(column_heights))
            self._grid.addWidget(widget, column_rows[col], col, alignment=Qt.AlignmentFlag.AlignTop)
            column_rows[col] += 1
            column_heights[col] += widget.sizeHint().height() + self._grid.spacing()
        for col in range(columns):
            self._grid.setColumnStretch(col, 1)
