"""Push-to-talk button, mirrored bottom-left of the preview opposite the
manual pan/tilt joystick (bottom-right) -- same big glove-sized glass
treatment, same "always in the same spot" reasoning as the joystick."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget

DEFAULT_WIDGET_SIZE = 108
COMPACT_WIDGET_SIZE = 84


class PTTButton(QWidget):
    pressed = pyqtSignal()
    released = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._talking = False
        self._available = True
        self._compact = False
        self._widget_size = DEFAULT_WIDGET_SIZE
        self.setFixedSize(DEFAULT_WIDGET_SIZE, DEFAULT_WIDGET_SIZE)

    def set_compact(self, compact: bool) -> None:
        if compact == self._compact:
            return
        self._compact = compact
        self._widget_size = COMPACT_WIDGET_SIZE if compact else DEFAULT_WIDGET_SIZE
        self.setFixedSize(self._widget_size, self._widget_size)
        self.update()

    def set_available(self, available: bool) -> None:
        self._available = available
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._available:
            return
        self._talking = True
        self.update()
        self.pressed.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._talking:
            return
        self._talking = False
        self.update()
        self.released.emit()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.rect().center())
        radius = self._widget_size / 2 - 6

        if not self._available:
            fill, border, text_color, label = QColor(255, 255, 255, 18), QColor(255, 255, 255, 45), "#8b93a1", "NO MIC"
        elif self._talking:
            fill, border, text_color, label = QColor(220, 38, 38, 232), QColor(255, 255, 255, 190), "white", "TALK"
        else:
            fill, border, text_color, label = QColor(12, 15, 22, 215), QColor(255, 255, 255, 110), "#eef1f6", "PTT"

        painter.setPen(QPen(QColor(0, 0, 0, 140), 5))
        painter.setBrush(QColor(0, 0, 0, 90))
        painter.drawEllipse(center, radius + 4, radius + 4)

        painter.setPen(QPen(border, 3))
        painter.setBrush(fill)
        painter.drawEllipse(center, radius, radius)

        painter.setPen(QColor(text_color))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10 if self._compact else 12)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, label)
