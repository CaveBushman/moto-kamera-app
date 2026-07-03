"""Semi-transparent on-screen joystick for manual pan/tilt (design doc 7.2
Manual, 12.1 joystick) -- an actual persistent thumb control anchored to
the bottom-right of the preview, rather than a drag-anywhere gesture, so
the operator always knows where it is without looking.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget

# Sized for thumb operation with riding gloves on (design doc 11.1 "velka
# tlacitka" / 12.3 glove-friendly controls) -- this is deliberately much
# bigger than a typical mobile-game joystick.
DEFAULT_WIDGET_SIZE = 206
DEFAULT_BASE_RADIUS = 90
DEFAULT_KNOB_RADIUS = 36
COMPACT_WIDGET_SIZE = 156
COMPACT_BASE_RADIUS = 68
COMPACT_KNOB_RADIUS = 28


class VirtualJoystick(QWidget):
    """Normalized (dx, dy) in [-1, 1] while dragging; springs back to
    (0, 0) and emits `released` on touch-up."""

    moved = pyqtSignal(float, float)
    released = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._dragging = False
        self._active = True
        self._compact = False
        self._size_scale = 1.0
        self._apply_geometry()

    def set_compact(self, compact: bool) -> None:
        if compact == self._compact:
            return
        self._compact = compact
        self._apply_geometry()
        self.update()

    def set_size_scale(self, scale: float) -> None:
        """Operator-tunable enlargement of this thumb control (design doc
        11.1/12.3 glove-friendly sizing) -- see display.controls_scale."""
        scale = max(0.5, min(3.0, scale))
        if scale == self._size_scale:
            return
        self._size_scale = scale
        self._apply_geometry()
        self.update()

    def _apply_geometry(self) -> None:
        if self._compact:
            size, base, knob = COMPACT_WIDGET_SIZE, COMPACT_BASE_RADIUS, COMPACT_KNOB_RADIUS
        else:
            size, base, knob = DEFAULT_WIDGET_SIZE, DEFAULT_BASE_RADIUS, DEFAULT_KNOB_RADIUS
        s = self._size_scale
        self._set_geometry(int(size * s), int(base * s), int(knob * s))

    def set_active(self, active: bool) -> None:
        if active == self._active:
            return
        self._active = active
        if not active and self._dragging:
            self._end_drag()
        self.update()

    @property
    def is_dragging(self) -> bool:
        return self._dragging

    @property
    def offset(self) -> tuple[float, float]:
        dx = (self._knob.x() - self._center.x()) / self._max_travel
        dy = (self._knob.y() - self._center.y()) / self._max_travel
        return dx, dy

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._active:
            return
        self._dragging = True
        self._update_knob(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging:
            return
        self._update_knob(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._end_drag()

    def _end_drag(self) -> None:
        self._dragging = False
        self._knob = QPointF(self._center)
        self.update()
        self.released.emit()

    def _update_knob(self, pos: QPointF) -> None:
        delta = pos - self._center
        distance = math.hypot(delta.x(), delta.y())
        if distance > self._max_travel:
            scale = self._max_travel / distance
            delta = QPointF(delta.x() * scale, delta.y() * scale)
        self._knob = self._center + delta
        self.update()
        dx, dy = self.offset
        self.moved.emit(dx, dy)

    def _set_geometry(self, widget_size: int, base_radius: int, knob_radius: int) -> None:
        self._widget_size = widget_size
        self._base_radius = base_radius
        self._knob_radius = knob_radius
        self._max_travel = base_radius - 18
        self.setFixedSize(widget_size, widget_size)
        self._center = QPointF(widget_size / 2, widget_size / 2)
        self._knob = QPointF(self._center)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._active:
            base_fill = QColor(8, 11, 17, 205)
            base_border = QColor(255, 255, 255, 110)
            knob_fill = QColor(0, 119, 255, 242) if self._dragging else QColor(0, 119, 255, 178)
            text_color = QColor(255, 255, 255, 160)
        else:
            base_fill = QColor(255, 255, 255, 18)
            base_border = QColor(255, 255, 255, 45)
            knob_fill = QColor(120, 128, 138, 120)
            text_color = QColor(255, 255, 255, 70)

        painter.setPen(QPen(QColor(0, 0, 0, 135), 5))
        painter.setBrush(QColor(0, 0, 0, 80))
        painter.drawEllipse(self._center, self._base_radius + 4, self._base_radius + 4)

        painter.setPen(QPen(base_border, 2))
        painter.setBrush(base_fill)
        painter.drawEllipse(self._center, self._base_radius, self._base_radius)

        painter.setPen(QPen(base_border, 1))
        painter.drawLine(
            QPointF(self._center.x() - self._base_radius + 10, self._center.y()),
            QPointF(self._center.x() + self._base_radius - 10, self._center.y()),
        )
        painter.drawLine(
            QPointF(self._center.x(), self._center.y() - self._base_radius + 10),
            QPointF(self._center.x(), self._center.y() + self._base_radius - 10),
        )

        painter.setPen(QPen(QColor(255, 255, 255, 160), 2))
        painter.setBrush(knob_fill)
        painter.drawEllipse(self._knob, self._knob_radius, self._knob_radius)

        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        label_rect = QRectF((self._widget_size - 82) / 2, 12, 82, 24)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 145))
        painter.drawRoundedRect(label_rect, 6, 6)
        painter.setPen(text_color)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, "PAN/TILT")
