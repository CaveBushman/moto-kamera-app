"""Horizontal spring-back zoom rocker for the live preview overlay."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget

DEFAULT_WIDGET_WIDTH = 206
DEFAULT_WIDGET_HEIGHT = 70
DEFAULT_TRACK_WIDTH = 176
DEFAULT_TRACK_HEIGHT = 32
DEFAULT_KNOB_RADIUS = 25
COMPACT_WIDGET_WIDTH = 156
COMPACT_WIDGET_HEIGHT = 54
COMPACT_TRACK_WIDTH = 132
COMPACT_TRACK_HEIGHT = 26
COMPACT_KNOB_RADIUS = 21


class ZoomRocker(QWidget):
    """Normalized horizontal zoom velocity in [-1, 1].

    Negative means zoom out, positive means zoom in. The knob springs
    back to center on release and emits `released`.
    """

    moved = pyqtSignal(float)
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
        """Operator-tunable enlargement (glove-friendly) -- see
        display.controls_scale."""
        scale = max(0.5, min(3.0, scale))
        if scale == self._size_scale:
            return
        self._size_scale = scale
        self._apply_geometry()
        self.update()

    def _apply_geometry(self) -> None:
        if self._compact:
            w, h, tw, th, knob = (
                COMPACT_WIDGET_WIDTH, COMPACT_WIDGET_HEIGHT,
                COMPACT_TRACK_WIDTH, COMPACT_TRACK_HEIGHT, COMPACT_KNOB_RADIUS,
            )
        else:
            w, h, tw, th, knob = (
                DEFAULT_WIDGET_WIDTH, DEFAULT_WIDGET_HEIGHT,
                DEFAULT_TRACK_WIDTH, DEFAULT_TRACK_HEIGHT, DEFAULT_KNOB_RADIUS,
            )
        s = self._size_scale
        self._set_geometry(int(w * s), int(h * s), int(tw * s), int(th * s), int(knob * s))

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
    def offset(self) -> float:
        return (self._knob_x - self._center.x()) / self._max_travel

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._active:
            return
        self._dragging = True
        self._update_knob(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging:
            return
        self._update_knob(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._end_drag()

    def _end_drag(self) -> None:
        self._dragging = False
        self._knob_x = self._center.x()
        self.update()
        self.released.emit()

    def _update_knob(self, x_pos: float) -> None:
        delta = max(-self._max_travel, min(self._max_travel, x_pos - self._center.x()))
        self._knob_x = self._center.x() + delta
        self.update()
        self.moved.emit(self.offset)

    def _set_geometry(
        self,
        widget_width: int,
        widget_height: int,
        track_width: int,
        track_height: int,
        knob_radius: int,
    ) -> None:
        self._widget_width = widget_width
        self._widget_height = widget_height
        self._track_width = track_width
        self._track_height = track_height
        self._knob_radius = knob_radius
        self._max_travel = (track_width / 2) - 16
        self.setFixedSize(widget_width, widget_height)
        self._center = QPointF(widget_width / 2, widget_height / 2 + 3)
        self._knob_x = self._center.x()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._active:
            base_fill = QColor(8, 11, 17, 210)
            base_border = QColor(255, 255, 255, 105)
            knob_fill = QColor(0, 194, 255, 238) if self._dragging else QColor(0, 194, 255, 172)
            text_color = QColor(255, 255, 255, 185)
        else:
            base_fill = QColor(255, 255, 255, 18)
            base_border = QColor(255, 255, 255, 45)
            knob_fill = QColor(120, 128, 138, 120)
            text_color = QColor(255, 255, 255, 80)

        track = QRectF(
            self._center.x() - self._track_width / 2,
            self._center.y() - self._track_height / 2,
            self._track_width,
            self._track_height,
        )
        painter.setPen(QPen(QColor(0, 0, 0, 135), 5))
        painter.setBrush(QColor(0, 0, 0, 85))
        painter.drawRoundedRect(track.adjusted(-4, -4, 4, 4), self._track_height / 2 + 4, self._track_height / 2 + 4)

        painter.setPen(QPen(base_border, 2))
        painter.setBrush(base_fill)
        painter.drawRoundedRect(track, self._track_height / 2, self._track_height / 2)

        painter.setPen(QPen(base_border, 2))
        painter.drawLine(
            QPointF(self._center.x(), self._center.y() - self._track_height / 2 + 7),
            QPointF(self._center.x(), self._center.y() + self._track_height / 2 - 7),
        )

        label_rect = QRectF((self._widget_width - 68) / 2, 0, 68, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 145))
        painter.drawRoundedRect(label_rect, 6, 6)

        painter.setPen(text_color)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, "ZOOM")
        painter.drawText(QRectF(12, self._center.y() - 10, 34, 20), Qt.AlignmentFlag.AlignCenter, "-")
        painter.drawText(QRectF(self._widget_width - 46, self._center.y() - 10, 34, 20), Qt.AlignmentFlag.AlignCenter, "+")

        painter.setPen(QPen(QColor(255, 255, 255, 170), 2))
        painter.setBrush(knob_fill)
        painter.drawEllipse(QPointF(self._knob_x, self._center.y()), self._knob_radius, self._knob_radius)
