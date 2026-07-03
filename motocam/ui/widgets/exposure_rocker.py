"""Horizontal spring-back ISO/gain rocker for the live preview overlay --
the exposure counterpart to the zoom rocker, mirrored to the opposite
(left) side so the operator has one thumb control for framing (zoom) and
one for brightness when riding in and out of tree shade / bright sun.

Unlike the zoom rocker (a continuous velocity), exposure is a *stepped*
value: each flick past the threshold steps ISO by one stop and then must
return toward center before it will step again (hysteresis), so a single
push moves exactly one stop rather than racing through the whole range.
The knob springs back to center on release."""
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

# Flick past this fraction of full travel to trigger a step; must fall back
# below the lower fraction before it will fire again (hysteresis).
_TRIGGER_FRACTION = 0.62
_REARM_FRACTION = 0.28


class ExposureRocker(QWidget):
    """Stepped exposure rocker. Emits `stepped(+1)` (brighter / higher ISO)
    or `stepped(-1)` (darker / lower ISO) once per flick, and `released`
    when the knob springs back."""

    stepped = pyqtSignal(int)
    released = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._dragging = False
        self._active = True
        self._compact = False
        self._size_scale = 1.0
        self._armed = True
        self._value_text = "--"
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

    def set_value_text(self, text: str) -> None:
        """Current ISO shown in the center pill (e.g. "800"). Updated from
        the camera state refresh so the operator sees where they are."""
        text = text or "--"
        if text == self._value_text:
            return
        self._value_text = text
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
        self._armed = True
        self.update()
        self.released.emit()

    def _update_knob(self, x_pos: float) -> None:
        delta = max(-self._max_travel, min(self._max_travel, x_pos - self._center.x()))
        self._knob_x = self._center.x() + delta
        offset = self.offset
        if self._armed and abs(offset) >= _TRIGGER_FRACTION:
            self._armed = False
            self.stepped.emit(1 if offset > 0 else -1)
        elif not self._armed and abs(offset) <= _REARM_FRACTION:
            self._armed = True
        self.update()

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
            # amber knob to distinguish exposure from the cyan zoom rocker
            knob_fill = QColor(255, 176, 32, 238) if self._dragging else QColor(255, 176, 32, 172)
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

        # Top pill shows the live ISO value, not a static title, so the
        # operator reads their exposure at a glance while flicking.
        label_rect = QRectF((self._widget_width - 92) / 2, 0, 92, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 145))
        painter.drawRoundedRect(label_rect, 6, 6)

        painter.setPen(text_color)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, f"ISO {self._value_text}")
        painter.drawText(QRectF(12, self._center.y() - 10, 34, 20), Qt.AlignmentFlag.AlignCenter, "-")
        painter.drawText(QRectF(self._widget_width - 46, self._center.y() - 10, 34, 20), Qt.AlignmentFlag.AlignCenter, "+")

        painter.setPen(QPen(QColor(255, 255, 255, 170), 2))
        painter.setBrush(knob_fill)
        painter.drawEllipse(QPointF(self._knob_x, self._center.y()), self._knob_radius, self._knob_radius)
