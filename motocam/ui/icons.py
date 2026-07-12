"""Vector pictograms drawn with QPainter (no font, no files).

The Ri5 has no color-emoji font (emoji in labels render as tofu boxes --
that's why the Settings section emoji were removed), and shipping SVG
assets would mean a QtSvg dependency plus files to keep in sync. Drawing
the handful of pictograms the HUD needs directly with QPainter avoids
both: they render identically on macOS and the Pi, scale crisply on any
DPI (drawn at the device pixel ratio), and recolor per state via a plain
QColor argument.

All icons share a 24x24 design grid and a line style (round caps, ~2px
stroke at 24px) so they read as one family next to each other in the
pill row.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

DEFAULT_COLOR = QColor("#eef1f6")
GRID = 24.0  # design-space size all draw functions target


def _make_pixmap(size: int, dpr: float, draw_fn, color: QColor) -> QPixmap:
    pixmap = QPixmap(int(size * dpr), int(size * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / GRID, size / GRID)
    draw_fn(painter, color)
    painter.end()
    return pixmap


def _icon(draw_fn, color: QColor | None = None, size: int = 24) -> QIcon:
    color = color or DEFAULT_COLOR
    icon = QIcon()
    # Two DPRs cover both the Pi touchscreen (1x) and macOS retina dev (2x).
    for dpr in (1.0, 2.0):
        icon.addPixmap(_make_pixmap(size, dpr, draw_fn, color))
    return icon


def _pen(color: QColor, width: float = 2.0) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _draw_rider(painter: QPainter, color: QColor, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> None:
    """One side-view cyclist on the 24x24 grid: two wheels, a frame line,
    a leaning torso and a head dot -- minimal, but unmistakably a bike at
    pill-icon sizes."""
    painter.save()
    painter.translate(dx, dy)
    painter.scale(scale, scale)
    painter.setPen(_pen(color, 1.8))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QPointF(6.0, 17.5), 4.0, 4.0)  # rear wheel
    painter.drawEllipse(QPointF(18.0, 17.5), 4.0, 4.0)  # front wheel
    # frame: rear hub -> bottom bracket -> front hub, plus seat tube
    painter.drawPolyline([QPointF(6.0, 17.5), QPointF(10.5, 11.0), QPointF(14.5, 14.0), QPointF(18.0, 17.5)])
    painter.drawLine(QPointF(10.5, 11.0), QPointF(9.0, 9.0))  # seat post
    # rider: torso leaning to the bars, head dot
    painter.drawLine(QPointF(9.0, 9.0), QPointF(15.0, 6.5))
    painter.drawLine(QPointF(15.0, 6.5), QPointF(17.0, 10.5))  # arm to bars
    painter.setBrush(color)
    painter.drawEllipse(QPointF(16.2, 4.0), 1.9, 1.9)  # head
    painter.restore()


def cyclist_icon(color: QColor | None = None) -> QIcon:
    return _icon(lambda p, c: _draw_rider(p, c), color)


def peloton_icon(color: QColor | None = None) -> QIcon:
    """Group of riders: a dimmed rider behind, a full-strength one in
    front -- overlap is what says "group" at 20px, not headcount."""

    def draw(painter: QPainter, color: QColor) -> None:
        shadow = QColor(color)
        shadow.setAlphaF(color.alphaF() * 0.45)
        _draw_rider(painter, shadow, scale=0.82, dx=5.0, dy=-1.0)
        _draw_rider(painter, color, scale=0.92, dx=-1.0, dy=2.2)

    return _icon(draw, color)


def finish_flag_icon(color: QColor | None = None) -> QIcon:
    """Checkered finish flag on a pole."""

    def draw(painter: QPainter, color: QColor) -> None:
        painter.setPen(_pen(color, 2.0))
        painter.drawLine(QPointF(5.0, 3.0), QPointF(5.0, 21.0))  # pole
        painter.setPen(_pen(color, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(5.0, 3.5, 14.0, 9.0))  # flag outline
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        cell_w, cell_h = 14.0 / 4.0, 9.0 / 3.0
        for row in range(3):
            for col in range(4):
                if (row + col) % 2 == 0:
                    painter.drawRect(QRectF(5.0 + col * cell_w, 3.5 + row * cell_h, cell_w, cell_h))

    return _icon(draw, color)


def gear_icon(color: QColor | None = None) -> QIcon:
    """Settings gear: ring, radial teeth, center hole."""

    def draw(painter: QPainter, color: QColor) -> None:
        painter.setPen(_pen(color, 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(12.0, 12.0), 5.2, 5.2)
        painter.drawEllipse(QPointF(12.0, 12.0), 1.8, 1.8)
        painter.save()
        painter.translate(12.0, 12.0)
        for _ in range(8):
            painter.drawLine(QPointF(0.0, -7.0), QPointF(0.0, -9.2))
            painter.rotate(45.0)
        painter.restore()

    return _icon(draw, color)


def camera_icon(color: QColor | None = None) -> QIcon:
    """Camera body + lens, for the CAM/GIMBAL HUD toggle."""

    def draw(painter: QPainter, color: QColor) -> None:
        painter.setPen(_pen(color, 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(3.0, 8.0, 18.0, 11.0), 2.0, 2.0)  # body
        painter.drawPolyline([QPointF(8.0, 8.0), QPointF(9.5, 5.0), QPointF(14.5, 5.0), QPointF(16.0, 8.0)])
        painter.drawEllipse(QPointF(12.0, 13.5), 3.4, 3.4)  # lens

    return _icon(draw, color)
