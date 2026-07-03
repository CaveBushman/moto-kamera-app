"""Soft drop shadows for the glass-card look (theme.py) -- Qt stylesheets
have no box-shadow/backdrop-filter equivalent, so real depth has to be
added in code via QGraphicsDropShadowEffect instead of CSS.
"""
from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QWidget


def apply_glass_shadow(widget: QWidget, blur_radius: int = 36, y_offset: int = 10, alpha: int = 150) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur_radius)
    effect.setOffset(0, y_offset)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)
