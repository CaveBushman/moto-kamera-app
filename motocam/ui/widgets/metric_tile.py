"""Compact metric tile used by the cockpit panels."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class MetricTile(QFrame):
    def __init__(self, title: str, value: str = "--", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("metricTile")
        self.setMinimumHeight(52)
        self.setMaximumHeight(56)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(2)

        self.title_label = QLabel(title.upper())
        self.title_label.setProperty("class", "metricTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.value_label = QLabel(value)
        self.value_label.setProperty("class", "metricValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setMinimumWidth(70)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)
