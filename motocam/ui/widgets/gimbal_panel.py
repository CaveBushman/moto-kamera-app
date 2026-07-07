"""Gimbal panel: orientation readout + lock/home controls."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from motocam.ui.widgets.metric_tile import MetricTile
from motocam.ui.widgets.camera_panel import BUTTON_ROW_HEIGHT, METRIC_AREA_HEIGHT


class GimbalPanel(QGroupBox):
    """Cockpit panel showing GimbalController orientation as MetricTiles,
    plus the recenter/lock buttons wired up by ui/main_window.py."""

    home_requested = pyqtSignal()
    lock_toggled = pyqtSignal(bool)

    def __init__(self):
        super().__init__("GIMBAL")
        self.setObjectName("gimbalPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setSpacing(8)

        metrics_area = QWidget()
        metrics_area.setFixedHeight(METRIC_AREA_HEIGHT)
        metrics_grid = QGridLayout(metrics_area)
        metrics_grid.setContentsMargins(0, 0, 0, 0)
        metrics_grid.setHorizontalSpacing(8)
        metrics_grid.setVerticalSpacing(8)

        self.pan_tile = MetricTile("PAN", "0.0°")
        self.tilt_tile = MetricTile("TILT", "0.0°")
        self.roll_tile = MetricTile("ROLL", "0.0°")
        for i, w in enumerate((self.pan_tile, self.tilt_tile, self.roll_tile)):
            metrics_grid.addWidget(w, 0, i)
        metrics_grid.setRowStretch(1, 1)
        layout.addWidget(metrics_area)

        buttons_area = QWidget()
        buttons_area.setFixedHeight(BUTTON_ROW_HEIGHT)
        button_row = QHBoxLayout(buttons_area)
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)

        self.lock_button = QPushButton("LOCK")
        self.lock_button.setCheckable(True)
        self.lock_button.setFixedHeight(BUTTON_ROW_HEIGHT)
        self.lock_button.toggled.connect(self.lock_toggled.emit)
        button_row.addWidget(self.lock_button, stretch=1)

        self.home_button = QPushButton("HOME")
        self.home_button.setFixedHeight(BUTTON_ROW_HEIGHT)
        self.home_button.clicked.connect(self.home_requested.emit)
        button_row.addWidget(self.home_button, stretch=1)

        self.connection_label = QLabel("DISCONNECTED")
        self.connection_label.setProperty("class", "connectionBadge bad")
        self.connection_label.setFixedHeight(BUTTON_ROW_HEIGHT)
        button_row.addWidget(self.connection_label, stretch=1)
        layout.addWidget(buttons_area)

    def update_orientation(self, pan: float, tilt: float, roll: float) -> None:
        self.pan_tile.set_value(f"{pan:.1f}°")
        self.tilt_tile.set_value(f"{tilt:.1f}°")
        self.roll_tile.set_value(f"{roll:.1f}°")

    def set_connected(self, connected: bool) -> None:
        self.connection_label.setText("CONNECTED" if connected else "DISCONNECTED")
        self.connection_label.setProperty("class", "connectionBadge ok" if connected else "connectionBadge bad")
        self.connection_label.style().unpolish(self.connection_label)
        self.connection_label.style().polish(self.connection_label)
