"""Diagnostics / service screen (design doc 19.1)."""
from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QFormLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout

from motocam.ui.effects import apply_glass_shadow


class ServiceScreen(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diagnostics")
        self.resize(480, 420)

        group = QGroupBox("DIAGNOSTICS")
        form = QFormLayout(group)
        self.device_label = QLabel("--")
        self.format_label = QLabel("--")
        self.fps_label = QLabel("--")
        self.cpu_temp_label = QLabel("--")
        self.cpu_load_label = QLabel("--")
        self.ram_label = QLabel("--")
        self.net_label = QLabel("--")
        self.gps_label = QLabel("--")
        self.services_label = QLabel("--")

        for label, widget in (
            ("Video device", self.device_label),
            ("Video format", self.format_label),
            ("Video FPS", self.fps_label),
            ("CPU temp", self.cpu_temp_label),
            ("CPU load", self.cpu_load_label),
            ("RAM used", self.ram_label),
            ("Control room link", self.net_label),
            ("GPS fix", self.gps_label),
            ("Services", self.services_label),
        ):
            form.addRow(label, widget)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(group)
        layout.addWidget(close_btn)

        apply_glass_shadow(group, blur_radius=32, y_offset=8, alpha=140)
