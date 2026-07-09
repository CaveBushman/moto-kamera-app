"""Diagnostics / service screen (design doc 19.1).

Read-only snapshot of subsystem health for field debugging -- no hardware
access of its own. MainWindow.refresh() feeds it the same values already
computed for the control-room telemetry tick (_telemetry_tick) and the UI
watchdog's stall-log context (pipeline_diagnostics_summary), so this can
never drift from what the rest of the app already believes is true.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QFormLayout, QGroupBox, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from motocam.ui.effects import apply_glass_shadow


class ServiceScreen(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diagnostics")
        self.resize(560, 560)

        group = QGroupBox("🔧 DIAGNOSTICS")
        form = QFormLayout(group)
        self.device_label = QLabel("--")
        self.format_label = QLabel("--")
        self.fps_label = QLabel("--")
        self.cpu_temp_label = QLabel("--")
        self.cpu_load_label = QLabel("--")
        self.ram_label = QLabel("--")
        self.net_label = QLabel("--")
        self.net_label.setProperty("class", "connectionBadge")
        # Fixed/hugging width: QFormLayout's default field growth would
        # otherwise stretch this into a full-row bar (fine in gimbal_panel's
        # button row, wrong for a compact status badge in a label list).
        self.net_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.net_label.setMinimumWidth(90)
        self.gps_label = QLabel("--")
        self.services_label = QLabel("--")
        self.services_label.setWordWrap(True)

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

    def refresh(
        self,
        *,
        video_device: object,
        video_format: str,
        video_fps: float,
        cpu_temp_c: float | None,
        cpu_load_pct: float | None,
        ram_used_pct: float | None,
        link_connected: bool,
        gps_text: str,
        services_summary: str,
    ) -> None:
        self.device_label.setText(str(video_device))
        self.format_label.setText(video_format)
        self.fps_label.setText(f"{video_fps:.1f}")
        self.cpu_temp_label.setText(f"{cpu_temp_c:.1f}°C" if cpu_temp_c is not None else "--")
        self.cpu_load_label.setText(f"{cpu_load_pct:.0f}%" if cpu_load_pct is not None else "--")
        self.ram_label.setText(f"{ram_used_pct:.0f}%" if ram_used_pct is not None else "--")
        self.net_label.setText("UP" if link_connected else "DOWN")
        self.net_label.setProperty("class", "connectionBadge " + ("ok" if link_connected else "bad"))
        self.net_label.style().unpolish(self.net_label)
        self.net_label.style().polish(self.net_label)
        self.gps_label.setText(gps_text)
        self.services_label.setText(services_summary)
