"""Camera control panel: exposure/status readout + record control."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from motocam.ui.widgets.metric_tile import MetricTile

METRIC_AREA_HEIGHT = 132
BUTTON_ROW_HEIGHT = 58


class CameraPanel(QGroupBox):
    record_toggled = pyqtSignal(bool)
    autofocus_requested = pyqtSignal()

    def __init__(self):
        super().__init__("CAMERA")
        self.setObjectName("cameraPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setSpacing(8)

        metrics_area = QWidget()
        metrics_area.setFixedHeight(METRIC_AREA_HEIGHT)
        metrics_grid = QGridLayout(metrics_area)
        metrics_grid.setContentsMargins(0, 0, 0, 0)
        metrics_grid.setHorizontalSpacing(8)
        metrics_grid.setVerticalSpacing(8)

        self.iso_tile = MetricTile("ISO")
        self.wb_tile = MetricTile("WB")
        self.shutter_tile = MetricTile("SHUTTER")
        self.iris_tile = MetricTile("IRIS")
        self.fps_tile = MetricTile("FPS")
        self.media_tile = MetricTile("MEDIA")

        for i, w in enumerate((self.iso_tile, self.wb_tile, self.shutter_tile)):
            metrics_grid.addWidget(w, 0, i)
        for i, w in enumerate((self.iris_tile, self.fps_tile, self.media_tile)):
            metrics_grid.addWidget(w, 1, i)
        layout.addWidget(metrics_area)

        buttons_area = QWidget()
        buttons_area.setFixedHeight(BUTTON_ROW_HEIGHT)
        button_row = QHBoxLayout(buttons_area)
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)

        self.rec_button = QPushButton("REC")
        self.rec_button.setObjectName("recordButton")
        self.rec_button.setCheckable(True)
        self.rec_button.setFixedHeight(BUTTON_ROW_HEIGHT)
        self.rec_button.toggled.connect(self.record_toggled.emit)
        button_row.addWidget(self.rec_button, stretch=1)

        self.af_button = QPushButton("AF")
        self.af_button.setObjectName("autofocusButton")
        self.af_button.setFixedHeight(BUTTON_ROW_HEIGHT)
        self.af_button.clicked.connect(self.autofocus_requested.emit)
        button_row.addWidget(self.af_button, stretch=1)
        layout.addWidget(buttons_area)

    def update_state(self, state) -> None:
        self.iso_tile.set_value(str(state.iso) if state.iso is not None else "--")
        self.wb_tile.set_value(f"{state.white_balance} K" if state.white_balance is not None else "--")
        self.shutter_tile.set_value(state.shutter or "--")
        self.iris_tile.set_value(state.iris or "--")
        self.fps_tile.set_value(f"{state.fps:.0f}" if state.fps is not None else "--")
        remaining = state.media_remaining_min
        self.media_tile.set_value(f"{remaining:.0f} min" if remaining is not None else "--")
        self.rec_button.blockSignals(True)
        self.rec_button.setChecked(state.recording)
        self.rec_button.blockSignals(False)
