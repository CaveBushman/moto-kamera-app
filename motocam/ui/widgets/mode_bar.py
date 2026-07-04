"""Bottom mode bar: MANUAL | AI ASSIST | FULL AI | LOCK | AF | RESET | OVERRIDE.

AF triggers camera autofocus (not a mode -- a one-shot action, like
RESET). HOME was removed; RESET now re-centres the gimbal in its place.

MANUAL OVERRIDE (design doc 12.3) is deliberately the largest, reddest
button and wired directly to an immediate AI-off call in main_window.py
-- it must never depend on the AI/tracking pipeline being healthy to work.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget

from motocam.core.protocol import OperatingMode


class ModeBar(QWidget):
    mode_selected = pyqtSignal(OperatingMode)
    reset_requested = pyqtSignal()
    autofocus_requested = pyqtSignal()
    manual_override = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("modeBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(10)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        self._mode_buttons: dict[OperatingMode, QPushButton] = {}
        for mode, label in (
            (OperatingMode.MANUAL, "MANUAL"),
            (OperatingMode.AI_ASSIST, "AI ASSIST"),
            (OperatingMode.FULL_AI, "FULL AI"),
            (OperatingMode.LOCK, "LOCK"),
        ):
            btn = QPushButton(label)
            btn.setMinimumWidth(112)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, m=mode: self.mode_selected.emit(m))
            self._group.addButton(btn)
            self._mode_buttons[mode] = btn
            layout.addWidget(btn)
        self._mode_buttons[OperatingMode.MANUAL].setChecked(True)

        # AF: one-shot camera autofocus, always on the main screen (not
        # tucked in the collapsible CAM/GIMBAL HUD). An action, not a mode,
        # so it is not part of the exclusive mode group.
        af_btn = QPushButton("AF")
        af_btn.setObjectName("afButton")
        af_btn.setMinimumWidth(96)
        af_btn.clicked.connect(self.autofocus_requested.emit)
        layout.addWidget(af_btn)

        reset_btn = QPushButton("RESET")
        reset_btn.setMinimumWidth(108)
        reset_btn.clicked.connect(self.reset_requested.emit)
        layout.addWidget(reset_btn)

        override_btn = QPushButton("MANUAL\nOVERRIDE")
        override_btn.setObjectName("manualOverride")
        override_btn.setMinimumWidth(170)
        override_btn.clicked.connect(self._on_override)
        layout.addWidget(override_btn)

    def _on_override(self) -> None:
        self._mode_buttons[OperatingMode.MANUAL].setChecked(True)
        self.manual_override.emit()

    def set_mode(self, mode: OperatingMode) -> None:
        btn = self._mode_buttons.get(mode)
        if btn is not None:
            btn.setChecked(True)
