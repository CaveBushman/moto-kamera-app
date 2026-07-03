"""Top status bar: CAM | GIMBAL | GPS | LINK | REC | AI | FPS | LATENCY (design doc 11)."""
from __future__ import annotations

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from motocam.ui.widgets.logo import LogoWidget

BLINK_INTERVAL_MS = 500


class StatusChip(QLabel):
    def __init__(self, label: str):
        super().__init__(label)
        self._label = label
        self.setProperty("class", "statusChip statusIdle")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(46)
        self.setFixedWidth(70)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.set_state("idle", "--")

        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_tick)
        self._blink_phase = True
        self._blink_text = ""
        self._blink_on_state = "bad"
        self._blink_off_state = "idle"

    def set_state(self, state: str, text: str | None = None) -> None:
        css_class = {
            "ok": "statusOk",
            "warn": "statusWarn",
            "bad": "statusBad",
            "idle": "statusIdle",
        }.get(state, "statusIdle")
        self.setProperty("class", f"statusChip {css_class}")
        value = self._value_text(text if text is not None else self._label)
        self.setText(
            f"<div style='font-size:9px; font-weight:800; letter-spacing:0; color:#8f98a8;'>{self._label}</div>"
            f"<div style='font-size:14px; font-weight:950; color:#eef1f6;'>{value}</div>"
        )
        self.style().unpolish(self)
        self.style().polish(self)

    def set_blinking(self, blinking: bool, text: str = "", on_state: str = "bad", off_state: str = "idle") -> None:
        """Alternates between on_state/off_state every BLINK_INTERVAL_MS --
        used for REC while actively recording, the one indicator on this
        screen where "attention, this is live" is the whole point."""
        if blinking:
            self._blink_text, self._blink_on_state, self._blink_off_state = text, on_state, off_state
            if not self._blink_timer.isActive():
                self._blink_phase = True
                self.set_state(on_state, text)
                self._blink_timer.start(BLINK_INTERVAL_MS)
        else:
            self._blink_timer.stop()

    def _blink_tick(self) -> None:
        self._blink_phase = not self._blink_phase
        state = self._blink_on_state if self._blink_phase else self._blink_off_state
        self.set_state(state, self._blink_text)

    def _value_text(self, text: str) -> str:
        cleaned = text.replace("●", "").strip()
        label = self._label
        if cleaned.upper().startswith(label.upper()):
            cleaned = cleaned[len(label):].strip()
        aliases = {
            "CONNECTED": "OK",
            "DISCONNECTED": "DOWN",
            "SYNTHETIC": "SYNTH",
            "RECONNECTING": "RECON",
            "READY": "RDY",
            "LOCKED": "LOCK",
            "MANUAL_REQUIRED": "MANUAL",
            "PREVIEW": "PREV",
        }
        return aliases.get(cleaned.upper(), cleaned or label)


class TopBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("topBar")
        self.setMinimumHeight(76)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Which physical bike this unit is -- on a kiosk-style touchscreen
        # with no window title bar visible, this is the only on-screen way
        # to tell units apart when several are built/serviced side by side.
        self.unit_label = QLabel()
        self.unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unit_label.setObjectName("unitBadge")
        self.unit_label.setFixedWidth(140)
        self.unit_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.unit_label)
        self.set_unit_id("moto-1")

        self.cam_chip = StatusChip("CAM")
        self.gimbal_chip = StatusChip("GIMBAL")
        self.gps_chip = StatusChip("GPS")
        self.net_chip = StatusChip("CTRL")
        self.switcher_chip = StatusChip("ATEM")
        self.rec_chip = StatusChip("REC")
        self.ai_chip = StatusChip("AI")
        self.fps_chip = StatusChip("FPS")
        self.latency_chip = StatusChip("LAT")

        for chip in (
            self.cam_chip, self.gimbal_chip, self.gps_chip, self.net_chip,
            self.switcher_chip, self.rec_chip, self.ai_chip, self.fps_chip, self.latency_chip,
        ):
            layout.addWidget(chip)
        layout.addStretch(1)
        layout.addWidget(LogoWidget(height=24))

    def set_unit_id(self, unit_id: str) -> None:
        prefix, _, number = unit_id.rpartition("-")
        text = f"MOTO {number}" if prefix == "moto" and number.isdigit() else unit_id.upper()
        self.unit_label.setText(
            "<div style='font-size:10px; font-weight:850; color:#c8d0dd;'>UNIT</div>"
            f"<div style='font-size:24px; font-weight:1000; color:white;'>{text}</div>"
        )
