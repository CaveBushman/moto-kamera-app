"""Top status bar: CAM | GIMBAL | GPS | LINK | REC | AI | FPS | LATENCY (design doc 11)."""
from __future__ import annotations

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from motocam.ui.widgets.logo import LogoWidget

BLINK_INTERVAL_MS = 500


class StatusChip(QLabel):
    def __init__(self, label: str):
        super().__init__(label)
        self._label = label
        self.setProperty("class", "statusChip statusIdle")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(40)
        # Fluid: chips share the bar width equally and grow with the
        # window instead of a fixed 70px that stayed tiny on a big screen.
        # A minimum keeps them tappable when the bar is narrow.
        self.setMinimumWidth(58)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setTextFormat(Qt.TextFormat.RichText)
        self._state = ""
        self._value = ""
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
        class_name = f"statusChip {css_class}"
        value = self._value_text(text if text is not None else self._label)
        if state == self._state and value == self._value and self.property("class") == class_name:
            return
        class_changed = self.property("class") != class_name
        self._state = state
        self._value = value
        if class_changed:
            self.setProperty("class", class_name)
        self._render()
        if class_changed:
            self.style().unpolish(self)
            self.style().polish(self)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._render()  # font sizes track the chip's real height

    def _render(self) -> None:
        # Text sizes track the chip's real geometry so the bar reads well
        # at any window size / UI scale. Bounded by BOTH height and width:
        # a tall-but-narrow chip must shrink its font to the width or the
        # value string ("SYNTH") clips horizontally.
        h = max(self.height(), self.minimumHeight())
        # Subtract the QSS horizontal padding (10px each side, theme.py
        # .statusChip) so the width budget is the real text area.
        w = max(self.width() - 24, 24)
        # ~0.72 px advance per char at this bold weight (measured against
        # clipping on "SYNTH"/"GIMBAL"); size to the longer of label/value
        # so neither overflows the chip width.
        longest = max(len(self._label), len(self._value), 1)
        width_cap = int(w / (longest * 0.72))
        label_px = max(8, min(int(h * 0.22), width_cap))
        value_px = max(11, min(int(h * 0.36), width_cap))
        self.setText(
            f"<div style='font-size:{label_px}px; font-weight:800; letter-spacing:0; color:#8f98a8;'>{self._label}</div>"
            f"<div style='font-size:{value_px}px; font-weight:950; color:#eef1f6;'>{self._value}</div>"
        )

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

        # "VIDEO" = the UVC/V4L2 preview feed (grabber); "BMD" = the
        # Blackmagic Camera Control REST link (PYXIS, Studio Cameras, Micro
        # Studio Camera 4K G2, ... -- same generic REST API, so the chip
        # isn't named after one specific camera). They are independent --
        # you can have camera control up with no video grabber, or vice
        # versa -- so they get separate chips (previously only video had
        # one, labelled "CAM", which read as the camera control link and
        # caused confusion).
        self.cam_chip = StatusChip("VIDEO")
        self.bmd_chip = StatusChip("BMD")
        self.gimbal_chip = StatusChip("GIMBAL")
        self.gps_chip = StatusChip("GPS")
        self.net_chip = StatusChip("CTRL")
        self.switcher_chip = StatusChip("ATEM")
        self.rec_chip = StatusChip("REC")
        self.ai_chip = StatusChip("AI")
        self.fps_chip = StatusChip("FPS")
        self.latency_chip = StatusChip("LAT")

        for chip in (
            self.cam_chip, self.bmd_chip, self.gimbal_chip, self.gps_chip, self.net_chip,
            self.switcher_chip, self.rec_chip, self.ai_chip, self.fps_chip, self.latency_chip,
        ):
            layout.addWidget(chip, stretch=1)  # chips share the width evenly and fill the bar
        layout.addWidget(LogoWidget(height=30))

    def set_unit_id(self, unit_id: str) -> None:
        prefix, _, number = unit_id.rpartition("-")
        text = f"MOTO {number}" if prefix == "moto" and number.isdigit() else unit_id.upper()
        self.unit_label.setText(
            "<div style='font-size:10px; font-weight:850; color:#c8d0dd;'>UNIT</div>"
            f"<div style='font-size:24px; font-weight:1000; color:white;'>{text}</div>"
        )
