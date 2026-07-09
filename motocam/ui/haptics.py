"""Virtual haptic feedback for touch buttons (design doc 11.1 glove use).

The Waveshare touchscreen has no vibration motor, so "haptics" here means
an unmissable *visual* acknowledgment that a tap registered: a bright
flash overlay that fades out over ~150ms on every button press. With
gloves on a vibrating moto in sunlight, the standard Qt :pressed
stylesheet shade is far too subtle to confirm a tap at a glance -- and a
tap the operator isn't sure landed gets tapped again, which for buttons
like record start/stop is an actual double-action hazard, not just a UX
nit.

Implemented as one app-wide event filter (install_haptics) rather than
per-button wiring, so every QAbstractButton in the app -- including ones
added later -- gets the same feedback with zero per-widget code. The
flash is a child overlay of the pressed button (mouse-transparent,
self-destroying), so it never fights layouts, never replaces an existing
QGraphicsEffect on the button, and costs nothing except during the
150ms after an actual press.

If a real vibration actuator is ever added (GPIO haptic driver on the
Pi), trigger it from HapticFeedback.eventFilter alongside the flash --
this module is the single choke point for all button-press feedback.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPropertyAnimation, Qt
from PyQt6.QtWidgets import QAbstractButton, QApplication, QGraphicsOpacityEffect, QWidget

FLASH_DURATION_MS = 150
FLASH_START_OPACITY = 0.55


class _PressFlash(QWidget):
    """Translucent white overlay covering a just-pressed button, fading
    out and deleting itself. Mouse-transparent so it can never swallow
    the release/click event of the press that spawned it."""

    def __init__(self, button: QAbstractButton):
        super().__init__(button)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        # Match the theme's button rounding (8px) so the flash reads as
        # the button lighting up, not a rectangle stamped on top of it.
        self.setStyleSheet("background-color: rgba(255, 255, 255, 255); border-radius: 8px;")
        self.setGeometry(button.rect())

        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(FLASH_START_OPACITY)
        self.setGraphicsEffect(effect)
        self.show()

        self._animation = QPropertyAnimation(effect, b"opacity", self)
        self._animation.setDuration(FLASH_DURATION_MS)
        self._animation.setStartValue(FLASH_START_OPACITY)
        self._animation.setEndValue(0.0)
        self._animation.finished.connect(self.deleteLater)
        self._animation.start()


class HapticFeedback(QObject):
    """App-wide event filter: flashes any QAbstractButton on press."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(obj, QAbstractButton) and obj.isEnabled():
            _PressFlash(obj)
        return False  # observe only -- never consume the event


def install_haptics(app: QApplication) -> HapticFeedback:
    """Install the press-flash filter app-wide. Returns the filter object;
    the caller must keep a reference (Qt does not own event filters)."""
    feedback = HapticFeedback(app)
    app.installEventFilter(feedback)
    return feedback
