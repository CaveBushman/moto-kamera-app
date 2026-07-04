"""Live preview widget (design doc 10.9 Overlay Renderer + 8.2 tap-to-select).
Manual pan/tilt is a dedicated VirtualJoystick overlay anchored to the
bottom-right corner (see virtual_joystick.py), not a drag-anywhere gesture
on the video itself -- that way the operator always knows where the
control lives without hunting for it."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from motocam.ui.widgets.exposure_rocker import ExposureRocker
from motocam.ui.widgets.ptt_button import PTTButton
from motocam.ui.widgets.virtual_joystick import VirtualJoystick
from motocam.ui.widgets.zoom_rocker import ZoomRocker

JOYSTICK_MARGIN = 24
PTT_MARGIN = 24
ZOOM_MARGIN = 24
ZOOM_JOYSTICK_GAP = 36
HUD_SIDE_MARGIN = 16
HUD_TOP_GAP = 16
CONTROL_HUD_GAP = 16
# Comfortably covers the compact joystick (156px) + its bottom margin +
# CONTROL_HUD_GAP -- the floor the HUD is never allowed to shrink below.
MIN_CONTROL_ZONE = 200


class PreviewView(QFrame):
    """Renders the live frame and reports taps in *frame* pixel coordinates
    (AI target selection). Manual gimbal control is exposed via `.joystick`."""

    tapped = pyqtSignal(int, int)
    cancel_track_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("previewFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(320, 180)
        layout.addWidget(self._image_label)

        self.joystick = VirtualJoystick(self)
        self.joystick.raise_()

        self.zoom_rocker = ZoomRocker(self)
        self.zoom_rocker.raise_()

        self.exposure_rocker = ExposureRocker(self)
        self.exposure_rocker.raise_()

        self.ptt_button = PTTButton(self)
        self.ptt_button.raise_()

        self.talkback_label = QLabel("CONTROL ROOM TALKBACK", self)
        self.talkback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.talkback_label.setStyleSheet(
            "background-color: rgba(94, 21, 26, 226); color: white; "
            "border: 1px solid rgba(255, 255, 255, 145); border-radius: 8px; "
            "padding: 10px 18px; font-size: 18px; font-weight: 950;"
        )
        self.talkback_label.adjustSize()
        self.talkback_label.hide()
        self.talkback_label.raise_()

        self.switcher_label = QLabel("", self)
        self.switcher_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.switcher_label.hide()
        self.switcher_label.raise_()

        self.link_status_label = QLabel("CTRL DOWN · PREVIEW OFF", self)
        self.link_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.link_status_label.hide()
        self.link_status_label.raise_()

        # Ride-safe lockout (design suggestion: no fiddly ISO/shutter/lens
        # taps above riding speed) -- see set_ride_locked. PTT/joystick/
        # zoom stay live always; those are the controls the rig exists for.
        self._ride_locked = False
        self.ride_lock_label = QLabel("SPEED LOCK — SLOW DOWN TO ADJUST", self)
        self.ride_lock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ride_lock_label.setStyleSheet(
            "background-color: rgba(146, 64, 14, 232); color: white; "
            "border: 1px solid rgba(255, 200, 120, 175); border-radius: 8px; "
            "padding: 10px 18px; font-size: 16px; font-weight: 950;"
        )
        self.ride_lock_label.adjustSize()
        self.ride_lock_label.hide()
        self.ride_lock_label.raise_()

        # Only shown while a target is actually locked/weak/lost-and-
        # pending -- tapping a rider had no direct undo short of hitting
        # the big RESET button (which also re-homes the gimbal), or
        # switching modes away and back. This is the small, obvious way
        # to just drop the current target.
        self.cancel_track_button = QPushButton("✕ CANCEL TRACKING", self)
        self.cancel_track_button.setObjectName("cancelTrackButton")
        self.cancel_track_button.adjustSize()
        self.cancel_track_button.hide()
        self.cancel_track_button.raise_()

        self.hud_widget: QWidget | None = None
        # Off by default -- the feed is the centerpiece, and glancing at
        # ISO/shutter/pan-tilt numbers is occasional, not constant, unlike
        # PTT/zoom/joystick. A small always-visible pill brings it back.
        self._hud_visible = False
        self.hud_toggle = QPushButton("CAM / GIMBAL", self)
        self.hud_toggle.setObjectName("hudToggle")
        self.hud_toggle.setCheckable(True)
        self.hud_toggle.setChecked(False)
        self.hud_toggle.toggled.connect(self._on_hud_toggled)
        self.hud_toggle.adjustSize()
        self.hud_toggle.raise_()

        # Mirrors hud_toggle on the opposite (top-left) corner -- always
        # reachable regardless of whether the CAM/GIMBAL HUD is open, same
        # as the toggle itself.
        self.settings_button = QPushButton("SETTINGS", self)
        self.settings_button.setObjectName("hudToggle")
        self.settings_button.adjustSize()
        self.settings_button.raise_()

        self.cancel_track_button.clicked.connect(self.cancel_track_requested.emit)

        self._frame_size: tuple[int, int] | None = None
        self._bbox: tuple[int, int, int, int] | None = None
        self._target_state = "idle"
        self._last_pixmap: QPixmap | None = None

        self._press_pos: QPoint | None = None
        self._recording = False
        self._link_connected = False
        self._preview_streaming = False

    def set_controls_scale(self, scale: float) -> None:
        """Enlarge the PTT/zoom/joystick thumb controls (display.controls_scale)
        for glove operation, then re-place them for the new sizes."""
        self.joystick.set_size_scale(scale)
        self.zoom_rocker.set_size_scale(scale)
        self.exposure_rocker.set_size_scale(scale)
        self.ptt_button.set_size_scale(scale)
        self._reposition_layout()

    def set_hud_widget(self, widget: QWidget) -> None:
        """Host the CAMERA/GIMBAL readout panels as a floating overlay on
        top of the live feed instead of a solid row that steals height from
        it -- the video is the centerpiece; this is just glass on top of
        it. Anchored to the top so it never fights the PTT/zoom/joystick
        thumb controls pinned to the bottom corners/center."""
        widget.setParent(self)
        self.hud_widget = widget
        widget.raise_()
        self.hud_toggle.raise_()
        self.settings_button.raise_()
        self._reposition_layout()

    def _on_hud_toggled(self, checked: bool) -> None:
        self._hud_visible = checked
        self.hud_toggle.setText("HIDE CAM / GIMBAL" if checked else "CAM / GIMBAL")
        self.hud_toggle.adjustSize()
        self._reposition_layout()

    def set_ride_locked(self, locked: bool) -> None:
        """Above the configured speed threshold, gray out SETTINGS and the
        CAM/GIMBAL HUD (ISO/shutter/lens taps, multi-item combos) so a
        gloved thumb isn't tempted into fiddly exposure tweaks at speed.
        PTT, the joystick and zoom stay fully live -- those are exactly
        the controls this rig exists to let the rider use while riding."""
        if locked == self._ride_locked:
            return
        self._ride_locked = locked
        self.settings_button.setEnabled(not locked)
        self.hud_toggle.setEnabled(not locked)
        # The exposure rocker IS an ISO tap -- exactly what ride-lock guards
        # against -- so it grays out with the rest. Zoom stays live.
        self.exposure_rocker.set_active(not locked)
        if locked and self.hud_toggle.isChecked():
            self.hud_toggle.setChecked(False)  # also collapses the HUD
        if self.hud_widget is not None:
            self.hud_widget.setEnabled(not locked)
        self.ride_lock_label.setVisible(locked)
        if locked:
            self.ride_lock_label.raise_()
        self._reposition_layout()

    def set_tracking_active(self, active: bool) -> None:
        """Show/hide the CANCEL TRACKING pill -- there was previously no
        direct way to drop a tapped target short of the big RESET button
        (which also re-homes the gimbal) or cycling modes away and back."""
        self.cancel_track_button.setVisible(active)
        if active:
            self.cancel_track_button.raise_()
        self._reposition_layout()

    def set_recording(self, recording: bool) -> None:
        # Red frame border while actively recording -- the standard
        # camera-UI "this is live" cue, on top of the REC chip/blink.
        if recording == self._recording:
            return
        self._recording = recording
        if recording:
            self.setStyleSheet("QFrame#previewFrame { background: black; border: 3px solid #ff4d4d; border-radius: 0px; }")
        else:
            self.setStyleSheet("")

    def set_talkback_active(self, active: bool) -> None:
        self.talkback_label.setVisible(active)
        if active:
            self.talkback_label.raise_()
        # Talkback visibility feeds into _hud_top_margin() too (see
        # set_switcher_state) -- a plain _position_overlays() would leave
        # the HUD's vertical offset stale.
        self._reposition_layout()

    def set_link_state(self, connected: bool, preview_streaming: bool) -> None:
        self._link_connected = connected
        self._preview_streaming = preview_streaming
        ctrl = "CTRL UP" if connected else "CTRL DOWN"
        preview = "PREVIEW ON" if preview_streaming else "PREVIEW OFF"
        self.link_status_label.setText(f"{ctrl} · {preview}")
        if connected:
            border = "rgba(34, 197, 94, 155)" if preview_streaming else "rgba(255, 255, 255, 95)"
            color = "#dfffee" if preview_streaming else "#dce4f0"
            background = "rgba(6, 18, 12, 190)" if preview_streaming else "rgba(10, 12, 17, 185)"
        else:
            border = "rgba(255, 77, 77, 165)"
            color = "#ffe5e5"
            background = "rgba(70, 18, 24, 205)"
        self.link_status_label.setStyleSheet(
            f"background-color: {background}; color: {color}; "
            f"border: 1px solid {border}; border-radius: 16px; "
            "padding: 8px 16px; font-size: 13px; font-weight: 850;"
        )
        self.link_status_label.adjustSize()
        self.link_status_label.show()
        self.link_status_label.raise_()
        self._reposition_layout()

    def set_switcher_state(self, state: str | None) -> None:
        normalized = (state or "").lower()
        if normalized == "live":
            self.switcher_label.setText("LIVE")
            self.switcher_label.setStyleSheet(
                "background-color: rgba(185, 28, 28, 235); color: white; "
                "border: 1px solid rgba(255, 255, 255, 165); border-radius: 8px; "
                "padding: 11px 24px; font-size: 25px; font-weight: 1000;"
            )
            self.switcher_label.show()
        elif normalized == "preview":
            self.switcher_label.setText("PREVIEW")
            self.switcher_label.setStyleSheet(
                "background-color: rgba(34, 197, 94, 222); color: #051108; "
                "border: 1px solid rgba(255, 255, 255, 160); border-radius: 8px; "
                "padding: 11px 24px; font-size: 25px; font-weight: 1000;"
            )
            self.switcher_label.show()
        else:
            self.switcher_label.hide()
        # Switcher visibility/size feeds into _hud_top_margin() -- needs a
        # full re-layout, not just repositioning the overlay chips, or the
        # HUD's vertical offset (set the last time the layout ran) goes
        # stale and can overlap the now-visible/now-taller chip.
        self._reposition_layout()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._press_pos = event.position().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press_pos is not None and self._frame_size is not None:
            pos = self._map_to_frame(self._press_pos)
            if pos is not None:
                self.tapped.emit(*pos)
        self._press_pos = None

    def set_bbox(self, bbox: tuple[int, int, int, int] | None, state: str) -> None:
        self._bbox = bbox
        self._target_state = state

    def update_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self._frame_size = (w, h)

        image = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888)
        # Keep the clean, unannotated full-res pixmap: the tracking rectangle
        # is drawn onto the (much smaller) scaled copy in _render_scaled, not
        # onto a full-res copy of every frame. That drops a ~2 MP QPixmap
        # copy + allocation per tracked frame off the UI thread.
        self._last_pixmap = QPixmap.fromImage(image)
        self._render_scaled()

    def _render_scaled(self) -> None:
        """Scale the last frame to the label and draw the tracking box on the
        scaled pixmap (coords mapped from frame space). Called per frame and
        on resize."""
        if self._last_pixmap is None:
            return
        scaled = self._last_pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._bbox is not None and self._frame_size is not None:
            fw, fh = self._frame_size
            if fw > 0 and fh > 0:
                sx = scaled.width() / fw
                sy = scaled.height() / fh
                color_map = {
                    "locked": QColor("#3ddc84"),
                    "weak": QColor("#ffb020"),
                    "manual_required": QColor("#ff4d4d"),
                }
                painter = QPainter(scaled)
                painter.setPen(QPen(color_map.get(self._target_state, QColor("#808892")), 3))
                bx, by, bw, bh = self._bbox
                painter.drawRect(int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy))
                painter.end()
        self._image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._render_scaled()
        self._reposition_layout()

    def _reposition_layout(self) -> None:
        self.hud_toggle.move(
            self.width() - self.hud_toggle.width() - HUD_SIDE_MARGIN,
            PTT_MARGIN,
        )
        self.settings_button.move(HUD_SIDE_MARGIN, PTT_MARGIN)
        # switcher/talkback depend only on settings_button/hud_toggle
        # (already placed above), and the HUD's own top offset depends on
        # switcher/talkback in turn -- so these must be positioned first.
        self._position_overlays()
        hud_top = self._hud_top_margin()

        # The HUD (CAMERA/GIMBAL readouts) floats over the top of the frame
        # and can eat a lot of a short window's height -- size/clamp the
        # bottom thumb controls against the room actually left *below* it,
        # not the raw frame height, or they end up stacked underneath it
        # (or worse, pushed below the visible widget entirely).
        hud_height = self._hud_height(hud_top)
        hud_bottom = hud_top + hud_height if hud_height > 0 else 0
        control_zone_height = max(0, self.height() - hud_bottom)

        compact_controls = control_zone_height < 330 or self.width() < 760
        self.joystick.set_compact(compact_controls)
        self.zoom_rocker.set_compact(compact_controls)
        self.exposure_rocker.set_compact(compact_controls)
        self.ptt_button.set_compact(compact_controls)

        min_control_y = hud_bottom + CONTROL_HUD_GAP if self.hud_widget else 0

        self.joystick.move(
            self.width() - self.joystick.width() - JOYSTICK_MARGIN,
            max(min_control_y, self.height() - self.joystick.height() - JOYSTICK_MARGIN),
        )
        self.ptt_button.move(
            PTT_MARGIN,
            max(min_control_y, self.height() - self.ptt_button.height() - PTT_MARGIN),
        )
        zoom_x = self.joystick.x() + (self.joystick.width() - self.zoom_rocker.width()) // 2
        zoom_above_y = self.joystick.y() - self.zoom_rocker.height() - ZOOM_JOYSTICK_GAP
        if zoom_above_y >= min_control_y:
            zoom_y = zoom_above_y
        else:
            zoom_x = max(PTT_MARGIN, self.joystick.x() - self.zoom_rocker.width() - ZOOM_JOYSTICK_GAP)
            zoom_y = max(min_control_y, self.joystick.y())
        self.zoom_rocker.move(zoom_x, zoom_y)

        # Exposure rocker mirrors the zoom rocker on the PTT (left) side:
        # left-anchored over the PTT button, and pinned to the SAME height as
        # the zoom rocker so the two rockers read as a matched pair (the PTT
        # button is shorter than the joystick, so keying off it alone would
        # sit the exposure rocker too low).
        exp = self.exposure_rocker
        # Center over the PTT, but the rocker is wider than the PTT button so
        # a true center would run off the left edge -- clamp the left margin.
        exp_x = max(PTT_MARGIN, self.ptt_button.x() + (self.ptt_button.width() - exp.width()) // 2)
        exp.move(exp_x, zoom_y)
        self._position_hud(hud_top, hud_height)

    def _hud_top_margin(self) -> int:
        """Lowest point among the top-corner chrome (SETTINGS/CAM-GIMBAL
        pills, the LIVE/PREVIEW switcher chip, the talkback banner) --
        the HUD starts below all of it so it never overlaps them."""
        bottom = max(
            self.settings_button.y() + self.settings_button.height(),
            self.hud_toggle.y() + self.hud_toggle.height(),
        )
        if not self.switcher_label.isHidden():
            bottom = max(bottom, self.switcher_label.y() + self.switcher_label.height())
        if not self.talkback_label.isHidden():
            bottom = max(bottom, self.talkback_label.y() + self.talkback_label.height())
        if not self.ride_lock_label.isHidden():
            bottom = max(bottom, self.ride_lock_label.y() + self.ride_lock_label.height())
        if not self.cancel_track_button.isHidden():
            bottom = max(bottom, self.cancel_track_button.y() + self.cancel_track_button.height())
        return bottom + HUD_TOP_GAP

    def _hud_height(self, hud_top: int) -> int:
        """Natural HUD height, capped so it can never crowd out the room
        the bottom thumb controls need (MIN_CONTROL_ZONE) -- on a very
        short window the HUD shrinks/hides before the controls do, since
        those are the safety-critical ones."""
        if self.hud_widget is None or not self._hud_visible:
            return 0
        natural = self.hud_widget.sizeHint().height()
        max_allowed = max(0, self.height() - hud_top - MIN_CONTROL_ZONE)
        return min(natural, max_allowed)

    def _position_hud(self, hud_top: int, height: int) -> None:
        if self.hud_widget is None:
            return
        if height <= 0:
            self.hud_widget.hide()
            return
        self.hud_widget.show()
        width = max(0, self.width() - 2 * HUD_SIDE_MARGIN)
        self.hud_widget.setGeometry(HUD_SIDE_MARGIN, hud_top, width, height)

    def _position_overlays(self) -> None:
        self.talkback_label.adjustSize()
        self.switcher_label.adjustSize()
        self.link_status_label.adjustSize()
        self.ride_lock_label.adjustSize()
        self.cancel_track_button.adjustSize()
        # Below the SETTINGS pill, not on top of it -- both live in the
        # top-left corner now that SETTINGS mirrors the CAM/GIMBAL toggle.
        self.switcher_label.move(
            PTT_MARGIN,
            self.settings_button.y() + self.settings_button.height() + 8,
        )
        self.link_status_label.move(
            max(PTT_MARGIN, (self.width() - self.link_status_label.width()) // 2),
            max(PTT_MARGIN, self.height() - self.link_status_label.height() - PTT_MARGIN),
        )
        # Top-center stack: whichever of these are visible stack top to
        # bottom, centered, in priority order -- most safety-critical
        # (ride lock) first, then the tracking-cancel pill, then the
        # talkback banner. All three are rare/transient states that could
        # in principle coincide, so this can't just pick one fixed slot.
        y = PTT_MARGIN
        for widget in (self.ride_lock_label, self.cancel_track_button, self.talkback_label):
            if widget.isHidden():
                continue
            x = max(PTT_MARGIN, (self.width() - widget.width()) // 2)
            widget.move(x, y)
            y += widget.height() + 8

    def _map_to_frame(self, widget_pos: QPoint) -> tuple[int, int] | None:
        if self._frame_size is None or self._image_label.pixmap() is None:
            return None
        fw, fh = self._frame_size
        label_size = self._image_label.size()
        pixmap_size = self._image_label.pixmap().size()

        # scaled pixmap is centered within the label
        offset_x = (label_size.width() - pixmap_size.width()) / 2
        offset_y = (label_size.height() - pixmap_size.height()) / 2
        px = widget_pos.x() - offset_x
        py = widget_pos.y() - offset_y
        if px < 0 or py < 0 or px > pixmap_size.width() or py > pixmap_size.height():
            return None

        scale_x = fw / pixmap_size.width()
        scale_y = fh / pixmap_size.height()
        return int(px * scale_x), int(py * scale_y)
