"""Main application window (design doc 11)."""
from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from motocam.ai.ai_engine import AiEngine
from motocam.ai.hailo_detector import build_detector
from motocam.audio.ptt_engine import SAMPLE_RATE, PttEngine
from motocam.audio.talkback_player import TalkbackPlayer
from motocam.camera.base import CameraController
from motocam.core.protocol import (
    AiTelemetry,
    CameraTelemetry,
    GimbalTelemetry,
    MessageType,
    NetworkTelemetry,
    OperatingMode,
    Telemetry,
    TargetState,
)
from motocam.core.config import save_config
from motocam.gimbal.base import GimbalController
from motocam.gimbal.factory import build_gimbal_backend
from motocam.gps.gps_manager import GpsManager
from motocam.network.link_client import LinkClient
from motocam.tracking.pid import GimbalPid
from motocam.tracking.tracker import TrackingEngine
from motocam.ui.effects import apply_glass_shadow
from motocam.ui.widgets.camera_panel import CameraPanel
from motocam.ui.widgets.gimbal_panel import GimbalPanel
from motocam.ui.widgets.mode_bar import ModeBar
from motocam.ui.widgets.preview_view import PreviewView
from motocam.ui.widgets.settings_dialog import ISO_VALUES, SettingsDialog
from motocam.ui.widgets.stage_banner import StageBanner
from motocam.ui.widgets.top_bar import TopBar
from motocam.video.video_engine import VideoEngine
from motocam.watchdog.health import HealthMonitor

logger = logging.getLogger("motocam.ui")

STATE_TO_CHIP = {
    "locked": "ok",
    "weak": "warn",
    "manual_required": "bad",
    "lost": "bad",
    "idle": "idle",
}


class MainWindow(QMainWindow):
    def __init__(
        self,
        video_engine: VideoEngine,
        gimbal: GimbalController,
        camera: CameraController,
        gps: GpsManager,
        link: LinkClient,
        health: HealthMonitor,
        unit_id: str = "moto-1",
        camera_ip: str = "192.168.9.20",
        camera_port: int = 9993,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ):
        super().__init__()
        self.setWindowTitle(f"MotoCam - {unit_id}")
        self.resize(1280, 800)

        self.video_engine = video_engine
        self.gimbal = gimbal
        self.camera = camera
        self.gps = gps
        self.link = link
        self.health = health
        self.unit_id = unit_id
        self.camera_ip = camera_ip
        self.camera_port = camera_port
        self._config = config if config is not None else {}
        self._config_path = Path(config_path) if config_path is not None else None
        audio_cfg = self._config.get("audio") or {}
        safety_cfg = self._config.get("safety") or {}
        self._ride_lock_enabled = bool(safety_cfg.get("ride_lock_enabled", False))
        self._ride_lock_speed_kmh = float(safety_cfg.get("ride_lock_speed_kmh", 15.0))

        self.tracker = TrackingEngine()
        ai_cfg = self._config.get("ai", {})
        self.ai_engine = AiEngine(
            detector=build_detector(self._config),
            target_class=ai_cfg.get("target_class", "cyclist"),
            confidence=float(ai_cfg.get("confidence", 0.35)),
        )
        self.pid = GimbalPid(
            dead_zone_x=30, dead_zone_y=25,
            max_pan_speed=gimbal.max_pan_speed, max_tilt_speed=gimbal.max_tilt_speed,
        )
        self.ptt_engine = PttEngine(input_device=audio_cfg.get("input_device"))
        self.talkback_player = TalkbackPlayer(output_device=audio_cfg.get("output_device"))
        self._last_frame: np.ndarray | None = None
        self._preview_streaming = False
        self._preview_bytes_accum = 0
        self._preview_bitrate_kbps: float | None = None
        self._preview_resolution: str | None = None

        self._build_ui()
        self._wire_signals()
        self.top_bar.set_unit_id(unit_id)
        self.preview.ptt_button.set_available(self.ptt_engine.available)
        self.preview.set_link_state(self.link.connected, self._preview_streaming)
        self._sync_joystick_active()
        self._start_timers()

    # -- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.top_bar = TopBar()
        root.addWidget(self.top_bar)

        self.stage_banner = StageBanner()
        root.addWidget(self.stage_banner)

        self.preview = PreviewView()
        root.addWidget(self.preview, stretch=1)

        # CAMERA/GIMBAL readouts float on top of the live feed (see
        # PreviewView.set_hud_widget) instead of a solid row squeezing the
        # video down to a sliver below it -- the feed is the centerpiece
        # now, this is a semi-transparent glass HUD over it (see theme.py
        # #cameraPanel/#gimbalPanel).
        hud = QWidget()
        hud.setObjectName("previewHud")
        panels_row = QHBoxLayout(hud)
        panels_row.setContentsMargins(0, 0, 0, 0)
        panels_row.setSpacing(12)
        self.camera_panel = CameraPanel()
        self.gimbal_panel = GimbalPanel()
        self.camera_panel.setMaximumHeight(270)
        self.gimbal_panel.setMaximumHeight(270)
        panels_row.addWidget(self.camera_panel, stretch=1)
        panels_row.addWidget(self.gimbal_panel, stretch=1)
        self.preview.set_hud_widget(hud)
        # Glove-friendly thumb controls (design doc 11.1/12.3): the
        # PTT/zoom/joystick default sizes felt small on the real Pi
        # touchscreen -- display.controls_scale enlarges them on top of
        # the global UI scale, tunable without touching code.
        controls_scale = float(self._config.get("display", {}).get("controls_scale", 1.0))
        if controls_scale != 1.0:
            self.preview.set_controls_scale(controls_scale)
        # SETTINGS mirrors the CAM/GIMBAL toggle on the opposite corner --
        # always reachable, not tucked inside the collapsible HUD.
        self.preview.settings_button.clicked.connect(self._open_settings)

        self.mode_bar = ModeBar()
        root.addWidget(self.mode_bar)

        self.setCentralWidget(central)
        self._settings = SettingsDialog(self)

        # Glass-card depth (theme.py) -- QSS can't do box-shadow, so this is
        # applied in code. Kept off the mode bar / status chips on purpose:
        # those need to read as flat, high-contrast, unambiguous controls
        # while riding, not floating glass. Also deliberately NOT applied to
        # the joystick/PTT button: QGraphicsDropShadowEffect on a
        # WA_TranslucentBackground widget with a custom paintEvent that
        # repaints on every mouse-move (the joystick, while dragging) is a
        # known-fragile combination on Qt's macOS/Cocoa backend -- it was
        # the likely cause of a native SIGSEGV inside
        # QPaintDevice::devicePixelRatio() during window flush on this
        # machine's macOS beta. A raw paintEvent already gives the joystick
        # its glass look without needing the effect on top.
        for panel in (self.camera_panel, self.gimbal_panel):
            apply_glass_shadow(panel, blur_radius=28, y_offset=8, alpha=130)

    def _open_settings(self) -> None:
        self._settings.set_connection_values(self.link.unit_id, self.link.url)
        self._settings.set_video_values(self.video_engine.device)
        self._settings.set_audio_values(self.ptt_engine.input_device, self.talkback_player.output_device)
        self._settings.set_gimbal_values(self._config.get("gimbal", {}))
        self._settings.set_pyxis_values(self.camera_ip, self.camera_port)
        self._settings.set_camera_values(
            self.camera.state.iso, self.camera.state.white_balance,
            self.camera.state.shutter, self.camera.state.iris,
        )
        self._settings.set_tracking_values(
            self.ai_engine.target_class, self.ai_engine.confidence,
            int(self.pid.pan.dead_zone), int(self.pid.tilt.dead_zone),
            self.pid.pan.max_speed, self.pid.tilt.max_speed,
        )
        self._settings.set_safety_values(self._ride_lock_enabled, self._ride_lock_speed_kmh)
        self._settings.exec()

    # -- Signal wiring -----------------------------------------------------
    def _wire_signals(self) -> None:
        self.video_engine.frame_ready.connect(self._on_frame)
        self.video_engine.fps_updated.connect(self._on_fps)
        self.video_engine.status_changed.connect(self._on_video_status)

        self.preview.tapped.connect(self._on_tap)
        self.preview.cancel_track_requested.connect(self._on_cancel_track)
        self.preview.joystick.moved.connect(self._on_manual_drag)
        self.preview.joystick.released.connect(self._on_manual_drag_end)
        self.preview.zoom_rocker.moved.connect(self._on_zoom_drag)
        self.preview.zoom_rocker.released.connect(self._on_zoom_drag_end)
        self.preview.exposure_rocker.stepped.connect(self._on_exposure_step)
        self.preview.ptt_button.pressed.connect(self._on_ptt_pressed)
        self.preview.ptt_button.released.connect(self._on_ptt_released)
        self.ptt_engine.audio_chunk.connect(self._on_ptt_audio_chunk)

        self.mode_bar.mode_selected.connect(self._on_mode_selected)
        self.mode_bar.reset_requested.connect(self._on_reset)
        self.mode_bar.manual_override.connect(self._on_manual_override)

        self.camera_panel.record_toggled.connect(self._on_record_toggled)
        self.camera_panel.autofocus_requested.connect(self._on_autofocus_requested)
        self.gimbal_panel.home_requested.connect(self._on_home)
        self.gimbal_panel.lock_toggled.connect(self._on_lock_toggled)

        self.link.connected_changed.connect(self._on_link_connected)
        self.link.command_received.connect(self._on_command_received)
        self.link.latency_updated.connect(self._on_latency)

        self._settings.iso_changed.connect(lambda v: asyncio.ensure_future(self.camera.set_iso(v)))
        self._settings.white_balance_changed.connect(lambda v: asyncio.ensure_future(self.camera.set_white_balance(v)))
        self._settings.shutter_changed.connect(lambda v: asyncio.ensure_future(self.camera.set_shutter(v)))
        self._settings.iris_changed.connect(lambda v: asyncio.ensure_future(self.camera.set_iris(v)))

        self._settings.target_class_changed.connect(self._on_target_class_changed)
        self._settings.confidence_changed.connect(self._on_confidence_changed)
        self._settings.dead_zone_changed.connect(self._on_dead_zone_changed)
        self._settings.max_speed_changed.connect(self._on_max_speed_changed)

        self._settings.connection_apply_requested.connect(self._on_connection_apply)
        self._settings.pyxis_apply_requested.connect(self._on_pyxis_apply)
        self._settings.audio_apply_requested.connect(self._on_audio_apply)
        self._settings.video_device_apply_requested.connect(self._on_video_device_apply)
        self._settings.gimbal_apply_requested.connect(self._on_gimbal_apply)
        self._settings.ble_scan_requested.connect(self._on_ble_scan_requested)
        self._settings.safety_apply_requested.connect(self._on_safety_apply)
        self._settings.exit_requested.connect(self._on_exit_requested)

    def _start_timers(self) -> None:
        self._control_timer = QTimer(self)
        self._control_timer.timeout.connect(self._control_tick)
        self._control_timer.start(50)  # 20 Hz

        self._gps_timer = QTimer(self)
        self._gps_timer.timeout.connect(self._gps_tick)
        self._gps_timer.start(500)

        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.timeout.connect(self._telemetry_tick)
        self._telemetry_timer.start(500)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._async_refresh_tick)
        self._refresh_timer.start(500)

        self._ping_timer = QTimer(self)
        self._ping_timer.timeout.connect(self.link.send_ping)
        self._ping_timer.start(5000)

    # -- Video / tracking --------------------------------------------------
    def _on_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame
        self.tracker.update(frame)
        detections = self.ai_engine.process(frame)
        # FULL AI auto-acquire (design doc 8.2/8.3): with no active target,
        # lock onto the highest-confidence detection of the configured
        # class straight from the detector box. AI ASSIST keeps the rider
        # in charge (tap-to-select) and only uses detections to help; MANUAL
        # never auto-acquires.
        if (
            self.gimbal.mode == OperatingMode.FULL_AI
            and self.tracker.state in (TargetState.IDLE, TargetState.MANUAL_REQUIRED)
            and detections
        ):
            target = self._pick_target_detection(detections)
            if target is not None:
                self.tracker.select_box(frame, (target.x, target.y, target.w, target.h))
                self.pid.reset()
        self.preview.set_bbox(self.tracker.bbox, self.tracker.state.value)
        self.preview.update_frame(frame)

        chip_state = STATE_TO_CHIP.get(self.tracker.state.value, "idle")
        self.top_bar.ai_chip.set_state(chip_state, f"AI {self.tracker.state.value.upper()}")
        self.preview.set_tracking_active(self.tracker.state != TargetState.IDLE)

        h, w = frame.shape[:2]
        self._preview_resolution = f"{w}x{h}"

        if self._preview_streaming:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                self.link.send_preview_frame(buf.tobytes())
                # Real measured bandwidth of the low-fps preview relay to
                # the control room -- not the PYXIS's own recording bitrate,
                # see CameraTelemetry.preview_bitrate_kbps docstring.
                self._preview_bytes_accum += buf.nbytes

    def _pick_target_detection(self, detections):
        """Highest-confidence detection whose class matches the configured
        target (case-insensitive). Returns None if nothing matches."""
        target_class = (self.ai_engine.target_class or "").lower()
        matches = [d for d in detections if d.class_name.lower() == target_class]
        if not matches:
            return None
        return max(matches, key=lambda d: d.confidence)

    def _on_fps(self, fps: float) -> None:
        self.health.set_video_fps(fps)
        self.top_bar.fps_chip.set_state("ok" if fps > 20 else "warn", f"FPS {fps:.0f}")

    def _on_video_status(self, status: str) -> None:
        state = {"connected": "ok", "synthetic": "warn", "reconnecting": "warn", "lost": "bad"}.get(status, "idle")
        self.top_bar.cam_chip.set_state(state, f"CAM {status.upper()}")

    def _on_tap(self, x: int, y: int) -> None:
        if self._last_frame is not None:
            self.tracker.select_at(self._last_frame, x, y)
            self.pid.reset()

    def _on_cancel_track(self) -> None:
        # Drops the current target only -- unlike RESET, doesn't touch
        # gimbal mode/position, so it's safe to use in AI ASSIST/FULL AI
        # without also re-homing the camera.
        self.tracker.clear()
        self.pid.reset()

    # -- Manual pan/tilt drag (design doc 7.2 Manual, 12.1 joystick) ---------
    def _on_manual_drag(self, dx_norm: float, dy_norm: float) -> None:
        if self.gimbal.mode != OperatingMode.MANUAL:
            return
        pan_v = dx_norm * self.gimbal.max_pan_speed
        tilt_v = -dy_norm * self.gimbal.max_tilt_speed
        asyncio.ensure_future(self.gimbal.manual_move(pan_v, tilt_v))

    def _on_manual_drag_end(self) -> None:
        asyncio.ensure_future(self.gimbal.manual_move(0.0, 0.0))

    def _on_zoom_drag(self, zoom_speed: float) -> None:
        asyncio.ensure_future(self.camera.set_zoom_speed(zoom_speed))

    def _on_zoom_drag_end(self) -> None:
        asyncio.ensure_future(self.camera.set_zoom_speed(0.0))

    def _on_exposure_step(self, direction: int) -> None:
        """One flick of the exposure rocker = one ISO stop. Steps from the
        camera's current ISO through the standard stops and clamps at the
        ends; if the camera hasn't reported an ISO yet, start from a sane
        mid value so the first flick still does something."""
        current = self.camera.state.iso
        if current in ISO_VALUES:
            index = ISO_VALUES.index(current)
        elif current is not None:
            # snap an off-list value to the nearest standard stop
            index = min(range(len(ISO_VALUES)), key=lambda i: abs(ISO_VALUES[i] - current))
        else:
            index = ISO_VALUES.index(800) if 800 in ISO_VALUES else len(ISO_VALUES) // 2
        new_index = max(0, min(len(ISO_VALUES) - 1, index + direction))
        new_iso = ISO_VALUES[new_index]
        # reflect immediately so the rocker/HUD don't wait a refresh tick
        self.camera.state.iso = new_iso
        self.preview.exposure_rocker.set_value_text(str(new_iso))
        asyncio.ensure_future(self.camera.set_iso(new_iso))

    def _sync_joystick_active(self) -> None:
        self.preview.joystick.set_active(self.gimbal.mode == OperatingMode.MANUAL)

    # -- Push-to-talk (operator -> director, one-way) -------------------------
    def _on_ptt_pressed(self) -> None:
        if self.ptt_engine.start():
            self.link.send_ptt_start()

    def _on_ptt_released(self) -> None:
        self.ptt_engine.stop()
        self.link.send_ptt_stop()

    def _on_ptt_audio_chunk(self, pcm_bytes: bytes) -> None:
        self.link.send_ptt_audio(pcm_bytes, SAMPLE_RATE)

    # -- Mode / manual override --------------------------------------------
    def _on_mode_selected(self, mode: OperatingMode) -> None:
        self.gimbal.mode = mode
        self.ai_engine.enabled = mode in (OperatingMode.AI_ASSIST, OperatingMode.FULL_AI)
        asyncio.ensure_future(self.gimbal.set_mode(mode))
        if mode in (OperatingMode.LOCK, OperatingMode.HOME, OperatingMode.MANUAL):
            self.tracker.clear()
        self._sync_joystick_active()

    def _on_reset(self) -> None:
        asyncio.ensure_future(self._do_reset())

    async def _do_reset(self) -> None:
        await self.gimbal.set_mode(OperatingMode.RESET)
        self.ai_engine.enabled = False
        self.tracker.clear()
        self.mode_bar.set_mode(OperatingMode.MANUAL)
        self.gimbal.mode = OperatingMode.MANUAL
        self._sync_joystick_active()

    def _on_manual_override(self) -> None:
        logger.warning("MANUAL OVERRIDE pressed - AI disabled immediately")
        self.ai_engine.enabled = False
        self.gimbal.mode = OperatingMode.MANUAL
        asyncio.ensure_future(self.gimbal.stop())
        asyncio.ensure_future(self.gimbal.set_mode(OperatingMode.MANUAL))
        self._sync_joystick_active()

    def _on_home(self) -> None:
        asyncio.ensure_future(self.gimbal.set_mode(OperatingMode.HOME))

    def _on_lock_toggled(self, locked: bool) -> None:
        mode = OperatingMode.LOCK if locked else OperatingMode.MANUAL
        self.gimbal.mode = mode
        asyncio.ensure_future(self.gimbal.set_mode(mode))
        self._sync_joystick_active()

    def _on_record_toggled(self, recording: bool) -> None:
        coro = self.camera.start_record() if recording else self.camera.stop_record()
        asyncio.ensure_future(coro)

    def _on_autofocus_requested(self) -> None:
        asyncio.ensure_future(self.camera.trigger_autofocus())

    # -- PID / AI control loop ----------------------------------------------
    def _control_tick(self) -> None:
        if self.gimbal.mode in (OperatingMode.AI_ASSIST, OperatingMode.FULL_AI):
            if self._last_frame is None:
                return
            error = self.tracker.error_from_center(self._last_frame.shape)
            if error is None:
                return
            pan_v, tilt_v = self.pid.update(*error)
            asyncio.ensure_future(self.gimbal.ai_move(pan_v, tilt_v))
        elif self.gimbal.mode == OperatingMode.MANUAL and self.preview.joystick.is_dragging:
            # re-send every tick (not just on touch-move) so holding the
            # knob at an offset keeps panning/tilting instead of stopping
            # the instant the finger stops moving.
            dx_norm, dy_norm = self.preview.joystick.offset
            self._on_manual_drag(dx_norm, dy_norm)
        if self.preview.zoom_rocker.is_dragging:
            self._on_zoom_drag(self.preview.zoom_rocker.offset)

    # -- GPS / telemetry / health --------------------------------------------
    def _gps_tick(self) -> None:
        fix = self.gps.poll()
        self.top_bar.gps_chip.set_state("ok" if fix.fix else "warn", "FIX" if fix.fix else "NO FIX")
        locked = (
            self._ride_lock_enabled
            and fix.fix
            and fix.speed_kmh is not None
            and fix.speed_kmh >= self._ride_lock_speed_kmh
        )
        self.preview.set_ride_locked(locked)

    def _async_refresh_tick(self) -> None:
        asyncio.ensure_future(self.camera.refresh())
        if self.gimbal.connected:
            asyncio.ensure_future(self.gimbal.refresh_orientation())
        else:
            asyncio.ensure_future(self.gimbal.connect())
        self.gimbal_panel.update_orientation(self.gimbal.pan_deg, self.gimbal.tilt_deg, self.gimbal.roll_deg)
        self.gimbal_panel.set_connected(self.gimbal.connected)
        self.camera_panel.update_state(self.camera.state)
        self.preview.exposure_rocker.set_value_text(
            str(self.camera.state.iso) if self.camera.state.iso is not None else "--"
        )
        self.top_bar.gimbal_chip.set_state("ok" if self.gimbal.connected else "bad", "OK" if self.gimbal.connected else "DOWN")
        if self.camera.state.recording:
            self.top_bar.rec_chip.set_blinking(True, text="● REC")
        else:
            self.top_bar.rec_chip.set_blinking(False)
            self.top_bar.rec_chip.set_state("idle", "REC")
        self.preview.set_recording(self.camera.state.recording)

    def _telemetry_tick(self) -> None:
        sys_stats = self.health.sample()
        # 500ms window -> bytes * 8 bits/byte / 1000 (kb) / 0.5s
        if self._preview_streaming:
            self._preview_bitrate_kbps = self._preview_bytes_accum * 16 / 1000
        else:
            self._preview_bitrate_kbps = None
        self._preview_bytes_accum = 0
        telemetry = Telemetry(
            gps=self.gps.state,
            ai=AiTelemetry(
                enabled=self.ai_engine.enabled,
                state=self.tracker.state.value,
                inference_fps=self.ai_engine.detector.fps,
            ),
            gimbal=GimbalTelemetry(
                connected=self.gimbal.connected, mode=self.gimbal.mode.value,
                pan_deg=self.gimbal.pan_deg, tilt_deg=self.gimbal.tilt_deg, roll_deg=self.gimbal.roll_deg,
            ),
            camera=CameraTelemetry(
                connected=self.camera.connected, recording=self.camera.state.recording,
                iso=self.camera.state.iso, white_balance=self.camera.state.white_balance,
                shutter=self.camera.state.shutter, iris=self.camera.state.iris, fps=self.camera.state.fps,
                media_remaining_min=self.camera.state.media_remaining_min, battery_pct=self.camera.state.battery_pct,
                preview_resolution=self._preview_resolution, preview_bitrate_kbps=self._preview_bitrate_kbps,
            ),
            network=NetworkTelemetry(link_up=self.link.connected),
            system=sys_stats,
        )
        self.link.send_telemetry(telemetry)

    # -- Link (control room) -------------------------------------------------
    def _on_link_connected(self, connected: bool) -> None:
        self.top_bar.net_chip.set_state("ok" if connected else "bad", "UP" if connected else "DOWN")
        self.preview.set_link_state(connected, self._preview_streaming)

    def _on_latency(self, latency_ms: float) -> None:
        self.top_bar.latency_chip.set_state("ok" if latency_ms < 150 else "warn", f"LAT {latency_ms:.0f} ms")

    def _on_command_received(self, msg_type: str, payload: dict) -> None:
        logger.info("Command from control room: %s %s", msg_type, payload)
        if msg_type == MessageType.SET_MODE.value:
            try:
                mode = OperatingMode(payload.get("mode"))
            except ValueError:
                return
            self.mode_bar.set_mode(mode)
            self._on_mode_selected(mode)
        elif msg_type == MessageType.CAMERA_COMMAND.value:
            action = payload.get("action")
            if action == "record_start":
                asyncio.ensure_future(self.camera.start_record())
            elif action == "record_stop":
                asyncio.ensure_future(self.camera.stop_record())
        elif msg_type == MessageType.GIMBAL_COMMAND.value:
            if payload.get("action") == "home":
                self._on_home()
        elif msg_type == MessageType.REQUEST_PREVIEW.value:
            self._preview_streaming = bool(payload.get("enabled", False))
            self.preview.set_link_state(self.link.connected, self._preview_streaming)
        elif msg_type == MessageType.STAGE_INFO.value:
            self.stage_banner.set_stage(
                payload.get("stage_number", 0), payload.get("stage_name", ""), payload.get("laps", 0)
            )
        elif msg_type == MessageType.SWITCHER_STATE.value:
            self._on_switcher_state(payload)
        elif msg_type == MessageType.PTT_START.value:
            self.preview.set_talkback_active(True)
            if not self.talkback_player.start():
                logger.warning("Control-room talkback unavailable: no output device")
        elif msg_type == MessageType.PTT_AUDIO.value:
            pcm_bytes = base64.b64decode(payload.get("audio_b64", ""))
            self.talkback_player.feed(pcm_bytes, payload.get("sample_rate", SAMPLE_RATE))
        elif msg_type == MessageType.PTT_STOP.value:
            self.talkback_player.stop()
            self.preview.set_talkback_active(False)

    # -- Settings (design doc 6.3 camera params, 9.4/8.5 tracking params) ----
    def _on_target_class_changed(self, target_class: str) -> None:
        self.ai_engine.target_class = target_class

    def _on_confidence_changed(self, confidence: float) -> None:
        self.ai_engine.confidence = confidence

    def _on_dead_zone_changed(self, dead_zone_x: int, dead_zone_y: int) -> None:
        self.pid.pan.dead_zone = dead_zone_x
        self.pid.tilt.dead_zone = dead_zone_y

    def _on_max_speed_changed(self, max_pan_speed: float, max_tilt_speed: float) -> None:
        self.pid.pan.max_speed = max_pan_speed
        self.pid.tilt.max_speed = max_tilt_speed
        self.gimbal.max_pan_speed = max_pan_speed
        self.gimbal.max_tilt_speed = max_tilt_speed

    def _on_connection_apply(self, unit_id: str, control_room_url: str) -> None:
        self.unit_id = unit_id
        self.setWindowTitle(f"MotoCam - {unit_id}")
        self.top_bar.set_unit_id(unit_id)
        logger.info("Reconfiguring link: unit_id=%s url=%s", unit_id, control_room_url)
        self.link.reconfigure(control_room_url, unit_id)
        self._config["unit_id"] = unit_id
        self._config.setdefault("telemetry", {})["control_room_url"] = control_room_url
        self._save_config()

    def _on_pyxis_apply(self, ip: str, port: int) -> None:
        self.camera_ip = ip
        self.camera_port = port
        backend = self.camera.backend
        if hasattr(backend, "ip"):
            # Live retarget of the real REST backend -- next refresh tick
            # probes the new address (its reconnect throttle is reset by
            # dropping the connected flag), no app restart needed.
            backend.ip = ip
            backend._connected = False
            backend._last_connect_attempt = 0.0
            logger.info("PYXIS address changed to %s -- reconnecting on next refresh", ip)
        else:
            logger.info("Saved PYXIS address %s:%d (mock camera backend active)", ip, port)
        camera_cfg = self._config.setdefault("camera", {})
        camera_cfg["ip"] = ip
        camera_cfg["port"] = port
        self._save_config()

    def _on_audio_apply(self, input_device, output_device) -> None:
        self.ptt_engine.set_input_device(input_device)
        self.talkback_player.set_output_device(output_device)
        self.preview.ptt_button.set_available(self.ptt_engine.available)
        audio_cfg = self._config.setdefault("audio", {})
        audio_cfg["input_device"] = input_device
        audio_cfg["output_device"] = output_device
        self._save_config()
        logger.info("Audio devices updated")

    def _on_video_device_apply(self, device) -> None:
        self.video_engine.set_device(device)
        video_cfg = self._config.setdefault("video", {})
        video_cfg["device"] = device
        self._save_config()
        logger.info("Video capture device changed to %s", device)

    def _on_gimbal_apply(self, gimbal_cfg: dict) -> None:
        current_cfg = self._config.setdefault("gimbal", {})
        max_pan_speed = current_cfg.get("max_pan_speed", self.gimbal.max_pan_speed)
        max_tilt_speed = current_cfg.get("max_tilt_speed", self.gimbal.max_tilt_speed)
        current_cfg.update(gimbal_cfg)
        current_cfg["max_pan_speed"] = max_pan_speed
        current_cfg["max_tilt_speed"] = max_tilt_speed
        self._save_config()
        asyncio.ensure_future(self._rebuild_gimbal_backend())

    def _on_ble_scan_requested(self, name_filter: str) -> None:
        asyncio.ensure_future(self._scan_ble_devices(name_filter))

    async def _scan_ble_devices(self, name_filter: str) -> None:
        try:
            from motocam.gimbal.dji_rs4pro import scan_ble_devices

            devices = await scan_ble_devices(name_filter=name_filter, timeout_s=5.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BLE scan failed: %s", exc)
            self._settings.set_ble_scan_results([], error=f"BLE scan failed: {exc}")
            return
        self._settings.set_ble_scan_results(devices)

    async def _rebuild_gimbal_backend(self) -> None:
        try:
            await self.gimbal.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Old gimbal backend did not disconnect cleanly: %s", exc)
        self.gimbal.backend = build_gimbal_backend(self._config.get("gimbal", {}))
        await self.gimbal.connect()
        logger.info("Gimbal backend reconfigured to %s", self._config.get("gimbal", {}).get("connection", "mock"))

    def _on_safety_apply(self, enabled: bool, speed_kmh: float) -> None:
        self._ride_lock_enabled = enabled
        self._ride_lock_speed_kmh = speed_kmh
        if not enabled:
            self.preview.set_ride_locked(False)
        safety_cfg = self._config.setdefault("safety", {})
        safety_cfg["ride_lock_enabled"] = enabled
        safety_cfg["ride_lock_speed_kmh"] = speed_kmh
        self._save_config()
        logger.info("Ride-safe lockout %s (threshold %.0f km/h)", "enabled" if enabled else "disabled", speed_kmh)

    def _on_switcher_state(self, payload: dict) -> None:
        live_unit = payload.get("live_unit")
        preview_unit = payload.get("preview_unit")
        if live_unit == self.unit_id:
            self.preview.set_switcher_state("live")
            self.top_bar.switcher_chip.set_state("bad", "LIVE")
        elif preview_unit == self.unit_id:
            self.preview.set_switcher_state("preview")
            self.top_bar.switcher_chip.set_state("ok", "PREVIEW")
        else:
            self.preview.set_switcher_state(None)
            self.top_bar.switcher_chip.set_state("idle", "STANDBY")

    def _save_config(self) -> None:
        if self._config_path is None:
            return
        try:
            save_config(self._config, self._config_path)
        except OSError as exc:
            logger.warning("Failed to save config %s: %s", self._config_path, exc)

    def _on_exit_requested(self) -> None:
        logger.info("Exit requested from Settings -- shutting down")
        self._settings.accept()  # close the dialog first
        self.close()             # triggers closeEvent cleanup, then quit

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.ptt_engine.stop()
        self.talkback_player.stop()
        # Stop hardware/timers so nothing keeps the event loop alive after
        # the window is gone (fullscreen kiosk: closing the window IS quit).
        # link.stop() is a coroutine -- schedule it on the running loop
        # best-effort rather than calling it synchronously (which would
        # never actually run and warns).
        try:
            self.video_engine.stop()
        except Exception as exc:  # noqa: BLE001
            logger.debug("video stop failed: %s", exc)
        try:
            self.gps.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("gps close failed: %s", exc)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.link.stop())
        except Exception as exc:  # noqa: BLE001
            logger.debug("link stop scheduling failed: %s", exc)
        super().closeEvent(event)
        QApplication.instance().quit()
