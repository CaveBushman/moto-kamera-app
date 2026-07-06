"""Entry point. Wires config -> hardware backends -> UI and runs the
Qt event loop with qasync so async gimbal/camera/network code shares
it with the GUI thread (no extra worker threads needed)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import subprocess
import time
from pathlib import Path

# Allow running this file directly (`python3 main.py` from inside motocam/)
# in addition to `python3 -m motocam.main` from the project root -- both are
# common habits, so make both work by putting the project root on sys.path
# before the package-absolute imports below run.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qasync
from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from motocam.camera.base import CameraController
from motocam.camera.mock_camera import MockCameraBackend
from motocam.camera.bmd_rest_camera import BlackmagicRestCameraBackend
from motocam.core.config import load_config, resolve_config_path
from motocam.core.logging_setup import (
    StartupLogBuffer,
    install_control_room_log_forwarder,
    install_crash_guard,
    setup_logging,
)
from motocam.gimbal.base import GimbalController
from motocam.gimbal.factory import build_gimbal_controller
from motocam.gps.gps_manager import GpsManager
from motocam.network.link_client import LinkClient
from motocam.ui.main_window import MainWindow
from motocam.ui.theme import DARK_STYLESHEET
from motocam.video.video_engine import VideoEngine
from motocam.watchdog.health import HealthMonitor
from motocam.watchdog.ui_watchdog import UiLatencyWatchdog


LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "logo_ccf" / "CSC Logo Horizontal EN RGB.png"
APP_ICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "icons" / "motocam.svg"
_GIT_REVISION_CACHE: str | None = None


def _git_revision() -> str:
    global _GIT_REVISION_CACHE
    if _GIT_REVISION_CACHE is not None:
        return _GIT_REVISION_CACHE
    root = Path(__file__).resolve().parents[1]
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        ).returncode != 0
        _GIT_REVISION_CACHE = f"{rev}{'-dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001 -- deployed Pi builds may not include .git
        _GIT_REVISION_CACHE = "unknown"
    return _GIT_REVISION_CACHE


def _write_startup_state(config_path: Path, phase: str, extra: dict | None = None) -> None:
    try:
        log_dir = Path("logs")
        if config_path:
            # config/logging.dir may be relative to the process cwd; keep
            # this independent and colocated with the active config for
            # boot triage even before normal logging is fully alive.
            log_dir = config_path.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "phase": phase,
            "revision": _git_revision(),
            "config": str(config_path),
        }
        if extra:
            payload.update(extra)
        (log_dir / "startup_state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _install_runtime_heartbeat(config_path: Path, window: MainWindow) -> QTimer:
    heartbeat = QTimer(window)
    heartbeat.setInterval(5000)

    def write_heartbeat() -> None:
        extra: dict = {}
        try:
            extra["link_connected"] = bool(window.link.connected)
            extra["runtime_started"] = bool(getattr(window, "_runtime_started", False))
        except Exception as exc:  # noqa: BLE001 -- heartbeat must never affect the UI
            extra["runtime_error"] = repr(exc)
        _write_startup_state(config_path, "runtime_heartbeat", extra)

    heartbeat.timeout.connect(write_heartbeat)
    heartbeat.start()
    write_heartbeat()
    return heartbeat


def _log_boot_config(logger: logging.Logger, cfg: dict, config_path: Path) -> None:
    ai_cfg = cfg.get("ai") or {}
    gimbal_cfg = cfg.get("gimbal") or {}
    video_cfg = cfg.get("video") or {}
    telemetry_cfg = cfg.get("telemetry") or {}
    startup_cfg = cfg.get("startup") or {}
    logger.info(
        "MotoCam build=%s config=%s unit=%s ai=%s ai_delay=%.1fs gimbal=%s/%s video=%s preview=%sfps "
        "control=%s hardware_delay=%.1fs",
        _git_revision(),
        config_path,
        cfg.get("unit_id", "moto-1"),
        ai_cfg.get("type", "disabled"),
        float(ai_cfg.get("startup_delay_s", 0.0) or 0.0),
        gimbal_cfg.get("type", "unknown"),
        gimbal_cfg.get("connection", "unknown"),
        video_cfg.get("device", "unknown"),
        (cfg.get("preview_relay") or {}).get("fps", "unknown"),
        telemetry_cfg.get("control_room_url", "unknown"),
        float(startup_cfg.get("hardware_connect_delay_s", 0.0) or 0.0),
    )


def build_gimbal(cfg: dict) -> GimbalController:
    return build_gimbal_controller(cfg)


def build_camera(cfg: dict) -> CameraController:
    camera_cfg = cfg.get("camera", {})
    camera_type = camera_cfg.get("type", "mock")
    if camera_type == "blackmagic_rest":
        # Real REST control (camera/bmd_rest_camera.py) -- generic across
        # PYXIS, Studio Cameras, and the Micro Studio Camera 4K G2 (the
        # current gimbal-mounted camera; PYXIS is too heavy for the
        # gimbal). The backend retries its /system probe on every refresh
        # tick (throttled), so a camera that's off at app start gets
        # picked up automatically -- no mock fallback that would fake
        # data while looking connected.
        backend = BlackmagicRestCameraBackend(
            ip=camera_cfg.get("ip", "192.168.9.20"),
            port=int(camera_cfg.get("rest_port", 80)),
            use_tls=bool(camera_cfg.get("use_tls", False)),
            verify_tls=bool(camera_cfg.get("verify_tls", False)),
            username=camera_cfg.get("username"),
            password=camera_cfg.get("password"),
            auth_token=camera_cfg.get("auth_token"),
        )
        return CameraController(backend)
    return CameraController(MockCameraBackend())


def build_gps(cfg: dict) -> GpsManager:
    gps_cfg = cfg.get("gps", {})
    return GpsManager(device=gps_cfg.get("device"), baudrate=gps_cfg.get("baudrate", 9600))


def build_video(cfg: dict) -> VideoEngine:
    video_cfg = cfg.get("video", {})
    device = video_cfg.get("device", 0)
    if isinstance(device, str) and device.startswith("/dev/video"):
        device = int(device.replace("/dev/video", "")) if device.replace("/dev/video", "").isdigit() else device
    allow_macos_capture = bool(video_cfg.get("allow_macos_capture", sys.platform != "darwin"))
    return VideoEngine(
        device=device,
        width=video_cfg.get("width", 1920),
        height=video_cfg.get("height", 1080),
        fps=video_cfg.get("fps", 30),
        allow_macos_capture=allow_macos_capture,
    )


def create_splash() -> QSplashScreen:
    pixmap = QPixmap(720, 360)
    pixmap.fill(QColor("#0b0f18"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#ffffff"))
    painter.drawRoundedRect(92, 54, 536, 100, 18, 18)

    logo = QPixmap(str(LOGO_PATH))
    if not logo.isNull():
        scaled_logo = logo.scaled(456, 66, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap(
            92 + (536 - scaled_logo.width()) // 2,
            54 + (100 - scaled_logo.height()) // 2,
            scaled_logo,
        )

    painter.setPen(QColor("#eef1f6"))
    title_font = QFont("DejaVu Sans", 26, QFont.Weight.Bold)
    painter.setFont(title_font)
    painter.drawText(QRect(0, 178, 720, 46), Qt.AlignmentFlag.AlignCenter, "MotoCam")

    painter.setPen(QColor("#7dd3fc"))
    subtitle_font = QFont("DejaVu Sans", 13, QFont.Weight.DemiBold)
    painter.setFont(subtitle_font)
    painter.drawText(QRect(0, 226, 720, 32), Qt.AlignmentFlag.AlignCenter, "Rider camera unit")

    painter.setPen(QColor("#8b93a1"))
    small_font = QFont("DejaVu Sans", 10)
    painter.setFont(small_font)
    painter.drawText(QRect(0, 292, 720, 28), Qt.AlignmentFlag.AlignCenter, "Starting video, telemetry and control link")
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
    return splash


async def _connect_hardware(name: str, connect_coro, timeout_s: float) -> None:
    logger = logging.getLogger("motocam.main")
    try:
        logger.info("Starting %s connect", name)
        await asyncio.wait_for(connect_coro(), timeout=timeout_s)
        logger.info("%s connect finished", name)
    except asyncio.TimeoutError:
        logger.warning("%s connect timed out after %.1fs; UI stays live and refresh/retry will continue", name, timeout_s)
    except Exception as exc:  # noqa: BLE001 -- startup must degrade, not freeze the app
        logger.warning("%s connect failed during startup: %s", name, exc)


async def async_connect_all(gimbal: GimbalController, camera: CameraController) -> None:
    await asyncio.gather(
        _connect_hardware("gimbal", gimbal.connect, 8.0),
        _connect_hardware("camera", camera.connect, 3.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MotoCam - motorcycle AI camera control unit")
    parser.add_argument("--config", help="Path to config.yaml", default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    _write_startup_state(config_path, "config_resolved")
    cfg = load_config(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    setup_logging(cfg.get("logging", {}).get("dir", "logs"))
    logger = logging.getLogger("motocam.main")
    install_crash_guard(logging.getLogger("motocam"))
    startup_log_buffer = StartupLogBuffer()
    logging.getLogger("motocam").addHandler(startup_log_buffer)
    logger.info("Starting MotoCam")
    _log_boot_config(logger, cfg, config_path)
    _write_startup_state(
        config_path,
        "config_loaded",
        {
            "unit_id": cfg.get("unit_id", "moto-1"),
            "ai": (cfg.get("ai") or {}).get("type"),
            "gimbal": (cfg.get("gimbal") or {}).get("connection"),
        },
    )

    # Global UI scale: the whole app was sized in logical pixels on a
    # desktop; on the real Pi touchscreen everything reads too small for
    # gloved operation (design doc 11.1). QT_SCALE_FACTOR multiplies the
    # ENTIRE UI -- fonts, buttons, chips, spacing -- uniformly, so one
    # config value fixes "everything is small" without touching any
    # widget. Must be set before QApplication is constructed.
    display_cfg = cfg.get("display", {})
    ui_scale = float(display_cfg.get("ui_scale", 1.0))
    if ui_scale != 1.0 and "QT_SCALE_FACTOR" not in os.environ:
        os.environ["QT_SCALE_FACTOR"] = f"{ui_scale:.3f}"
        logger.info("UI scale factor set to %.2f", ui_scale)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    app.setStyleSheet(DARK_STYLESHEET)
    logger.info("Qt application created")
    _write_startup_state(config_path, "qt_app_created")
    splash = create_splash()
    splash.show()
    app.processEvents()
    logger.info("Splash screen shown")
    _write_startup_state(config_path, "splash_shown")

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    logger.info("Qt asyncio event loop installed")
    _write_startup_state(config_path, "event_loop_installed")

    logger.info("Building video backend")
    video_engine = build_video(cfg)
    logger.info("Building gimbal backend")
    gimbal = build_gimbal(cfg)
    logger.info("Building camera backend")
    camera = build_camera(cfg)
    logger.info("Building GPS backend")
    gps = build_gps(cfg)
    health = HealthMonitor()
    logger.info("Hardware backends created")
    _write_startup_state(config_path, "backends_created")

    telemetry_cfg = cfg.get("telemetry", {})
    link = LinkClient(
        url=telemetry_cfg.get("control_room_url", "ws://127.0.0.1:8765"),
        unit_id=cfg.get("unit_id", "moto-1"),
    )
    install_control_room_log_forwarder(logging.getLogger("motocam"), link)
    startup_log_buffer.replay_to(link)
    logging.getLogger("motocam").removeHandler(startup_log_buffer)
    logger.info("Control room link created: %s", telemetry_cfg.get("control_room_url", "ws://127.0.0.1:8765"))
    _write_startup_state(config_path, "link_created")

    camera_cfg = cfg.get("camera", {})
    logger.info("Creating main window")
    window = MainWindow(
        video_engine, gimbal, camera, gps, link, health,
        unit_id=cfg.get("unit_id", "moto-1"),
        camera_ip=camera_cfg.get("ip", "192.168.9.20"),
        camera_port=camera_cfg.get("rest_port", 80),
        config=cfg,
        config_path=config_path,
    )
    logger.info("Main window created")
    _write_startup_state(config_path, "main_window_created")
    # Fullscreen on the kiosk touchscreen (no title bar / desktop taskbar
    # eating space); windowed on a dev desktop. Config-driven so the Mac
    # stays windowed.
    fullscreen = bool(display_cfg.get("fullscreen", False))
    if sys.platform == "darwin" and fullscreen and not bool(display_cfg.get("allow_macos_fullscreen", False)):
        logger.info("macOS fullscreen disabled by display.allow_macos_fullscreen=false")
        fullscreen = False

    if fullscreen:
        window.showFullScreen()
        logger.info("Main window shown fullscreen")
    else:
        window.show()
        logger.info("Main window shown windowed")
    _write_startup_state(config_path, "main_window_shown", {"fullscreen": fullscreen})

    app.processEvents()
    splash.finish(window)
    logger.info("Splash screen finished")
    _write_startup_state(config_path, "splash_finished")
    QTimer.singleShot(0, window.start_runtime)

    # Field canary: warns to the log if anything ever blocks the UI thread
    # again (all the heavy work is now off it). Silent on a healthy unit.
    ui_watchdog = UiLatencyWatchdog(parent=window, context_provider=window.pipeline_diagnostics_summary)
    ui_watchdog.start()
    logger.info("UI watchdog started")
    heartbeat = _install_runtime_heartbeat(config_path, window)
    logger.info("Runtime heartbeat started")

    with loop:
        logger.info("Entering Qt event loop")
        _write_startup_state(config_path, "event_loop_entered")
        loop.run_forever()

    heartbeat.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
