"""Entry point. Wires config -> hardware backends -> UI and runs the
Qt event loop with qasync so async gimbal/camera/network code shares
it with the GUI thread (no extra worker threads needed)."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
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
from motocam.core.logging_setup import install_control_room_log_forwarder, install_crash_guard, setup_logging
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
    return VideoEngine(
        device=device,
        width=video_cfg.get("width", 1920),
        height=video_cfg.get("height", 1080),
        fps=video_cfg.get("fps", 30),
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


async def async_connect_all(gimbal: GimbalController, camera: CameraController) -> None:
    await asyncio.gather(gimbal.connect(), camera.connect())


def main() -> int:
    parser = argparse.ArgumentParser(description="MotoCam - motorcycle AI camera control unit")
    parser.add_argument("--config", help="Path to config.yaml", default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    setup_logging(cfg.get("logging", {}).get("dir", "logs"))
    logger = logging.getLogger("motocam.main")
    install_crash_guard(logging.getLogger("motocam"))
    logger.info("Starting MotoCam")

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
    splash = create_splash()
    splash.show()
    app.processEvents()

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    video_engine = build_video(cfg)
    gimbal = build_gimbal(cfg)
    camera = build_camera(cfg)
    gps = build_gps(cfg)
    health = HealthMonitor()

    telemetry_cfg = cfg.get("telemetry", {})
    link = LinkClient(
        url=telemetry_cfg.get("control_room_url", "ws://127.0.0.1:8765"),
        unit_id=cfg.get("unit_id", "moto-1"),
    )
    install_control_room_log_forwarder(logging.getLogger("motocam"), link)

    camera_cfg = cfg.get("camera", {})
    window = MainWindow(
        video_engine, gimbal, camera, gps, link, health,
        unit_id=cfg.get("unit_id", "moto-1"),
        camera_ip=camera_cfg.get("ip", "192.168.9.20"),
        camera_port=camera_cfg.get("rest_port", 80),
        config=cfg,
        config_path=config_path,
    )
    # Fullscreen on the kiosk touchscreen (no title bar / desktop taskbar
    # eating space); windowed on a dev desktop. Config-driven so the Mac
    # stays windowed.
    if bool(display_cfg.get("fullscreen", False)):
        window.showFullScreen()
    else:
        window.show()
    QTimer.singleShot(1200, lambda: splash.finish(window))

    gps.open()
    video_engine.start()
    link.start()

    # Field canary: warns to the log if anything ever blocks the UI thread
    # again (all the heavy work is now off it). Silent on a healthy unit.
    ui_watchdog = UiLatencyWatchdog(parent=window)
    ui_watchdog.start()

    with loop:
        loop.create_task(async_connect_all(gimbal, camera))
        loop.run_forever()

    return 0


if __name__ == "__main__":
    sys.exit(main())
