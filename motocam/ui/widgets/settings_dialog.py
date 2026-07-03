"""Camera/lens settings and AI tracking settings (design doc 6.3, 8.5, 9.4).

Two group boxes in one dialog rather than a tab-per-concern: this is a
one-operator touchscreen unit, not a multi-page config app -- everything
that's adjustable in the field fits on one scrollable-free screen.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from motocam.audio.devices import list_audio_devices
from motocam.video.devices import list_video_devices

logger = logging.getLogger("motocam.ui.settings")

ISO_VALUES = [100, 200, 400, 800, 1600, 3200, 6400]
WB_VALUES = [2800, 3200, 4300, 5000, 5600, 6500, 7500]
SHUTTER_VALUES = ["1/25", "1/50", "1/100", "1/200", "1/400", "180°", "172.8°"]
IRIS_VALUES = ["f/1.4", "f/2.0", "f/2.8", "f/4.0", "f/5.6", "f/8.0", "f/11"]
# Editable: "cyclist" suits a custom-trained HEF, but a stock COCO
# YOLOv8/v11 HEF has no "cyclist" class -- it detects "bicycle"/"person".
# The class name must match the loaded model's labels exactly or FULL-AI
# auto-acquire silently never fires, so the operator can type any label.
TARGET_CLASSES = ["cyclist", "bicycle", "person", "motorcycle", "car"]
UNIT_IDS = [f"moto-{i}" for i in range(1, 5)]
GIMBAL_CONNECTIONS = [
    ("Mock / desk test", "mock"),
    ("DJI RS4 Pro BLE", "ble"),
    ("DJI R SDK CAN", "can"),
    ("DJI R SDK UART", "uart"),
]
DEFAULT_CONTROL_ROOM_PORT = 8765
DEFAULT_PYXIS_PORT = 9993


class SettingsDialog(QDialog):
    iso_changed = pyqtSignal(int)
    white_balance_changed = pyqtSignal(int)
    shutter_changed = pyqtSignal(str)
    iris_changed = pyqtSignal(str)

    target_class_changed = pyqtSignal(str)
    confidence_changed = pyqtSignal(float)
    dead_zone_changed = pyqtSignal(int, int)
    max_speed_changed = pyqtSignal(float, float)

    connection_apply_requested = pyqtSignal(str, str)  # (unit_id, control_room_url)
    pyxis_apply_requested = pyqtSignal(str, int)  # (ip, port)
    audio_apply_requested = pyqtSignal(object, object)  # input_device, output_device
    video_device_apply_requested = pyqtSignal(object)  # device (int index or /dev/videoN path)
    gimbal_apply_requested = pyqtSignal(object)  # gimbal config dict
    ble_scan_requested = pyqtSignal(str)  # preferred BLE name filter
    safety_apply_requested = pyqtSignal(bool, float)  # (ride_lock_enabled, ride_lock_speed_kmh)
    exit_requested = pyqtSignal()  # quit the whole app (kiosk has no window chrome)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(640, 820)

        outer_layout = QVBoxLayout(self)

        # Top toolbar: HIDE KEYBOARD lives HERE, at the top, so it's always
        # reachable above the Pi on-screen keyboard -- which covers the
        # bottom of the screen and would otherwise hide a footer button
        # (the very button you need to dismiss the keyboard). Close sits
        # here too for the same reason.
        toolbar = QHBoxLayout()
        hide_kb_btn = QPushButton("⌨  HIDE KEYBOARD")
        hide_kb_btn.clicked.connect(self.hide_keyboard)
        toolbar.addWidget(hide_kb_btn)
        toolbar.addStretch(1)
        top_close_btn = QPushButton("Close")
        top_close_btn.clicked.connect(self._on_close)
        toolbar.addWidget(top_close_btn)
        outer_layout.addLayout(toolbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_video_group())
        layout.addWidget(self._build_audio_group())
        layout.addWidget(self._build_gimbal_group())
        layout.addWidget(self._build_camera_group())
        layout.addWidget(self._build_tracking_group())
        layout.addWidget(self._build_safety_group())
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        footer = QHBoxLayout()
        # Fullscreen kiosk has no window title bar / close button, so the
        # only way out of the app lives here. Confirmed so an accidental
        # tap mid-operation can't kill the live feed.
        exit_btn = QPushButton("EXIT MOTOCAM")
        exit_btn.setObjectName("exitButton")
        exit_btn.clicked.connect(self._confirm_exit)
        footer.addWidget(exit_btn)
        footer.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._on_close)
        footer.addWidget(close_btn)
        outer_layout.addLayout(footer)

        # Pressing Enter/Return in any field dismisses the OS keyboard too
        # (the squeekboard ↵ key sends returnPressed here).
        for line_edit in self.findChildren(QLineEdit):
            line_edit.returnPressed.connect(self.hide_keyboard)
        # Any button tap (APPLY, RESCAN, ...) also dismisses it -- the Pi
        # squeekboard doesn't hide on focus-out on its own, so a tap that
        # moves focus off a field otherwise leaves it stuck up. Idempotent
        # on the keyboard/close/exit buttons that already handle it.
        for button in self.findChildren(QPushButton):
            button.clicked.connect(self.hide_keyboard)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        # Fit the dialog to the screen and pin it to the top edge: the Pi
        # on-screen keyboard eats the bottom third, so a tall dialog runs
        # its footer off-screen behind the keyboard. Anchored at the top,
        # the toolbar (HIDE KEYBOARD / Close) is always reachable and the
        # scroll area handles the rest.
        screen = self.screen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        max_h = int(avail.height() * 0.95)
        if self.height() > max_h:
            self.resize(self.width(), max_h)
        # top-centered
        x = avail.x() + (avail.width() - self.width()) // 2
        self.move(max(avail.x(), x), avail.y() + 8)

    def hide_keyboard(self) -> None:
        """Dismiss the OS on-screen keyboard: drop input focus, ask the Qt
        platform input method to hide (works when Qt drives the Wayland
        text-input), and as a belt-and-braces fallback tell Raspberry Pi
        OS's squeekboard directly over DBus (works even when Qt runs under
        XWayland and never touched the Wayland text-input)."""
        widget = self.focusWidget()
        if widget is not None:
            widget.clearFocus()
        QGuiApplication.inputMethod().hide()
        self._squeekboard_hide()

    @staticmethod
    def _squeekboard_hide() -> None:
        """Tell Raspberry Pi OS's squeekboard to hide via its DBus method
        SetVisible(false).

        This MUST be a method call, not a property write: squeekboard's
        `Visible` property is read-only, so `busctl set-property ... Visible`
        errors out -- which is exactly why the button did nothing (busctl is
        found before dbus-send, so the working fallback was never reached).
        We now issue the SetVisible method on whichever tool exists, and if
        the first one errors we fall through to the next."""
        import shutil
        import subprocess
        import sys

        if not sys.platform.startswith("linux"):
            return
        attempts = []
        if shutil.which("busctl"):
            attempts.append([
                "busctl", "--user", "call", "sm.puri.OSK0",
                "/sm/puri/OSK0", "sm.puri.OSK0", "SetVisible", "b", "false",
            ])
        if shutil.which("dbus-send"):
            attempts.append([
                "dbus-send", "--type=method_call", "--dest=sm.puri.OSK0",
                "/sm/puri/OSK0", "sm.puri.OSK0.SetVisible", "boolean:false",
            ])
        for cmd in attempts:
            try:
                result = subprocess.run(cmd, timeout=1.0, capture_output=True, check=False)
            except Exception as exc:  # noqa: BLE001 -- absent dbus tool must stay silent
                logger.debug("squeekboard hide via %s failed: %s", cmd[0], exc)
                continue
            if result.returncode == 0:
                return
            logger.debug(
                "squeekboard hide via %s returned %d: %s",
                cmd[0], result.returncode, result.stderr.decode(errors="replace").strip(),
            )

    def _on_close(self) -> None:
        self.hide_keyboard()
        self.accept()

    def _confirm_exit(self) -> None:
        reply = QMessageBox.question(
            self,
            "Exit MotoCam",
            "Quit MotoCam?\n\nThis stops the camera control, tracking and the "
            "link to the control room on this unit.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.exit_requested.emit()

    # -- unit identity & control room link (design doc 6.2, 21) ---------------
    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("CONNECTION")
        form = QFormLayout(group)

        self.unit_id_combo = QComboBox()
        self.unit_id_combo.setEditable(True)
        self.unit_id_combo.addItems(UNIT_IDS)
        form.addRow("This unit", self.unit_id_combo)

        self.control_room_host_edit = QLineEdit()
        self.control_room_host_edit.setPlaceholderText("192.168.9.100")
        form.addRow("Control room IP", self.control_room_host_edit)

        self.control_room_port_spin = QSpinBox()
        self.control_room_port_spin.setRange(1, 65535)
        self.control_room_port_spin.setValue(DEFAULT_CONTROL_ROOM_PORT)
        form.addRow("Control room port", self.control_room_port_spin)

        apply_btn = QPushButton("APPLY && RECONNECT")
        apply_btn.clicked.connect(self._emit_connection_apply)
        form.addRow(apply_btn)

        return group

    def set_connection_values(self, unit_id: str, control_room_url: str) -> None:
        idx = self.unit_id_combo.findText(unit_id)
        if idx >= 0:
            self.unit_id_combo.setCurrentIndex(idx)
        else:
            self.unit_id_combo.setEditText(unit_id)
        host, port = self._parse_ws_url(control_room_url)
        self.control_room_host_edit.setText(host)
        self.control_room_port_spin.setValue(port)

    def _emit_connection_apply(self) -> None:
        unit_id = self.unit_id_combo.currentText().strip() or "moto-1"
        host = self.control_room_host_edit.text().strip()
        if not host:
            return
        url = f"ws://{host}:{self.control_room_port_spin.value()}"
        self.connection_apply_requested.emit(unit_id, url)

    @staticmethod
    def _parse_ws_url(url: str) -> tuple[str, int]:
        stripped = url.split("://", 1)[-1]
        if ":" in stripped:
            host, port_str = stripped.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = DEFAULT_CONTROL_ROOM_PORT
        else:
            host, port = stripped, DEFAULT_CONTROL_ROOM_PORT
        return host, port

    # -- video source ----------------------------------------------------------
    def _build_video_group(self) -> QGroupBox:
        group = QGroupBox("VIDEO SOURCE")
        outer = QVBoxLayout(group)

        note = QLabel(
            "The live preview feed comes from a UVC/V4L2 capture device (e.g. "
            "a Magewell HDMI/SDI grabber on the field unit, or a webcam for "
            "desk testing) -- separate from the PYXIS IP control link above, "
            "which only handles exposure/lens commands."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        form = QFormLayout()
        outer.addLayout(form)

        self.video_device_combo = QComboBox()
        form.addRow("Capture device", self.video_device_combo)

        refresh_btn = QPushButton("RESCAN DEVICES")
        refresh_btn.clicked.connect(lambda: self._populate_video_combo(self.video_device_combo.currentData()))
        form.addRow(refresh_btn)

        apply_btn = QPushButton("APPLY VIDEO SOURCE")
        apply_btn.clicked.connect(self._emit_video_apply)
        form.addRow(apply_btn)

        return group

    def set_video_values(self, current_device: int | str | None) -> None:
        self._populate_video_combo(current_device)

    def _populate_video_combo(self, current_device: int | str | None) -> None:
        combo = self.video_device_combo
        combo.blockSignals(True)
        combo.clear()
        devices = list_video_devices()
        if not devices:
            combo.addItem("No capture device found (synthetic preview)", current_device)
        for device in devices:
            combo.addItem(device.label, device.device)
        selected = 0
        for index in range(combo.count()):
            if combo.itemData(index) == current_device:
                selected = index
                break
        combo.setCurrentIndex(selected)
        combo.blockSignals(False)

    def _emit_video_apply(self) -> None:
        self.video_device_apply_requested.emit(self.video_device_combo.currentData())

    # -- audio ---------------------------------------------------------------
    def _build_audio_group(self) -> QGroupBox:
        group = QGroupBox("AUDIO")
        form = QFormLayout(group)

        self.input_device_combo = QComboBox()
        form.addRow("PTT microphone", self.input_device_combo)

        self.output_device_combo = QComboBox()
        form.addRow("Talkback speaker", self.output_device_combo)

        apply_btn = QPushButton("APPLY AUDIO")
        apply_btn.clicked.connect(self._emit_audio_apply)
        form.addRow(apply_btn)

        return group

    def set_audio_values(self, input_device: int | str | None, output_device: int | str | None) -> None:
        self._populate_device_combo(self.input_device_combo, "input", input_device)
        self._populate_device_combo(self.output_device_combo, "output", output_device)

    def _emit_audio_apply(self) -> None:
        self.audio_apply_requested.emit(self.input_device_combo.currentData(), self.output_device_combo.currentData())

    # -- gimbal --------------------------------------------------------------
    def _build_gimbal_group(self) -> QGroupBox:
        group = QGroupBox("GIMBAL CONTROL")
        outer = QVBoxLayout(group)

        note = QLabel(
            "DJI R SDK frames are sent over the selected transport. BLE can "
            "auto-discover write/notify characteristics; fill UUIDs only when "
            "auto-discovery is not reliable with the real RS4 Pro profile."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        form = QFormLayout()
        outer.addLayout(form)

        self.gimbal_connection_combo = QComboBox()
        for label, value in GIMBAL_CONNECTIONS:
            self.gimbal_connection_combo.addItem(label, value)
        form.addRow("Transport", self.gimbal_connection_combo)

        self.ble_device_combo = QComboBox()
        self.ble_device_combo.addItem("Scan to select BLE device", None)
        self.ble_device_combo.currentIndexChanged.connect(self._on_ble_device_selected)
        form.addRow("BLE device", self.ble_device_combo)

        self.ble_scan_button = QPushButton("SCAN BLE DEVICES")
        self.ble_scan_button.clicked.connect(self._emit_ble_scan)
        form.addRow(self.ble_scan_button)

        self.ble_scan_status_label = QLabel("")
        self.ble_scan_status_label.setWordWrap(True)
        form.addRow(self.ble_scan_status_label)

        self.ble_name_edit = QLineEdit()
        self.ble_name_edit.setPlaceholderText("RS 4 Pro")
        form.addRow("BLE name", self.ble_name_edit)

        self.ble_address_edit = QLineEdit()
        self.ble_address_edit.setPlaceholderText("Optional BLE address / UUID")
        form.addRow("BLE address", self.ble_address_edit)

        self.ble_service_uuid_edit = QLineEdit()
        self.ble_service_uuid_edit.setPlaceholderText("Optional service UUID")
        form.addRow("BLE service UUID", self.ble_service_uuid_edit)

        self.ble_tx_char_uuid_edit = QLineEdit()
        self.ble_tx_char_uuid_edit.setPlaceholderText("Optional write characteristic UUID")
        form.addRow("BLE write UUID", self.ble_tx_char_uuid_edit)

        self.ble_rx_char_uuid_edit = QLineEdit()
        self.ble_rx_char_uuid_edit.setPlaceholderText("Optional notify characteristic UUID")
        form.addRow("BLE notify UUID", self.ble_rx_char_uuid_edit)

        self.can_channel_edit = QLineEdit()
        self.can_channel_edit.setPlaceholderText("can0")
        form.addRow("CAN channel", self.can_channel_edit)

        self.uart_device_edit = QLineEdit()
        self.uart_device_edit.setPlaceholderText("/dev/ttyAMA0")
        form.addRow("UART device", self.uart_device_edit)

        self.uart_baudrate_spin = QSpinBox()
        self.uart_baudrate_spin.setRange(1200, 1_000_000)
        self.uart_baudrate_spin.setValue(115200)
        form.addRow("UART baudrate", self.uart_baudrate_spin)

        apply_btn = QPushButton("APPLY GIMBAL")
        apply_btn.clicked.connect(self._emit_gimbal_apply)
        form.addRow(apply_btn)

        return group

    def set_gimbal_values(self, gimbal_cfg: dict | None) -> None:
        cfg = gimbal_cfg or {}
        connection = str(cfg.get("connection", "mock")).lower()
        for index in range(self.gimbal_connection_combo.count()):
            if self.gimbal_connection_combo.itemData(index) == connection:
                self.gimbal_connection_combo.setCurrentIndex(index)
                break

        self.ble_name_edit.setText(str(cfg.get("ble_name", "RS 4 Pro")))
        self.ble_address_edit.setText(str(cfg.get("ble_address", "") or ""))
        self._set_current_ble_device_placeholder(self.ble_name_edit.text(), self.ble_address_edit.text())
        self.ble_service_uuid_edit.setText(str(cfg.get("ble_service_uuid", "") or ""))
        self.ble_tx_char_uuid_edit.setText(str(cfg.get("ble_tx_char_uuid", "") or ""))
        self.ble_rx_char_uuid_edit.setText(str(cfg.get("ble_rx_char_uuid", "") or ""))
        self.can_channel_edit.setText(str(cfg.get("can_channel", "can0")))
        self.uart_device_edit.setText(str(cfg.get("uart_device", "/dev/ttyAMA0")))
        self.uart_baudrate_spin.setValue(int(cfg.get("uart_baudrate", 115200)))

    def _emit_gimbal_apply(self) -> None:
        self.gimbal_apply_requested.emit(
            {
                "type": "dji_rs4_pro",
                "connection": self.gimbal_connection_combo.currentData() or "mock",
                "ble_name": self.ble_name_edit.text().strip() or "RS 4 Pro",
                "ble_address": self._text_or_none(self.ble_address_edit),
                "ble_service_uuid": self._text_or_none(self.ble_service_uuid_edit),
                "ble_tx_char_uuid": self._text_or_none(self.ble_tx_char_uuid_edit),
                "ble_rx_char_uuid": self._text_or_none(self.ble_rx_char_uuid_edit),
                "can_channel": self.can_channel_edit.text().strip() or "can0",
                "uart_device": self.uart_device_edit.text().strip() or "/dev/ttyAMA0",
                "uart_baudrate": self.uart_baudrate_spin.value(),
            }
        )

    def _emit_ble_scan(self) -> None:
        self.ble_scan_button.setEnabled(False)
        self.ble_scan_status_label.setText("Scanning...")
        self.ble_scan_requested.emit(self.ble_name_edit.text().strip() or "RS 4 Pro")

    def set_ble_scan_results(self, devices: list, error: str | None = None) -> None:
        self.ble_scan_button.setEnabled(True)
        if error:
            self.ble_scan_status_label.setText(error)
            return

        current_address = self.ble_address_edit.text().strip()
        self.ble_device_combo.blockSignals(True)
        self.ble_device_combo.clear()
        if not devices:
            self.ble_device_combo.addItem("No BLE devices found", None)
            self.ble_scan_status_label.setText("No BLE devices found.")
        else:
            self.ble_device_combo.addItem("Select BLE device", None)
            for device in devices:
                name = getattr(device, "name", "") or "Unknown BLE device"
                address = getattr(device, "address", "") or ""
                rssi = getattr(device, "rssi", None)
                suffix = f" ({rssi} dBm)" if rssi is not None else ""
                self.ble_device_combo.addItem(f"{name} - {address}{suffix}", {"name": name, "address": address})
            self.ble_scan_status_label.setText(f"{len(devices)} BLE device(s) found.")
        selected = 0
        if current_address:
            for index in range(self.ble_device_combo.count()):
                data = self.ble_device_combo.itemData(index)
                if data and data.get("address") == current_address:
                    selected = index
                    break
        self.ble_device_combo.setCurrentIndex(selected)
        self.ble_device_combo.blockSignals(False)

    def _on_ble_device_selected(self, index: int) -> None:
        data = self.ble_device_combo.itemData(index)
        if not data:
            return
        self._select_connection("ble")
        self.ble_name_edit.setText(data.get("name", "") or "RS 4 Pro")
        self.ble_address_edit.setText(data.get("address", "") or "")

    def _set_current_ble_device_placeholder(self, name: str, address: str) -> None:
        self.ble_device_combo.blockSignals(True)
        self.ble_device_combo.clear()
        self.ble_device_combo.addItem("Scan to select BLE device", None)
        if address:
            self.ble_device_combo.addItem(f"Current: {name or 'BLE device'} - {address}", {"name": name, "address": address})
            self.ble_device_combo.setCurrentIndex(1)
        self.ble_device_combo.blockSignals(False)

    def _select_connection(self, connection: str) -> None:
        for index in range(self.gimbal_connection_combo.count()):
            if self.gimbal_connection_combo.itemData(index) == connection:
                self.gimbal_connection_combo.setCurrentIndex(index)
                break

    # -- camera & lens --------------------------------------------------------
    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("CAMERA && LENS")  # "&&" escapes to a literal "&" (Qt mnemonic syntax)
        outer = QVBoxLayout(group)

        note = QLabel(
            "Blackmagic PYXIS 6K REST control is applied live. Individual "
            "camera or lens endpoints may report unavailable depending on "
            "firmware and attached lens."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        conn_form = QFormLayout()
        outer.addLayout(conn_form)

        self.pyxis_ip_edit = QLineEdit()
        self.pyxis_ip_edit.setPlaceholderText("192.168.9.20")
        conn_form.addRow("PYXIS 6K IP", self.pyxis_ip_edit)

        self.pyxis_port_spin = QSpinBox()
        self.pyxis_port_spin.setRange(1, 65535)
        self.pyxis_port_spin.setValue(DEFAULT_PYXIS_PORT)
        conn_form.addRow("PYXIS 6K port", self.pyxis_port_spin)

        pyxis_apply_btn = QPushButton("SAVE PYXIS ADDRESS")
        pyxis_apply_btn.clicked.connect(self._emit_pyxis_apply)
        conn_form.addRow(pyxis_apply_btn)

        form = QFormLayout()
        outer.addLayout(form)

        self.iso_combo = QComboBox()
        self.iso_combo.addItems([str(v) for v in ISO_VALUES])
        self.iso_combo.currentTextChanged.connect(lambda t: self.iso_changed.emit(int(t)))
        form.addRow("ISO", self.iso_combo)

        self.wb_combo = QComboBox()
        self.wb_combo.addItems([f"{v} K" for v in WB_VALUES])
        self.wb_combo.currentTextChanged.connect(
            lambda t: self.white_balance_changed.emit(int(t.split()[0]))
        )
        form.addRow("White balance", self.wb_combo)

        self.shutter_combo = QComboBox()
        self.shutter_combo.addItems(SHUTTER_VALUES)
        self.shutter_combo.currentTextChanged.connect(self.shutter_changed.emit)
        form.addRow("Shutter", self.shutter_combo)

        self.iris_combo = QComboBox()
        self.iris_combo.addItems(IRIS_VALUES)
        self.iris_combo.currentTextChanged.connect(self.iris_changed.emit)
        form.addRow("Iris", self.iris_combo)

        return group

    def set_camera_values(self, iso: int | None, white_balance: int | None, shutter: str | None, iris: str | None) -> None:
        self._select(self.iso_combo, str(iso) if iso is not None else None)
        self._select(self.wb_combo, f"{white_balance} K" if white_balance is not None else None)
        self._select(self.shutter_combo, shutter)
        self._select(self.iris_combo, iris)

    def set_pyxis_values(self, ip: str, port: int) -> None:
        self.pyxis_ip_edit.setText(ip)
        self.pyxis_port_spin.setValue(port)

    def _emit_pyxis_apply(self) -> None:
        ip = self.pyxis_ip_edit.text().strip()
        if not ip:
            return
        self.pyxis_apply_requested.emit(ip, self.pyxis_port_spin.value())

    # -- AI tracking ------------------------------------------------------------
    def _build_tracking_group(self) -> QGroupBox:
        group = QGroupBox("AI TRACKING")
        outer = QVBoxLayout(group)

        # A word-wrapped QLabel added straight to QFormLayout.addRow() can
        # get an under-computed height on first layout pass (height-for-width
        # isn't recalculated until a later resize) -- putting it in the
        # group's own QVBoxLayout instead avoids that and always sizes
        # correctly on first show.
        hint = QLabel(
            "Tap a rider in the live preview to lock tracking onto them. "
            "Switch to AI ASSIST or FULL AI mode for the gimbal to follow automatically."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        form = QFormLayout()
        outer.addLayout(form)

        self.target_class_combo = QComboBox()
        self.target_class_combo.setEditable(True)  # any label the loaded HEF exposes
        self.target_class_combo.addItems(TARGET_CLASSES)
        self.target_class_combo.currentTextChanged.connect(self.target_class_changed.emit)
        form.addRow("Target class", self.target_class_combo)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.05, 0.95)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.valueChanged.connect(self.confidence_changed.emit)
        form.addRow("Min. confidence", self.confidence_spin)

        self.dead_zone_x_spin = QSpinBox()
        self.dead_zone_x_spin.setRange(0, 200)
        self.dead_zone_x_spin.setSuffix(" px")
        self.dead_zone_x_spin.valueChanged.connect(self._emit_dead_zone)
        form.addRow("Dead zone (pan)", self.dead_zone_x_spin)

        self.dead_zone_y_spin = QSpinBox()
        self.dead_zone_y_spin.setRange(0, 200)
        self.dead_zone_y_spin.setSuffix(" px")
        self.dead_zone_y_spin.valueChanged.connect(self._emit_dead_zone)
        form.addRow("Dead zone (tilt)", self.dead_zone_y_spin)

        self.max_pan_speed_spin = QDoubleSpinBox()
        self.max_pan_speed_spin.setRange(1.0, 90.0)
        self.max_pan_speed_spin.setSuffix(" °/s")
        self.max_pan_speed_spin.valueChanged.connect(self._emit_max_speed)
        form.addRow("Max pan speed", self.max_pan_speed_spin)

        self.max_tilt_speed_spin = QDoubleSpinBox()
        self.max_tilt_speed_spin.setRange(1.0, 90.0)
        self.max_tilt_speed_spin.setSuffix(" °/s")
        self.max_tilt_speed_spin.valueChanged.connect(self._emit_max_speed)
        form.addRow("Max tilt speed", self.max_tilt_speed_spin)

        return group

    def set_tracking_values(
        self, target_class: str, confidence: float,
        dead_zone_x: int, dead_zone_y: int,
        max_pan_speed: float, max_tilt_speed: float,
    ) -> None:
        self._select(self.target_class_combo, target_class)
        for spin, value in (
            (self.confidence_spin, confidence),
            (self.dead_zone_x_spin, dead_zone_x),
            (self.dead_zone_y_spin, dead_zone_y),
            (self.max_pan_speed_spin, max_pan_speed),
            (self.max_tilt_speed_spin, max_tilt_speed),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _emit_dead_zone(self) -> None:
        self.dead_zone_changed.emit(self.dead_zone_x_spin.value(), self.dead_zone_y_spin.value())

    def _emit_max_speed(self) -> None:
        self.max_speed_changed.emit(self.max_pan_speed_spin.value(), self.max_tilt_speed_spin.value())

    # -- ride-safe lockout ------------------------------------------------------
    def _build_safety_group(self) -> QGroupBox:
        group = QGroupBox("SAFETY")
        outer = QVBoxLayout(group)

        hint = QLabel(
            "Above the speed below, SETTINGS and CAM/GIMBAL are grayed out so "
            "a gloved thumb isn't tempted into ISO/shutter/lens taps while "
            "riding. PTT, the joystick and zoom always stay live."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        form = QFormLayout()
        outer.addLayout(form)

        self.ride_lock_checkbox = QCheckBox("Lock camera/gimbal settings while riding")
        form.addRow(self.ride_lock_checkbox)

        self.ride_lock_speed_spin = QDoubleSpinBox()
        self.ride_lock_speed_spin.setRange(1.0, 150.0)
        self.ride_lock_speed_spin.setSuffix(" km/h")
        form.addRow("Lock above", self.ride_lock_speed_spin)

        apply_btn = QPushButton("APPLY SAFETY")
        apply_btn.clicked.connect(self._emit_safety_apply)
        form.addRow(apply_btn)

        return group

    def set_safety_values(self, enabled: bool, speed_kmh: float) -> None:
        self.ride_lock_checkbox.setChecked(enabled)
        self.ride_lock_speed_spin.setValue(speed_kmh)

    def _emit_safety_apply(self) -> None:
        self.safety_apply_requested.emit(self.ride_lock_checkbox.isChecked(), self.ride_lock_speed_spin.value())

    @staticmethod
    def _select(combo: QComboBox, text: str | None) -> None:
        if text is None:
            return
        idx = combo.findText(text)
        combo.blockSignals(True)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    @staticmethod
    def _populate_device_combo(combo: QComboBox, kind: str, current_device: int | str | None) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("System default", None)
        for device in list_audio_devices(kind):  # type: ignore[arg-type]
            combo.addItem(device.label, device.index)
        selected = 0
        for index in range(combo.count()):
            if combo.itemData(index) == current_device:
                selected = index
                break
        combo.setCurrentIndex(selected)
        combo.blockSignals(False)

    @staticmethod
    def _text_or_none(edit: QLineEdit) -> str | None:
        text = edit.text().strip()
        return text or None
