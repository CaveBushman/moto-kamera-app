"""Builds the configured gimbal backend.

Kept outside main.py so Settings can rebuild the backend live after the
operator switches between mock, BLE, CAN, and UART.
"""
from __future__ import annotations

import logging
from typing import Any

from motocam.gimbal.base import GimbalBackend, GimbalController
from motocam.gimbal.mock_gimbal import MockGimbalBackend

logger = logging.getLogger("motocam.gimbal.factory")


def build_gimbal_controller(cfg: dict[str, Any]) -> GimbalController:
    gimbal_cfg = cfg.get("gimbal", {})
    return GimbalController(
        build_gimbal_backend(gimbal_cfg),
        max_pan_speed=gimbal_cfg.get("max_pan_speed", 20.0),
        max_tilt_speed=gimbal_cfg.get("max_tilt_speed", 12.0),
    )


def build_gimbal_backend(gimbal_cfg: dict[str, Any]) -> GimbalBackend:
    connection = str(gimbal_cfg.get("connection", "mock")).lower()
    if gimbal_cfg.get("type") == "dji_rs4_pro" and connection in ("ble", "can", "uart"):
        from motocam.gimbal.dji_rs4pro import BleTransport, CanTransport, DjiRs4ProBackend, UartTransport

        try:
            if connection == "ble":
                transport = BleTransport(
                    address=_blank_to_none(gimbal_cfg.get("ble_address")),
                    name=str(gimbal_cfg.get("ble_name", "RS 4 Pro")),
                    service_uuid=_blank_to_none(gimbal_cfg.get("ble_service_uuid")),
                    tx_char_uuid=_blank_to_none(gimbal_cfg.get("ble_tx_char_uuid")),
                    rx_char_uuid=_blank_to_none(gimbal_cfg.get("ble_rx_char_uuid")),
                    scan_timeout_s=float(gimbal_cfg.get("ble_scan_timeout_s", 8.0)),
                    mtu_payload_bytes=int(gimbal_cfg.get("ble_mtu_payload_bytes", 20)),
                )
            elif connection == "can":
                transport = CanTransport(
                    channel=str(gimbal_cfg.get("can_channel", "can0")),
                    bitrate=int(gimbal_cfg.get("can_bitrate", 1_000_000)),
                )
            else:
                transport = UartTransport(
                    device=str(gimbal_cfg.get("uart_device", "/dev/ttyAMA0")),
                    baudrate=int(gimbal_cfg.get("uart_baudrate", 115200)),
                )
            return DjiRs4ProBackend(
                transport,
                max_pan_speed=float(gimbal_cfg.get("max_pan_speed", 20.0)),
                max_tilt_speed=float(gimbal_cfg.get("max_tilt_speed", 12.0)),
                velocity_send_timeout_s=float(gimbal_cfg.get("ble_velocity_timeout_s", 0.15)),
                notify_stale_s=float(gimbal_cfg.get("ble_notify_stale_s", 3.0)),
            )
        except RuntimeError as exc:
            logger.warning("R SDK transport unavailable (%s) -- falling back to mock gimbal", exc)

    return MockGimbalBackend()


def _blank_to_none(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
