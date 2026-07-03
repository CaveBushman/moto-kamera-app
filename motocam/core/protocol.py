"""Wire protocol shared between the moto camera unit and the control room.

This module is the single source of truth for the JSON message schema
exchanged over the WebSocket link (see network/link_client.py on the moto
side and network/link_server.py on the control room side). The control
room project keeps an identical copy at controlroom/core/protocol.py --
if you change one, change the other.

Envelope shape on the wire:
    {"type": "<MessageType>", "ts": <unix ms float>, "unit_id": "<str>", "payload": {...}}
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    # unit -> control room
    HELLO = "hello"
    TELEMETRY = "telemetry"
    PREVIEW_FRAME = "preview_frame"
    LOG_EVENT = "log_event"
    PONG = "pong"
    PTT_START = "ptt_start"
    PTT_AUDIO = "ptt_audio"
    PTT_STOP = "ptt_stop"

    # control room -> unit
    SET_MODE = "set_mode"
    CAMERA_COMMAND = "camera_command"
    GIMBAL_COMMAND = "gimbal_command"
    REQUEST_PREVIEW = "request_preview"
    STAGE_INFO = "stage_info"
    SWITCHER_STATE = "switcher_state"
    PING = "ping"


class OperatingMode(str, Enum):
    MANUAL = "manual"
    AI_ASSIST = "ai_assist"
    FULL_AI = "full_ai"
    LOCK = "lock"
    HOME = "home"
    RESET = "reset"


class TargetState(str, Enum):
    IDLE = "idle"
    LOCKED = "locked"
    WEAK = "weak"
    LOST = "lost"
    MANUAL_REQUIRED = "manual_required"


@dataclass
class GpsTelemetry:
    lat: float | None = None
    lon: float | None = None
    speed_kmh: float | None = None
    heading_deg: float | None = None
    fix: bool = False
    satellites: int = 0
    utc: str | None = None


@dataclass
class AiTelemetry:
    enabled: bool = False
    state: str = TargetState.IDLE.value
    target_id: int | None = None
    inference_fps: float = 0.0


@dataclass
class GimbalTelemetry:
    connected: bool = False
    mode: str = OperatingMode.MANUAL.value
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    roll_deg: float = 0.0


@dataclass
class CameraTelemetry:
    connected: bool = False
    recording: bool = False
    iso: int | None = None
    white_balance: int | None = None
    shutter: str | None = None
    iris: str | None = None
    fps: float | None = None
    media_remaining_min: float | None = None
    battery_pct: float | None = None
    # These describe the low-fps JPEG *preview relay* to the control room
    # (design doc section 5), not the PYXIS's own SDI recording -- that's a
    # hardware-encoder path this software has no visibility into.
    # Deliberately kept separate so a
    # director never mistakes preview bandwidth for broadcast quality.
    preview_resolution: str | None = None
    preview_bitrate_kbps: float | None = None


@dataclass
class NetworkTelemetry:
    latency_ms: float | None = None
    link_up: bool = True


@dataclass
class SystemTelemetry:
    cpu_temp_c: float | None = None
    cpu_load_pct: float | None = None
    ram_used_pct: float | None = None
    video_fps: float = 0.0


@dataclass
class StageInfo:
    """Race stage identity, set once by the director and broadcast to every
    connected unit (design doc has no dedicated section for this -- it's a
    control-room-driven addition: the moto app just displays whatever the
    control room last told it, it never originates this itself).

    The route GPX itself stays local to the control room (director
    reference / future route-progress display) -- pushing trackpoint data
    over this link isn't worth the bandwidth for what's currently just a
    banner heading, so only the lap count crosses the wire."""
    stage_number: int = 0
    stage_name: str = ""
    laps: int = 0


@dataclass
class SwitcherState:
    preview_unit: str | None = None
    live_unit: str | None = None
    preview_input: int | str | None = None
    live_input: int | str | None = None


@dataclass
class Telemetry:
    gps: GpsTelemetry = field(default_factory=GpsTelemetry)
    ai: AiTelemetry = field(default_factory=AiTelemetry)
    gimbal: GimbalTelemetry = field(default_factory=GimbalTelemetry)
    camera: CameraTelemetry = field(default_factory=CameraTelemetry)
    network: NetworkTelemetry = field(default_factory=NetworkTelemetry)
    system: SystemTelemetry = field(default_factory=SystemTelemetry)


@dataclass
class Envelope:
    type: str
    payload: dict[str, Any]
    unit_id: str = "moto-1"
    ts: float = field(default_factory=lambda: time.time() * 1000)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @staticmethod
    def from_json(raw: str) -> "Envelope":
        data = json.loads(raw)
        return Envelope(
            type=data["type"],
            payload=data.get("payload", {}),
            unit_id=data.get("unit_id", "moto-1"),
            ts=data.get("ts", time.time() * 1000),
        )


def make_envelope(msg_type: MessageType, payload: dict[str, Any] | Any, unit_id: str = "moto-1") -> Envelope:
    if not isinstance(payload, dict):
        payload = asdict(payload)
    return Envelope(type=msg_type.value, payload=payload, unit_id=unit_id)


def parse_stage_info(payload: dict[str, Any]) -> StageInfo:
    return StageInfo(**{k: v for k, v in payload.items() if k in ("stage_number", "stage_name", "laps")})


def parse_switcher_state(payload: dict[str, Any]) -> SwitcherState:
    allowed = ("preview_unit", "live_unit", "preview_input", "live_input")
    return SwitcherState(**{k: v for k, v in payload.items() if k in allowed})
