"""WebSocket link to the control room (design doc 6.4, 20).

Runs on the same asyncio loop that qasync drives inside the Qt event
loop (see motocam/main.py) -- no extra threads needed. Reconnects with
a fixed backoff if the control room is unreachable, which is the
normal case whenever the moto unit is off the Peplink network.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time

import websockets
from PyQt6.QtCore import QObject, pyqtSignal
from websockets.exceptions import ConnectionClosed

from motocam.core.protocol import Envelope, MessageType, Telemetry, make_envelope

logger = logging.getLogger("motocam.network")

RECONNECT_DELAY_S = 3.0
PING_INTERVAL_S = 5.0


class LinkClient(QObject):
    connected_changed = pyqtSignal(bool)
    command_received = pyqtSignal(str, dict)  # (MessageType value, payload)
    latency_updated = pyqtSignal(float)

    def __init__(self, url: str, unit_id: str = "moto-1"):
        super().__init__()
        self.url = url
        self.unit_id = unit_id
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._connected = False
        self._task: asyncio.Task | None = None
        self._last_ping_sent: float | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._run_forever())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._ws is not None:
            await self._ws.close()

    def reconfigure(self, url: str, unit_id: str) -> None:
        """Apply a new control-room URL / unit id from the settings UI and
        force an immediate reconnect (design doc 21 config -- exposed live
        instead of requiring a config.yaml edit + restart)."""
        self.url = url
        self.unit_id = unit_id
        if self._task is not None:
            self._task.cancel()
        if self._ws is not None:
            asyncio.ensure_future(self._ws.close())
        self._set_connected(False)
        self.start()

    async def _run_forever(self) -> None:
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws
                    self._set_connected(True)
                    logger.info("Connected to control room at %s", self.url)
                    await self._send(make_envelope(MessageType.HELLO, {"name": self.unit_id}, self.unit_id))
                    await self._receive_loop(ws)
            except (OSError, ConnectionClosed) as exc:
                logger.warning("Control room link unavailable (%s), retrying in %.0fs", exc, RECONNECT_DELAY_S)
            finally:
                self._ws = None
                self._set_connected(False)
            await asyncio.sleep(RECONNECT_DELAY_S)

    async def _receive_loop(self, ws) -> None:
        async for raw in ws:
            envelope = Envelope.from_json(raw)
            if envelope.type == MessageType.PING.value:
                await self._send(make_envelope(MessageType.PONG, {}, self.unit_id))
                continue
            if envelope.type == MessageType.PONG.value and self._last_ping_sent is not None:
                self.latency_updated.emit((time.time() - self._last_ping_sent) * 1000)
                continue
            self.command_received.emit(envelope.type, envelope.payload)

    def _set_connected(self, value: bool) -> None:
        if value != self._connected:
            self._connected = value
            self.connected_changed.emit(value)

    async def _send(self, envelope: Envelope) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(envelope.to_json())
        except ConnectionClosed:
            pass

    def send_telemetry(self, telemetry: Telemetry) -> None:
        asyncio.ensure_future(self._send(make_envelope(MessageType.TELEMETRY, telemetry, self.unit_id)))

    def send_preview_frame(self, jpeg_bytes: bytes) -> None:
        payload = {"jpeg_b64": base64.b64encode(jpeg_bytes).decode("ascii")}
        asyncio.ensure_future(self._send(make_envelope(MessageType.PREVIEW_FRAME, payload, self.unit_id)))

    def send_log_event(self, level: str, message: str, module: str = "") -> None:
        payload = {"level": level, "message": message, "module": module}
        asyncio.ensure_future(self._send(make_envelope(MessageType.LOG_EVENT, payload, self.unit_id)))

    def send_ping(self) -> None:
        self._last_ping_sent = time.time()
        asyncio.ensure_future(self._send(make_envelope(MessageType.PING, {}, self.unit_id)))

    def send_ptt_start(self) -> None:
        asyncio.ensure_future(self._send(make_envelope(MessageType.PTT_START, {}, self.unit_id)))

    def send_ptt_audio(self, pcm_bytes: bytes, sample_rate: int) -> None:
        payload = {"audio_b64": base64.b64encode(pcm_bytes).decode("ascii"), "sample_rate": sample_rate}
        asyncio.ensure_future(self._send(make_envelope(MessageType.PTT_AUDIO, payload, self.unit_id)))

    def send_ptt_stop(self) -> None:
        asyncio.ensure_future(self._send(make_envelope(MessageType.PTT_STOP, {}, self.unit_id)))
