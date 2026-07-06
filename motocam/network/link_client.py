"""WebSocket link to the control room (design doc 6.4, 20).

Runs on the same asyncio loop that qasync drives inside the Qt event
loop (see motocam/main.py) -- no extra threads needed. Reconnects with
a fixed backoff if the control room is unreachable, which is the
normal case whenever the moto unit is off the Peplink network.
"""
from __future__ import annotations

import asyncio
import base64
from collections import deque
import logging
import time
from urllib.parse import urlparse

import websockets
from PyQt6.QtCore import QObject, pyqtSignal
from websockets.exceptions import ConnectionClosed

from motocam.core.protocol import Envelope, MessageType, Telemetry, make_envelope

logger = logging.getLogger("motocam.network")

RECONNECT_DELAY_S = 3.0
PING_INTERVAL_S = 5.0
LOG_QUEUE_MAX = 5000
LOG_DRAIN_BATCH_MAX = 100


class LinkClient(QObject):
    connected_changed = pyqtSignal(bool)
    command_received = pyqtSignal(str, dict)  # (MessageType value, payload)
    error_received = pyqtSignal(str, str)  # code, message
    latency_updated = pyqtSignal(float)

    def __init__(self, url: str, unit_id: str = "moto-1"):
        super().__init__()
        self.url = url
        self.unit_id = unit_id
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._connected = False
        self._task: asyncio.Task | None = None
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = None
        self._last_ping_sent: float | None = None
        self._telemetry_send_task: asyncio.Task | None = None
        self._pending_telemetry: Telemetry | None = None
        self._preview_send_task: asyncio.Task | None = None
        self._pending_preview_jpeg: bytes | None = None
        self._log_queue: deque[Envelope] = deque()
        self._log_send_task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        self._task = asyncio.ensure_future(self._run_forever())

    async def stop(self) -> None:
        self._cancel_send_tasks()
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
        self._cancel_send_tasks()
        if self._task is not None:
            self._task.cancel()
        if self._ws is not None:
            asyncio.ensure_future(self._ws.close())
        self._set_connected(False)
        self.start()

    async def _run_forever(self) -> None:
        while True:
            try:
                self._validate_url()
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws
                    self._set_connected(True)
                    logger.info("Connected to control room at %s", self.url)
                    await self._send(make_envelope(MessageType.HELLO, {"name": self.unit_id}, self.unit_id))
                    await self._receive_loop(ws)
            except (ValueError, OSError, ConnectionClosed) as exc:
                logger.warning("Control room link unavailable (%s), retrying in %.0fs", exc, RECONNECT_DELAY_S)
            finally:
                self._ws = None
                self._set_connected(False)
            await asyncio.sleep(RECONNECT_DELAY_S)

    def _validate_url(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError(f"invalid control-room URL scheme: {self.url!r}")
        if not parsed.hostname:
            raise ValueError(f"control-room host is empty: {self.url!r}")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"invalid control-room port: {self.url!r}") from exc
        if port is None:
            raise ValueError(f"control-room port is missing: {self.url!r}")

    async def _receive_loop(self, ws) -> None:
        async for raw in ws:
            envelope = Envelope.from_json(raw)
            if envelope.type == MessageType.PING.value:
                await self._send(make_envelope(MessageType.PONG, {}, self.unit_id))
                continue
            if envelope.type == MessageType.PONG.value and self._last_ping_sent is not None:
                self.latency_updated.emit((time.time() - self._last_ping_sent) * 1000)
                continue
            if envelope.type == MessageType.ERROR.value:
                code = str(envelope.payload.get("code", "link_error"))
                message = str(envelope.payload.get("message", code))
                logger.error("Control room rejected link: %s (%s)", message, code)
                self.error_received.emit(code, message)
                await ws.close()
                return
            self.command_received.emit(envelope.type, envelope.payload)

    def _set_connected(self, value: bool) -> None:
        if value != self._connected:
            self._connected = value
            self.connected_changed.emit(value)
        if value:
            self._start_log_queue_drain()

    async def _send(self, envelope: Envelope) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(envelope.to_json())
        except ConnectionClosed:
            pass

    def send_telemetry(self, telemetry: Telemetry) -> None:
        if self._telemetry_send_task is not None and not self._telemetry_send_task.done():
            self._pending_telemetry = telemetry
            return
        self._start_telemetry_send(telemetry)

    def send_preview_frame(self, jpeg_bytes: bytes) -> None:
        if self._preview_send_task is not None and not self._preview_send_task.done():
            self._pending_preview_jpeg = jpeg_bytes
            return
        self._start_preview_send(jpeg_bytes)

    def _start_telemetry_send(self, telemetry: Telemetry) -> None:
        self._telemetry_send_task = self._schedule_send(
            make_envelope(MessageType.TELEMETRY, telemetry, self.unit_id),
            "telemetry send",
            self._on_telemetry_send_done,
        )

    def _on_telemetry_send_done(self, task: asyncio.Task) -> None:
        self._consume_send_task(task, "telemetry send")
        if self._telemetry_send_task is task:
            self._telemetry_send_task = None
        pending = self._pending_telemetry
        self._pending_telemetry = None
        if pending is not None:
            self._start_telemetry_send(pending)

    def _start_preview_send(self, jpeg_bytes: bytes) -> None:
        payload = {"jpeg_b64": base64.b64encode(jpeg_bytes).decode("ascii")}
        self._preview_send_task = self._schedule_send(
            make_envelope(MessageType.PREVIEW_FRAME, payload, self.unit_id),
            "preview send",
            self._on_preview_send_done,
        )

    def _on_preview_send_done(self, task: asyncio.Task) -> None:
        self._consume_send_task(task, "preview send")
        if self._preview_send_task is task:
            self._preview_send_task = None
        pending = self._pending_preview_jpeg
        self._pending_preview_jpeg = None
        if pending is not None:
            self._start_preview_send(pending)

    def _schedule_send(self, envelope: Envelope, label: str, callback=None) -> asyncio.Task:
        task = asyncio.ensure_future(self._send(envelope))
        if callback is not None:
            task.add_done_callback(callback)
        else:
            task.add_done_callback(lambda done_task, task_label=label: self._consume_send_task(done_task, task_label))
        return task

    def _consume_send_task(self, task: asyncio.Task, label: str) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", label, exc)

    def _cancel_send_tasks(self) -> None:
        for task in (self._telemetry_send_task, self._preview_send_task, self._log_send_task):
            if task is not None and not task.done():
                task.cancel()
        self._telemetry_send_task = None
        self._pending_telemetry = None
        self._preview_send_task = None
        self._pending_preview_jpeg = None
        self._log_send_task = None
        self._log_queue.clear()

    def send_log_event(self, level: str, message: str, module: str = "") -> None:
        payload = {"level": level, "message": message, "module": module, "ts": time.time()}
        envelope = make_envelope(MessageType.LOG_EVENT, payload, self.unit_id)

        def schedule() -> None:
            while len(self._log_queue) >= LOG_QUEUE_MAX:
                self._log_queue.popleft()
            self._log_queue.append(envelope)
            self._start_log_queue_drain()

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if self._loop is None and running_loop is None:
            return
        if self._loop is not None and running_loop is not self._loop:
            self._loop.call_soon_threadsafe(schedule)
            return
        schedule()

    def _start_log_queue_drain(self) -> None:
        if not self._connected or self._ws is None:
            return
        if self._log_send_task is None or self._log_send_task.done():
            self._log_send_task = asyncio.ensure_future(self._drain_log_queue())

    async def _drain_log_queue(self) -> None:
        sent_in_batch = 0
        while self._log_queue and self._ws is not None and self._connected:
            envelope = self._log_queue[0]
            try:
                await self._ws.send(envelope.to_json())
                self._log_queue.popleft()
            except Exception as exc:  # noqa: BLE001
                # Do not log through the motocam logger here: this method
                # *is* the log forwarder path, and recursive warning logs can
                # refill the queue during an unstable websocket reconnect.
                logging.getLogger("motocam.network.forwarder").debug("log send failed: %s", exc)
                return
            sent_in_batch += 1
            if sent_in_batch >= LOG_DRAIN_BATCH_MAX:
                sent_in_batch = 0
                await asyncio.sleep(0)

    def send_ping(self) -> None:
        self._last_ping_sent = time.time()
        self._schedule_send(make_envelope(MessageType.PING, {}, self.unit_id), "ping send")

    def send_ptt_start(self) -> None:
        self._schedule_send(make_envelope(MessageType.PTT_START, {}, self.unit_id), "ptt start send")

    def send_ptt_audio(self, pcm_bytes: bytes, sample_rate: int) -> None:
        payload = {"audio_b64": base64.b64encode(pcm_bytes).decode("ascii"), "sample_rate": sample_rate}
        self._schedule_send(make_envelope(MessageType.PTT_AUDIO, payload, self.unit_id), "ptt audio send")

    def send_ptt_stop(self) -> None:
        self._schedule_send(make_envelope(MessageType.PTT_STOP, {}, self.unit_id), "ptt stop send")
