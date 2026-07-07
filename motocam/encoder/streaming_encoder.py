"""Blackmagic Streaming Encoder status monitor over its documented
Control REST API (Blackmagic Streaming REST API, January 2026).

The encoder is a separate device from the gimbal-mounted camera: it takes
a clean SDI/HDMI tap of the same camera signal and pushes it to a CDN/
YouTube/etc. over RTMP or SRT as the broadcast uplink from the bike to
the van, independent of this app's own low-fps JPEG preview relay (see
video/preview_relay.py) and of the Blackmagic Camera Control REST link
(camera/bmd_rest_camera.py) -- three unrelated network links that happen
to share a REST-over-HTTP style.

Read-only monitor: exposes ON AIR status, bitrate, and cache fullness
from `GET /livestreams/0`. No start/stop/platform control here -- the
operator manages that from the encoder's own front panel or app, not
this cockpit, so there's nothing to accidentally trigger mid-ride.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger("motocam.encoder")

REQUEST_TIMEOUT_S = 1.5
RECONNECT_MIN_INTERVAL_S = 5.0
RECONNECT_MAX_INTERVAL_S = 30.0
OFFLINE_LOG_INTERVAL_S = 30.0

# /livestreams/0 status values per the Streaming REST API doc, mapped to
# this app's ok/warn/bad/idle chip vocabulary (see ui/widgets/top_bar.py).
STATUS_CHIP_STATE = {
    "Idle": "idle",
    "Connecting": "warn",
    "Streaming": "ok",
    "Flushing": "warn",
    "Interrupted": "bad",
}


@dataclass(frozen=True)
class EncoderStatus:
    connected: bool = False
    status: str = "unknown"
    bitrate_bps: int | None = None
    cache_pct: int | None = None

    @property
    def on_air(self) -> bool:
        return self.status == "Streaming"


class StreamingEncoderMonitor:
    """Polls a Blackmagic Streaming Encoder's `/livestreams/0` endpoint.
    Same throttled-reconnect/tolerant-GET shape as
    camera/bmd_rest_camera.py's BlackmagicRestCameraBackend -- a missing
    or unreachable encoder degrades to `status=EncoderStatus()`
    (disconnected), never fakes an ON AIR reading."""

    def __init__(self, ip: str, port: int = 80):
        self.ip = ip
        self.port = port
        self._connected = False
        self._last_connect_attempt = 0.0
        self._connect_failures = 0
        self._last_offline_log_at = 0.0
        self.status = EncoderStatus()

    @property
    def connected(self) -> bool:
        return self._connected

    def _url(self, path: str) -> str:
        return f"http://{self.ip}:{self.port}/control/api/v1{path}"

    def _request_sync(self, path: str) -> dict | None:
        request = urllib.request.Request(self._url(path), method="GET")
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            body = response.read()
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None

    async def _request(self, path: str) -> dict | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._request_sync, path)

    async def refresh(self) -> EncoderStatus:
        # Called every refresh tick while the caller wants a reading --
        # throttle so an absent encoder does not keep a slow REST timeout
        # permanently in the event-loop's executor queue (same reasoning
        # as BlackmagicRestCameraBackend.connect()).
        now = time.monotonic()
        if now - self._last_connect_attempt < self._current_reconnect_interval_s():
            return self.status
        self._last_connect_attempt = now
        try:
            payload = await self._request("/livestreams/0")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if self._connected:
                logger.warning("Streaming Encoder at %s:%d lost: %s", self.ip, self.port, exc)
                self._last_offline_log_at = now
            else:
                if now - self._last_offline_log_at >= OFFLINE_LOG_INTERVAL_S:
                    logger.info(
                        "Streaming Encoder not reachable at %s:%d (%s), will retry in %.0fs",
                        self.ip, self.port, exc, self._current_reconnect_interval_s(),
                    )
                    self._last_offline_log_at = now
            self._connect_failures += 1
            self._connected = False
            self.status = EncoderStatus(connected=False)
            return self.status

        if not self._connected:
            logger.info("Streaming Encoder connected at %s:%d", self.ip, self.port)
        self._connect_failures = 0
        self._last_offline_log_at = 0.0
        self._connected = True
        if payload is None:
            self.status = EncoderStatus(connected=True)
            return self.status
        self.status = EncoderStatus(
            connected=True,
            status=str(payload.get("status", "unknown")),
            bitrate_bps=int(payload["bitrate"]) if payload.get("bitrate") is not None else None,
            cache_pct=int(payload["cache"]) if payload.get("cache") is not None else None,
        )
        return self.status

    def _current_reconnect_interval_s(self) -> float:
        if self._connected:
            return RECONNECT_MIN_INTERVAL_S
        return min(RECONNECT_MAX_INTERVAL_S, RECONNECT_MIN_INTERVAL_S * (2 ** min(self._connect_failures, 3)))
