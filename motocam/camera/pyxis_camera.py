"""Blackmagic PYXIS backend over the documented Camera Control REST API.

PYXIS (like the Studio Cameras and Cinema Camera 6K line) exposes
Blackmagic's official REST control API over its Ethernet port: JSON
endpoints under http://<camera-ip>/control/api/v1/... -- documented in
the "Blackmagic Camera Control REST API" developer documentation. This
implementation uses only endpoints from that documentation:

    GET  /system                      liveness / codec info
    GET/PUT /transports/0/record      {"recording": bool}
    GET/PUT /video/iso                {"iso": 800}
    GET/PUT /video/whiteBalance       {"whiteBalance": 5600}
    GET/PUT /video/shutter            {"shutterSpeed": 50} or {"shutterAngle": 18000}
    PUT  /lens/iris                   {"apertureStop": 2.8}
    PUT  /lens/focus/doAutoFocus      (no body)
    GET/PUT /lens/zoom                {"normalised": 0..1}
    GET  /media/active                {"remainingRecordTime": seconds, ...}
    GET  /system/format               {"frameRate": "25", ...}

Individual endpoints that a given firmware/lens combination doesn't
serve (404) are tolerated per-field -- e.g. a prime lens has no servo
zoom -- and reported as None/ignored rather than failing the whole
state refresh. Field names should still be spot-checked against the
firmware's own OpenAPI files the first time real hardware is available
(Settings -> Camera & Lens keeps the IP editable at runtime for exactly
that session).

Networking is stdlib urllib in a thread-pool executor -- two-figure
milliseconds per call on a LAN, no extra dependency, and the qasync
event loop never blocks.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import ssl
import time
import urllib.error
import urllib.request

from motocam.camera.base import CameraBackend, CameraState

logger = logging.getLogger("motocam.camera.pyxis")

REQUEST_TIMEOUT_S = 1.5
RECONNECT_MIN_INTERVAL_S = 5.0
# One rocker tick at full deflection nudges the zoom target by this much
# of the lens range; the control loop re-sends at 20 Hz, so full-speed
# travel end-to-end takes ~1.7 s -- REST has position control only, no
# velocity endpoint, so velocity is synthesized from position nudges.
ZOOM_STEP_PER_TICK = 0.03


def format_shutter(value: dict) -> str | None:
    """REST /video/shutter payload -> the app's display string."""
    if value.get("shutterSpeed"):
        return f"1/{value['shutterSpeed']}"
    if value.get("shutterAngle"):
        return f"{value['shutterAngle'] / 100:.1f}°".replace(".0°", "°")
    return None


def parse_shutter(shutter: str) -> dict | None:
    """App display string ("1/50", "180°", "172.8°") -> REST payload."""
    text = shutter.strip()
    if text.startswith("1/"):
        try:
            return {"shutterSpeed": int(text[2:])}
        except ValueError:
            return None
    if text.endswith("°"):
        try:
            return {"shutterAngle": int(round(float(text[:-1]) * 100))}
        except ValueError:
            return None
    return None


def parse_iris(iris: str) -> dict | None:
    """App display string ("f/2.8") -> REST payload."""
    text = iris.strip().lower().lstrip("f").lstrip("/")
    try:
        return {"apertureStop": float(text)}
    except ValueError:
        return None


class PyxisCameraBackend(CameraBackend):
    source = "pyxis-rest"

    def __init__(
        self,
        ip: str,
        port: int = 80,
        use_tls: bool = False,
        verify_tls: bool = False,
        username: str | None = None,
        password: str | None = None,
        auth_token: str | None = None,
    ):
        """Blackmagic Camera Control REST backend.

        Recent firmware (PYXIS presents a device TLS cert on :443) can serve
        the control API over HTTPS and gate it behind credentials, so the
        transport is configurable:

        - use_tls: talk HTTPS instead of HTTP.
        - verify_tls: validate the cert chain. Off by default because the
          camera presents a self-signed Blackmagic *device* certificate that
          no public CA vouches for -- verifying would always fail. (Point
          this at a pinned CA bundle only if you provision one.)
        - auth_token / username+password: sent as an Authorization header
          (Bearer, else Basic) when the camera requires authentication.
        """
        self.ip = ip
        self.port = port
        self._scheme = "https" if use_tls else "http"
        self._ssl_context = self._build_ssl_context(use_tls, verify_tls)
        self._auth_header = self._build_auth_header(auth_token, username, password)
        self._connected = False
        self._last_connect_attempt = 0.0
        self._zoom_normalised: float | None = None
        self._zoom_unsupported_logged = False

    @staticmethod
    def _build_ssl_context(use_tls: bool, verify_tls: bool) -> ssl.SSLContext | None:
        if not use_tls:
            return None
        if verify_tls:
            return ssl.create_default_context()
        # Self-signed device cert: encrypt but don't verify the chain.
        return ssl._create_unverified_context()

    @staticmethod
    def _build_auth_header(auth_token: str | None, username: str | None, password: str | None) -> str | None:
        if auth_token:
            return f"Bearer {auth_token}"
        if username is not None and password is not None:
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            return f"Basic {token}"
        return None

    # -- plumbing ------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self._scheme}://{self.ip}:{self.port}/control/api/v1{path}"

    def _request_sync(self, method: str, path: str, payload: dict | None = None) -> dict | None:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self._url(path), data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if self._auth_header is not None:
            request.add_header("Authorization", self._auth_header)
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S, context=self._ssl_context) as response:
            body = response.read()
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None

    async def _request(self, method: str, path: str, payload: dict | None = None) -> dict | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._request_sync, method, path, payload)

    async def _try_get(self, path: str) -> dict | None:
        """GET tolerating a missing endpoint/feature (404 and friends) --
        per-field degradation, not whole-refresh failure."""
        try:
            return await self._request("GET", path)
        except urllib.error.HTTPError:
            return None

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        # Called every refresh tick while disconnected (CameraController
        # retries) -- throttle so an absent camera costs one probe per
        # RECONNECT_MIN_INTERVAL_S, not one per 500 ms tick.
        now = time.monotonic()
        if now - self._last_connect_attempt < RECONNECT_MIN_INTERVAL_S:
            return
        self._last_connect_attempt = now
        try:
            await self._request("GET", "/system")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if self._connected:
                logger.warning("PYXIS at %s:%d lost: %s", self.ip, self.port, exc)
            else:
                logger.info("PYXIS not reachable at %s:%d (%s), will retry", self.ip, self.port, exc)
            self._connected = False
            return
        if not self._connected:
            logger.info("PYXIS connected at %s:%d (REST /system OK)", self.ip, self.port)
        self._connected = True
        zoom = await self._try_get("/lens/zoom")
        if zoom is not None and "normalised" in zoom:
            self._zoom_normalised = float(zoom["normalised"])

    async def disconnect(self) -> None:
        self._connected = False

    # -- state ---------------------------------------------------------------
    async def get_state(self) -> CameraState:
        state = CameraState(connected=True)
        try:
            record = await self._request("GET", "/transports/0/record")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            # The transport endpoint is core -- losing it means the camera
            # is gone, not a per-feature gap.
            logger.warning("PYXIS state refresh failed (%s), marking disconnected", exc)
            self._connected = False
            return CameraState(connected=False)
        if record is not None:
            state.recording = bool(record.get("recording", False))

        iso = await self._try_get("/video/iso")
        if iso is not None and iso.get("iso") is not None:
            state.iso = int(iso["iso"])

        wb = await self._try_get("/video/whiteBalance")
        if wb is not None and wb.get("whiteBalance") is not None:
            state.white_balance = int(wb["whiteBalance"])

        shutter = await self._try_get("/video/shutter")
        if shutter is not None:
            state.shutter = format_shutter(shutter)

        media = await self._try_get("/media/active")
        if media is not None and media.get("remainingRecordTime") is not None:
            state.media_remaining_min = float(media["remainingRecordTime"]) / 60.0

        system_format = await self._try_get("/system/format")
        if system_format is not None and system_format.get("frameRate"):
            try:
                state.fps = float(str(system_format["frameRate"]).replace("p", ""))
            except ValueError:
                pass
        return state

    # -- commands ------------------------------------------------------------
    async def start_record(self) -> None:
        await self._request("PUT", "/transports/0/record", {"recording": True})

    async def stop_record(self) -> None:
        await self._request("PUT", "/transports/0/record", {"recording": False})

    async def set_iso(self, iso: int) -> None:
        await self._request("PUT", "/video/iso", {"iso": iso})

    async def set_white_balance(self, kelvin: int) -> None:
        await self._request("PUT", "/video/whiteBalance", {"whiteBalance": kelvin})

    async def set_shutter(self, shutter: str) -> None:
        payload = parse_shutter(shutter)
        if payload is not None:
            await self._request("PUT", "/video/shutter", payload)

    async def set_iris(self, iris: str) -> None:
        payload = parse_iris(iris)
        if payload is not None:
            await self._request("PUT", "/lens/iris", payload)

    async def trigger_autofocus(self) -> None:
        try:
            # This firmware requires a JSON body (an empty object) plus the
            # Content-Type header even on this parameterless action PUT -- a
            # bodyless PUT returns 400. Verified against a live PYXIS 6K with
            # an L-mount AF lens: {} -> 204, no body -> 400.
            await self._request("PUT", "/lens/focus/doAutoFocus", {})
        except urllib.error.HTTPError:
            logger.warning("PYXIS autofocus endpoint unavailable (manual lens?)")

    async def set_zoom_speed(self, speed: float) -> None:
        if speed == 0.0 or self._zoom_normalised is None:
            return
        target = max(0.0, min(1.0, self._zoom_normalised + speed * ZOOM_STEP_PER_TICK))
        if target == self._zoom_normalised:
            return
        try:
            await self._request("PUT", "/lens/zoom", {"normalised": target})
            self._zoom_normalised = target
        except urllib.error.HTTPError:
            if not self._zoom_unsupported_logged:
                logger.warning("PYXIS lens has no servo zoom endpoint; zoom rocker inactive")
                self._zoom_unsupported_logged = True
