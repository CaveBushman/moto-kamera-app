"""Tests for the PYXIS REST backend: payload translation (the
display-string <-> REST payload mapping that silently drifts), transport
configuration (HTTP/HTTPS + auth, added after a live PYXIS 6K was found
serving a device TLS cert), and the connect/get_state flow against a
local mock HTTP server."""
from __future__ import annotations

import asyncio
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from motocam.camera.pyxis_camera import PyxisCameraBackend, format_shutter, parse_iris, parse_shutter


def test_parse_shutter_speed_fraction():
    assert parse_shutter("1/50") == {"shutterSpeed": 50}
    assert parse_shutter("1/2000") == {"shutterSpeed": 2000}


def test_parse_shutter_angle_degrees():
    assert parse_shutter("180°") == {"shutterAngle": 18000}
    assert parse_shutter("172.8°") == {"shutterAngle": 17280}


def test_parse_shutter_garbage_returns_none():
    assert parse_shutter("fast") is None
    assert parse_shutter("1/x") is None
    assert parse_shutter("°") is None


def test_format_shutter_round_trips_both_forms():
    assert format_shutter({"shutterSpeed": 50}) == "1/50"
    assert format_shutter({"shutterAngle": 18000}) == "180°"
    assert format_shutter({"shutterAngle": 17280}) == "172.8°"
    assert format_shutter({}) is None


def test_parse_iris_accepts_common_spellings():
    assert parse_iris("f/2.8") == {"apertureStop": 2.8}
    assert parse_iris("F/4.0") == {"apertureStop": 4.0}
    assert parse_iris("5.6") == {"apertureStop": 5.6}
    assert parse_iris("wide open") is None


def test_rest_url_layout():
    backend = PyxisCameraBackend(ip="192.168.9.20", port=80)
    assert backend._url("/video/iso") == "http://192.168.9.20:80/control/api/v1/video/iso"


def test_backend_starts_disconnected():
    backend = PyxisCameraBackend(ip="192.168.9.20")
    assert backend.connected is False


# -- transport configuration (HTTPS + auth) -------------------------------
def test_tls_switches_url_scheme():
    backend = PyxisCameraBackend(ip="10.0.0.5", port=443, use_tls=True)
    assert backend._url("/system") == "https://10.0.0.5:443/control/api/v1/system"


def test_auth_header_bearer_preferred_over_basic():
    backend = PyxisCameraBackend(ip="10.0.0.5", auth_token="tok", username="u", password="p")
    assert backend._auth_header == "Bearer tok"


def test_auth_header_basic_from_credentials():
    backend = PyxisCameraBackend(ip="10.0.0.5", username="admin", password="secret")
    import base64

    expected = "Basic " + base64.b64encode(b"admin:secret").decode()
    assert backend._auth_header == expected


def test_no_auth_header_by_default():
    backend = PyxisCameraBackend(ip="10.0.0.5")
    assert backend._auth_header is None


def test_ssl_context_only_built_for_tls():
    assert PyxisCameraBackend(ip="x")._ssl_context is None
    ctx = PyxisCameraBackend(ip="x", use_tls=True)._ssl_context
    assert isinstance(ctx, ssl.SSLContext)
    # default (verify off) must not verify the self-signed device cert
    assert ctx.verify_mode == ssl.CERT_NONE
    verifying = PyxisCameraBackend(ip="x", use_tls=True, verify_tls=True)._ssl_context
    assert verifying.verify_mode == ssl.CERT_REQUIRED


# -- connect / get_state against a mock REST server -----------------------
class _MockPyxisHandler(BaseHTTPRequestHandler):
    responses = {
        "/control/api/v1/system": {"codec": "BRAW"},
        "/control/api/v1/lens/zoom": {"normalised": 0.4},
        "/control/api/v1/transports/0/record": {"recording": True},
        "/control/api/v1/video/iso": {"iso": 800},
        "/control/api/v1/video/whiteBalance": {"whiteBalance": 5600},
        "/control/api/v1/video/shutter": {"shutterSpeed": 50},
        "/control/api/v1/media/active": {"remainingRecordTime": 1800},
        "/control/api/v1/system/format": {"frameRate": "25"},
    }
    seen_auth: list = []

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        type(self).seen_auth.append(self.headers.get("Authorization"))
        body = self.responses.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence test server
        pass


@pytest.fixture
def mock_pyxis():
    _MockPyxisHandler.seen_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockPyxisHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address  # (host, port)
    finally:
        server.shutdown()
        thread.join(timeout=2.0)


def test_connect_and_get_state_against_mock(mock_pyxis):
    host, port = mock_pyxis
    backend = PyxisCameraBackend(ip=host, port=port)

    async def run():
        await backend.connect()
        assert backend.connected is True
        assert backend._zoom_normalised == pytest.approx(0.4)
        state = await backend.get_state()
        return state

    state = asyncio.run(run())
    assert state.connected is True
    assert state.recording is True
    assert state.iso == 800
    assert state.white_balance == 5600
    assert state.shutter == "1/50"
    assert state.media_remaining_min == pytest.approx(30.0)
    assert state.fps == pytest.approx(25.0)


def test_credentials_are_sent_as_authorization_header(mock_pyxis):
    host, port = mock_pyxis
    backend = PyxisCameraBackend(ip=host, port=port, username="admin", password="secret")

    asyncio.run(backend.connect())

    import base64

    expected = "Basic " + base64.b64encode(b"admin:secret").decode()
    assert expected in _MockPyxisHandler.seen_auth
