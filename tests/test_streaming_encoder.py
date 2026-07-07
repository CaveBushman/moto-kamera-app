"""Tests for the Blackmagic Streaming Encoder status monitor: URL layout,
reconnect backoff, and the connect/refresh flow against a local mock
HTTP server -- same shape as test_bmd_rest_camera.py's coverage of the
sibling Blackmagic REST backend."""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from motocam.encoder.streaming_encoder import EncoderStatus, StreamingEncoderMonitor


def test_rest_url_layout():
    monitor = StreamingEncoderMonitor(ip="192.168.9.40", port=80)
    assert monitor._url("/livestreams/0") == "http://192.168.9.40:80/control/api/v1/livestreams/0"


def test_monitor_starts_disconnected():
    monitor = StreamingEncoderMonitor(ip="192.168.9.40")
    assert monitor.connected is False
    assert monitor.status == EncoderStatus()


def test_disconnected_encoder_reconnect_interval_backs_off():
    monitor = StreamingEncoderMonitor(ip="192.168.9.40")

    assert monitor._current_reconnect_interval_s() == 5.0
    monitor._connect_failures = 1
    assert monitor._current_reconnect_interval_s() == 10.0
    monitor._connect_failures = 3
    assert monitor._current_reconnect_interval_s() == 30.0
    monitor._connect_failures = 8
    assert monitor._current_reconnect_interval_s() == 30.0


def test_on_air_only_true_while_streaming():
    assert EncoderStatus(connected=True, status="Streaming").on_air is True
    assert EncoderStatus(connected=True, status="Connecting").on_air is False
    assert EncoderStatus(connected=True, status="Idle").on_air is False
    assert EncoderStatus(connected=False).on_air is False


# -- refresh() against a mock REST server ----------------------------------
class _MockEncoderRestHandler(BaseHTTPRequestHandler):
    response_body = {"status": "Streaming", "bitrate": 6600000, "effectiveVideoFormat": "1920x1080p29.97", "cache": 3}

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path != "/control/api/v1/livestreams/0":
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(type(self).response_body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence test server
        pass


@pytest.fixture
def mock_encoder():
    _MockEncoderRestHandler.response_body = {
        "status": "Streaming", "bitrate": 6600000, "effectiveVideoFormat": "1920x1080p29.97", "cache": 3,
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockEncoderRestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address  # (host, port)
    finally:
        server.shutdown()
        thread.join(timeout=2.0)


def test_refresh_against_mock_reports_on_air_status(mock_encoder):
    host, port = mock_encoder
    monitor = StreamingEncoderMonitor(ip=host, port=port)

    status = asyncio.run(monitor.refresh())

    assert monitor.connected is True
    assert status.connected is True
    assert status.status == "Streaming"
    assert status.on_air is True
    assert status.bitrate_bps == 6600000
    assert status.cache_pct == 3


def test_refresh_reflects_idle_status(mock_encoder):
    _MockEncoderRestHandler.response_body = {"status": "Idle", "bitrate": 0, "effectiveVideoFormat": "Idle", "cache": 0}
    host, port = mock_encoder
    monitor = StreamingEncoderMonitor(ip=host, port=port)

    status = asyncio.run(monitor.refresh())

    assert status.on_air is False
    assert status.status == "Idle"


def test_refresh_unreachable_host_reports_disconnected():
    # Port 1 is a reserved/well-known port that's never listening on a
    # dev machine -- connection refused, no live server needed.
    monitor = StreamingEncoderMonitor(ip="127.0.0.1", port=1)

    status = asyncio.run(monitor.refresh())

    assert status.connected is False
    assert monitor.connected is False
