"""Pure-logic tests for the PYXIS REST backend's payload translation --
the network layer needs real hardware, but the display-string <-> REST
payload mapping is exactly the kind of thing that silently drifts."""
from __future__ import annotations

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
