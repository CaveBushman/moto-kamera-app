"""GPS manager tests: the serial read now runs on a background thread so
the blocking readline() never sits on the UI thread's poll(). These cover
that the reader parses NMEA into the published snapshot, that poll() is
non-blocking and consistent, that simulated mode still works, and that a
serial error degrades to simulated instead of throwing."""
from __future__ import annotations

import time

import serial

from motocam.gps import gps_manager as gps_module
from motocam.gps.gps_manager import DetectedGpsDevice, GpsManager

# Valid fixtures pynmea2 parses: a GGA (fix + position) and an RMC (speed).
GGA = b"$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n"
RMC = b"$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A\r\n"


class _FakeSerial:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    @property
    def in_waiting(self) -> int:
        return sum(len(line) for line in self._lines)

    def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""

    def close(self) -> None:
        pass


class _RaisingSerial:
    @property
    def in_waiting(self) -> int:
        return 10

    def readline(self):
        raise serial.SerialException("cable yanked")

    def close(self) -> None:
        pass


def _wait(predicate, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)


def test_reader_thread_parses_nmea_into_published_snapshot():
    gps = GpsManager(device="/dev/fake")
    gps._simulated = False
    gps._serial = _FakeSerial([GGA, RMC])
    gps._start_reader()
    try:
        _wait(lambda: gps.state.fix and gps.state.speed_kmh is not None)
    finally:
        gps.close()

    fix = gps.poll()
    assert fix.fix is True
    assert fix.satellites == 8
    assert fix.lat is not None and fix.lon is not None
    assert fix.speed_kmh == 022.4 * 1.852
    assert gps.source == "real"


def test_poll_is_nonblocking_and_returns_snapshot():
    gps = GpsManager(device="/dev/fake")
    gps._simulated = False
    gps._serial = _FakeSerial([GGA])
    gps._start_reader()
    try:
        _wait(lambda: gps.state.fix)
        start = time.monotonic()
        for _ in range(1000):
            gps.poll()
        elapsed = time.monotonic() - start
    finally:
        gps.close()
    # 1000 polls must be effectively instant -- no serial I/O on this path.
    assert elapsed < 0.1


def test_serial_error_degrades_to_simulated():
    gps = GpsManager(device="/dev/fake")
    gps._simulated = False
    gps._serial = _RaisingSerial()
    gps._start_reader()
    try:
        _wait(lambda: gps._simulated)
    finally:
        gps.close()
    assert gps._simulated is True
    assert gps.source == "simulated"
    # poll() now yields a simulated fix without touching the serial port
    assert gps.poll().fix is True


def test_simulated_mode_needs_no_thread():
    gps = GpsManager(device=None)
    assert gps.source == "simulated"
    first = gps.poll()
    second = gps.poll()
    assert first.fix is True and second.fix is True
    assert gps._reader_thread is None


def test_close_joins_reader_thread():
    gps = GpsManager(device="/dev/fake")
    gps._simulated = False
    gps._serial = _FakeSerial([GGA])
    gps._start_reader()
    gps.close()
    assert gps._reader_thread is None


def test_candidate_baudrates_prefers_configured_value_then_common_values():
    gps = GpsManager(device="auto", baudrate=38400)
    assert gps._candidate_baudrates() == [38400, 9600, 115200]


def test_candidate_baudrates_auto_uses_common_nmea_values():
    gps = GpsManager(device="auto", baudrate="auto")
    assert gps._candidate_baudrates() == [9600, 38400, 115200]


def test_auto_device_detection_returns_device_and_matching_baudrate(monkeypatch):
    gps = GpsManager(device="auto", baudrate=9600)
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(gps, "_candidate_devices", lambda: ["/dev/not-gps", "/dev/gps"])

    def looks_like_nmea(device: str, baudrate: int) -> bool:
        calls.append((device, baudrate))
        return device == "/dev/gps" and baudrate == 115200

    monkeypatch.setattr(gps, "_looks_like_nmea", looks_like_nmea)

    detected = gps._detect_device()

    assert detected == DetectedGpsDevice(device="/dev/gps", baudrate=115200)
    assert calls == [
        ("/dev/not-gps", 9600),
        ("/dev/not-gps", 38400),
        ("/dev/not-gps", 115200),
        ("/dev/gps", 9600),
        ("/dev/gps", 38400),
        ("/dev/gps", 115200),
    ]


def test_open_auto_device_uses_detected_baudrate(monkeypatch):
    opened: dict[str, object] = {}

    class _OpeningSerial:
        def __init__(self, device: str, baudrate: int, timeout: float):
            opened["device"] = device
            opened["baudrate"] = baudrate
            opened["timeout"] = timeout

        def close(self) -> None:
            pass

    gps = GpsManager(device="auto", baudrate=9600)
    monkeypatch.setattr(gps, "_detect_device", lambda: DetectedGpsDevice("/dev/gps", 115200))
    monkeypatch.setattr(gps, "_start_reader", lambda: None)
    monkeypatch.setattr(gps_module.serial, "Serial", _OpeningSerial)

    gps.open()

    assert gps.device == "/dev/gps"
    assert gps.baudrate == 115200
    assert opened == {"device": "/dev/gps", "baudrate": 115200, "timeout": 0.2}
    assert gps.source == "real"


def test_explicit_device_supports_auto_baudrate(monkeypatch):
    opened: dict[str, object] = {}

    class _OpeningSerial:
        def __init__(self, device: str, baudrate: int, timeout: float):
            opened["device"] = device
            opened["baudrate"] = baudrate
            opened["timeout"] = timeout

        def close(self) -> None:
            pass

    gps = GpsManager(device="/dev/gps", baudrate="auto")
    monkeypatch.setattr(gps, "_detect_baudrate", lambda device: 38400)
    monkeypatch.setattr(gps, "_start_reader", lambda: None)
    monkeypatch.setattr(gps_module.serial, "Serial", _OpeningSerial)

    gps.open()

    assert gps.baudrate == 38400
    assert opened == {"device": "/dev/gps", "baudrate": 38400, "timeout": 0.2}
    assert gps.source == "real"
