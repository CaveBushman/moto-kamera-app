"""GPS Manager (design doc 13, 10.8).

Reads NMEA 0183 (GGA/RMC/VTG) from a serial GNSS module. Falls back to a
simulated fix (slow circular path) when no serial device is configured
or reachable, so the rest of the app -- telemetry, UI, control room --
can be developed and demoed without GNSS hardware attached.
"""
from __future__ import annotations

import logging
import math
import time

import pynmea2
import serial

from motocam.core.protocol import GpsTelemetry

logger = logging.getLogger("motocam.gps")


class GpsManager:
    def __init__(self, device: str | None = None, baudrate: int = 9600):
        self.device = device
        self.baudrate = baudrate
        self._serial: serial.Serial | None = None
        self._simulated = device is None
        self._sim_t = 0.0
        self._sim_origin = (50.0755, 14.4378)  # Prague, arbitrary default
        self.state = GpsTelemetry()

    def open(self) -> None:
        if self.device is None:
            self._simulated = True
            logger.warning("No GPS device configured, using simulated fix")
            return
        try:
            self._serial = serial.Serial(self.device, self.baudrate, timeout=0.2)
            self._simulated = False
            logger.info("GPS serial opened on %s", self.device)
        except (serial.SerialException, FileNotFoundError) as exc:
            logger.warning("GPS device %s unavailable (%s), using simulated fix", self.device, exc)
            self._serial = None
            self._simulated = True

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def poll(self) -> GpsTelemetry:
        if self._simulated:
            self.state = self._simulate()
            return self.state

        assert self._serial is not None
        try:
            while self._serial.in_waiting:
                line = self._serial.readline().decode("ascii", errors="ignore").strip()
                self._parse_line(line)
        except serial.SerialException as exc:
            logger.warning("GPS read error (%s), switching to simulated fix", exc)
            self._simulated = True
        return self.state

    def _parse_line(self, line: str) -> None:
        if not line.startswith("$"):
            return
        try:
            msg = pynmea2.parse(line)
        except pynmea2.ParseError:
            return

        if isinstance(msg, pynmea2.types.talker.GGA):
            self.state.fix = msg.gps_qual is not None and msg.gps_qual > 0
            self.state.satellites = int(msg.num_sats) if msg.num_sats else 0
            if msg.latitude and msg.longitude:
                self.state.lat = msg.latitude
                self.state.lon = msg.longitude
        elif isinstance(msg, pynmea2.types.talker.RMC):
            if msg.spd_over_grnd is not None:
                self.state.speed_kmh = float(msg.spd_over_grnd) * 1.852
            if msg.true_course is not None:
                self.state.heading_deg = float(msg.true_course)
            if msg.datestamp and msg.timestamp:
                self.state.utc = f"{msg.datestamp}T{msg.timestamp}"
        elif isinstance(msg, pynmea2.types.talker.VTG):
            if msg.spd_over_grnd_kmph is not None:
                self.state.speed_kmh = float(msg.spd_over_grnd_kmph)

    def _simulate(self) -> GpsTelemetry:
        self._sim_t += 0.5
        radius = 0.01
        lat = self._sim_origin[0] + radius * math.sin(self._sim_t / 20)
        lon = self._sim_origin[1] + radius * math.cos(self._sim_t / 20)
        heading = (math.degrees(self._sim_t / 20) + 90) % 360
        return GpsTelemetry(
            lat=lat, lon=lon, speed_kmh=45.0 + 10 * math.sin(self._sim_t / 5),
            heading_deg=heading, fix=True, satellites=9,
            utc=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        )
