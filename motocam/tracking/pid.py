"""PID regulator producing pan/tilt velocity commands (design doc section 9)."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class PidGains:
    kp: float = 0.05
    ki: float = 0.0
    kd: float = 0.01


class AxisPid:
    """Single-axis PID with dead zone and max-speed clamp."""

    def __init__(self, gains: PidGains, dead_zone: float, max_speed: float):
        self.gains = gains
        self.dead_zone = dead_zone
        self.max_speed = max_speed
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float) -> float:
        now = time.monotonic()
        dt = (now - self._prev_time) if self._prev_time is not None else 0.0
        self._prev_time = now

        if abs(error) < self.dead_zone:
            self._integral = 0.0
            self._prev_error = error
            return 0.0

        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = (
            self.gains.kp * error
            + self.gains.ki * self._integral
            + self.gains.kd * derivative
        )
        return max(-self.max_speed, min(self.max_speed, output))


class GimbalPid:
    """Pan/tilt PID pair, matching the config.yaml gimbal defaults."""

    def __init__(
        self,
        dead_zone_x: float = 30,
        dead_zone_y: float = 25,
        max_pan_speed: float = 20,
        max_tilt_speed: float = 12,
        pan_gains: PidGains | None = None,
        tilt_gains: PidGains | None = None,
    ):
        self.pan = AxisPid(pan_gains or PidGains(0.05, 0.0, 0.01), dead_zone_x, max_pan_speed)
        self.tilt = AxisPid(tilt_gains or PidGains(0.04, 0.0, 0.01), dead_zone_y, max_tilt_speed)

    def reset(self) -> None:
        self.pan.reset()
        self.tilt.reset()

    def update(self, error_x: float, error_y: float) -> tuple[float, float]:
        return self.pan.update(error_x), self.tilt.update(error_y)
