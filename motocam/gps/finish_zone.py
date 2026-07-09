"""Finish-zone proximity rule (race regs: the camera moto must peel off
the rider/group it's filming inside a fixed distance of the finish line --
500 m mandatory per this federation's rules, with an earlier heads-up so
the pilot has time to react). Pure distance math over GPS fixes, no I/O
and no Qt -- easy to unit test, same convention as gimbal/rsdk_protocol.py.

Distance is plain Haversine: at race-moto speeds and finish-zone ranges
(hundreds of metres) the flat-Earth error is a few centimetres, well under
GPS fix noise, so there's no need for anything more elaborate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

EARTH_RADIUS_M = 6_371_000.0
DEFAULT_WARNING_M = 1000.0
DEFAULT_MANDATORY_M = 500.0


class FinishZoneState(str, Enum):
    NONE = "none"  # no finish line configured, or no GPS fix yet
    FAR = "far"  # outside the warning radius
    WARNING = "warning"  # inside the warning radius, outside the mandatory one
    MANDATORY = "mandatory"  # inside the mandatory peel-off radius


# Ordering used to decide whether a state is an escalation (e.g. for the
# one-shot audio cue) -- higher is more urgent.
_SEVERITY = {
    FinishZoneState.NONE: 0,
    FinishZoneState.FAR: 0,
    FinishZoneState.WARNING: 1,
    FinishZoneState.MANDATORY: 2,
}


def is_escalation(previous: FinishZoneState, current: FinishZoneState) -> bool:
    return _SEVERITY[current] > _SEVERITY[previous]


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class FinishZoneMonitor:
    warning_m: float = DEFAULT_WARNING_M
    mandatory_m: float = DEFAULT_MANDATORY_M
    finish_lat: float | None = None
    finish_lon: float | None = None

    def set_finish(self, lat: float | None, lon: float | None) -> None:
        self.finish_lat = lat
        self.finish_lon = lon

    def clear_finish(self) -> None:
        self.finish_lat = None
        self.finish_lon = None

    @property
    def has_finish(self) -> bool:
        return self.finish_lat is not None and self.finish_lon is not None

    def evaluate(self, lat: float | None, lon: float | None) -> tuple[FinishZoneState, float | None]:
        """Return (state, distance_m); distance_m is None when either the
        finish line or the current fix is unknown."""
        if not self.has_finish or lat is None or lon is None:
            return FinishZoneState.NONE, None
        distance_m = haversine_distance_m(lat, lon, self.finish_lat, self.finish_lon)
        if distance_m <= self.mandatory_m:
            return FinishZoneState.MANDATORY, distance_m
        if distance_m <= self.warning_m:
            return FinishZoneState.WARNING, distance_m
        return FinishZoneState.FAR, distance_m
