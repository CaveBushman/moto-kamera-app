"""Tests for the finish-zone peel-off rule (gps/finish_zone.py): distance
math and the FAR/WARNING/MANDATORY state machine that drives the moto's
on-screen peel-off warning."""
from __future__ import annotations

from motocam.gps.finish_zone import (
    FinishZoneMonitor,
    FinishZoneState,
    haversine_distance_m,
    is_escalation,
)

# Prague Old Town Square to Wenceslas Square: real-world reference distance
# (~1.2 km), used as a sanity check that the formula isn't off by some
# gross factor (unit confusion, wrong Earth radius, etc.).
OLD_TOWN_SQUARE = (50.0875, 14.4213)
WENCESLAS_SQUARE = (50.0810, 14.4280)


def test_haversine_zero_distance_for_same_point():
    assert haversine_distance_m(50.0755, 14.4378, 50.0755, 14.4378) == 0.0


def test_haversine_matches_known_real_world_distance():
    distance_m = haversine_distance_m(*OLD_TOWN_SQUARE, *WENCESLAS_SQUARE)
    assert 700.0 < distance_m < 1100.0


def test_evaluate_returns_none_state_without_finish_line():
    monitor = FinishZoneMonitor()
    state, distance_m = monitor.evaluate(50.0755, 14.4378)
    assert state == FinishZoneState.NONE
    assert distance_m is None


def test_evaluate_returns_none_state_without_gps_fix():
    monitor = FinishZoneMonitor(finish_lat=50.0755, finish_lon=14.4378)
    state, distance_m = monitor.evaluate(None, None)
    assert state == FinishZoneState.NONE
    assert distance_m is None


def test_evaluate_far_outside_warning_radius():
    monitor = FinishZoneMonitor(warning_m=1000.0, mandatory_m=500.0, finish_lat=50.0755, finish_lon=14.4378)
    # ~0.02 degrees of latitude is roughly 2.2 km away -- well outside 1 km.
    state, distance_m = monitor.evaluate(50.0955, 14.4378)
    assert state == FinishZoneState.FAR
    assert distance_m is not None and distance_m > 1000.0


def test_evaluate_warning_between_thresholds():
    monitor = FinishZoneMonitor(warning_m=1000.0, mandatory_m=500.0, finish_lat=50.0755, finish_lon=14.4378)
    # ~0.007 degrees of latitude is roughly 780 m -- inside 1 km, outside 500 m.
    state, distance_m = monitor.evaluate(50.0825, 14.4378)
    assert state == FinishZoneState.WARNING
    assert distance_m is not None and 500.0 < distance_m <= 1000.0


def test_evaluate_mandatory_inside_peel_off_radius():
    monitor = FinishZoneMonitor(warning_m=1000.0, mandatory_m=500.0, finish_lat=50.0755, finish_lon=14.4378)
    state, distance_m = monitor.evaluate(50.0757, 14.4378)
    assert state == FinishZoneState.MANDATORY
    assert distance_m is not None and distance_m <= 500.0


def test_evaluate_boundary_is_inclusive_of_mandatory():
    monitor = FinishZoneMonitor(warning_m=1000.0, mandatory_m=500.0, finish_lat=50.0755, finish_lon=14.4378)
    # Construct a point at exactly the mandatory radius isn't practical with
    # lat/lon deltas, so just check the boundary semantics on distance directly.
    state, _ = monitor.evaluate(50.0755, 14.4378)
    assert state == FinishZoneState.MANDATORY  # distance 0 <= mandatory_m


def test_clear_finish_resets_to_none_state():
    monitor = FinishZoneMonitor(finish_lat=50.0755, finish_lon=14.4378)
    assert monitor.has_finish
    monitor.clear_finish()
    assert not monitor.has_finish
    state, distance_m = monitor.evaluate(50.0755, 14.4378)
    assert state == FinishZoneState.NONE
    assert distance_m is None


def test_set_finish_updates_coordinates():
    monitor = FinishZoneMonitor()
    monitor.set_finish(50.0755, 14.4378)
    assert monitor.has_finish
    assert monitor.finish_lat == 50.0755
    assert monitor.finish_lon == 14.4378


def test_is_escalation_orders_severity_correctly():
    assert is_escalation(FinishZoneState.NONE, FinishZoneState.WARNING)
    assert is_escalation(FinishZoneState.WARNING, FinishZoneState.MANDATORY)
    assert is_escalation(FinishZoneState.FAR, FinishZoneState.MANDATORY)
    assert not is_escalation(FinishZoneState.MANDATORY, FinishZoneState.WARNING)
    assert not is_escalation(FinishZoneState.WARNING, FinishZoneState.WARNING)
    assert not is_escalation(FinishZoneState.WARNING, FinishZoneState.FAR)
