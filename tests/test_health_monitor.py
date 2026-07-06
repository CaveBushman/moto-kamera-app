from __future__ import annotations

import time

from motocam.watchdog import health as health_module
from motocam.watchdog.health import HealthMonitor


def test_health_monitor_caches_expensive_system_sample(monkeypatch):
    times = iter([10.0, 10.5, 12.0])
    monkeypatch.setattr(health_module.time, "monotonic", lambda: next(times))
    reads = 0

    def read_temp() -> float:
        nonlocal reads
        reads += 1
        return 42.0

    monkeypatch.setattr(HealthMonitor, "_read_cpu_temp", staticmethod(read_temp))
    monkeypatch.setattr(HealthMonitor, "_safe_psutil", staticmethod(lambda _fn: 18.0))

    monitor = HealthMonitor(sample_interval_s=1.5)
    first = monitor.sample()
    second = monitor.sample()
    third = monitor.sample()

    assert first is second
    assert third is not first
    assert reads == 2


def test_health_monitor_keeps_cached_video_fps_fresh(monkeypatch):
    monkeypatch.setattr(health_module.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(HealthMonitor, "_read_cpu_temp", staticmethod(lambda: None))
    monkeypatch.setattr(HealthMonitor, "_safe_psutil", staticmethod(lambda _fn: None))

    monitor = HealthMonitor(sample_interval_s=10.0)
    sample = monitor.sample()
    monitor.set_video_fps(29.7)

    assert sample.video_fps == 29.7
    assert monitor.sample().video_fps == 29.7


def test_health_latest_returns_immediately_and_samples_in_background(monkeypatch):
    calls = 0

    def slow_sample(_fn):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return 21.0

    monkeypatch.setattr(HealthMonitor, "_read_cpu_temp", staticmethod(lambda: 44.0))
    monkeypatch.setattr(HealthMonitor, "_safe_psutil", staticmethod(slow_sample))

    monitor = HealthMonitor(sample_interval_s=0.0)
    try:
        started = time.monotonic()
        first = monitor.latest()
        elapsed = time.monotonic() - started

        assert elapsed < 0.03
        assert first.cpu_load_pct is None

        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            latest = monitor.latest()
            if latest.cpu_load_pct == 21.0:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("background health sample did not publish")

        assert calls >= 2  # CPU and RAM collection both ran off the caller thread.
    finally:
        monitor.close()
