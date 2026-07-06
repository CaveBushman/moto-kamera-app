from __future__ import annotations

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
