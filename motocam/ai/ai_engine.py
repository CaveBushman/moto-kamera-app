"""AI Engine (design doc 8, 10.4).

Defines the `Detector` protocol and the plain-data `Detection` shape
every detector implementation (ai/hailo_detector.py: HailoDetector,
DotDetector, HailoCanaryDetector, NullDetector) produces, so
tracking/ui code never needs to know which one is active -- only
AiEngine.source (delegated to the detector's own `source`) reveals that,
for telemetry/UI display. `build_detector()` in hailo_detector.py picks
the concrete detector from config, falling back to `NullDetector` (no
detections at all) on any missing runtime/model/HAT+, so tap-to-select +
CSRT (tracking/tracker.py) still works with zero AI hardware present.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger("motocam.ai")


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    class_name: str


class Detector(Protocol):
    def infer(self, frame: np.ndarray) -> list[Detection]: ...

    @property
    def fps(self) -> float: ...


class NullDetector:
    """Placeholder used until a HailoDetector is implemented for the AI HAT+."""

    def __init__(self, source: str = "null"):
        self.source = source
        self._warned = False

    def infer(self, frame: np.ndarray) -> list[Detection]:
        if not self._warned:
            logger.warning("NullDetector active - no Hailo runtime, running tap-to-select tracking only")
            self._warned = True
        return []

    @property
    def fps(self) -> float:
        return 0.0


class AiEngine:
    """Thin wrapper the UI/tracker call into: gates inference behind
    `enabled` (AI ASSIST/FULL AI modes only) and filters the active
    detector's raw output down to `confidence`-passing boxes."""

    def __init__(self, detector: Detector | None = None, target_class: str = "cyclist", confidence: float = 0.35):
        self.detector = detector or NullDetector()
        self.target_class = target_class
        self.confidence = confidence
        self.enabled = False

    @property
    def source(self) -> str:
        return str(getattr(self.detector, "source", self.detector.__class__.__name__))

    def process(self, frame: np.ndarray) -> list[Detection]:
        if not self.enabled:
            return []
        detections = self.detector.infer(frame)
        return [d for d in detections if d.confidence >= self.confidence]
