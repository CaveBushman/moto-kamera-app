"""AI Engine (design doc 8, 10.4).

Interface point for the Hailo/AI HAT+ YOLO inference pipeline described
in the design doc (section 8.5: YOLOv8n/YOLOv11n on Hailo, classes
cyclist/motorcycle/car/person). Real Hailo inference requires the
`hailo-platform` runtime and a compiled `.hef` model, neither of which
is available off the Raspberry Pi, so this ships a `NullDetector` that
produces no detections and a `Detector` protocol so a `HailoDetector`
can be dropped in later without touching tracking/ui code.

Until that lands, target acquisition works purely through tap-to-select
+ OpenCV CSRT (tracking/tracker.py), which is exactly the documented
fallback path in section 8.2 ("pokud v miste kliknuti neni detekce...").
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
