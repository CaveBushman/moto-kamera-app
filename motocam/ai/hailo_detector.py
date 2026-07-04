"""Real object detector for the Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS).

Runs a compiled YOLO `.hef` model through the HailoRT Python API
(`hailo_platform`), the software stack shipped with the AI HAT+ on
Raspberry Pi OS. The design-doc target (section 8.5) is a
YOLOv8n/YOLOv11n model exported to ONNX and compiled to HEF with the
Hailo Model Zoo; this class consumes that HEF at runtime.

Structure mirrors the camera/gimbal integrations: the hardware-touching
parts (VDevice, InferModel) are import-guarded so the module loads on a
dev laptop with no Hailo runtime, and all the coordinate maths
(letterbox preprocessing, HAILO_NMS output parsing, un-letterboxing back
to source-frame pixels) is pure and unit-tested. `build_detector` falls
back to NullDetector -- tap-to-select CSRT still works -- when the
runtime or HEF is missing, and never fakes detections.

HailoRT API used (verified against hailo-ai/Hailo-Application-Code-Examples
common/hailo_inference.py):
    VDevice.create_params() / VDevice(params)
    vdevice.create_infer_model(hef_path)
    infer_model.set_batch_size(1)
    infer_model.input().set_format_type(FormatType.UINT8)
    infer_model.configure() -> configured_model
    hef.get_input_vstream_infos()[0].shape -> (H, W, C)
    configured_model.create_bindings(output_buffers=...) / run_async / wait
NMS output: list indexed by class_id; each entry an (N, 5) array of
[ymin, xmin, ymax, xmax, score] in 0..1 normalized coordinates.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from motocam.ai.ai_engine import Detection, NullDetector

logger = logging.getLogger("motocam.ai.hailo")

# Default label order for a stock COCO-trained YOLOv8/v11 HEF. A
# custom-trained "cyclist" model overrides this via ai.labels in config.
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
]

try:
    from hailo_platform import HEF, FormatType, VDevice  # type: ignore

    HAILO_AVAILABLE = True
except ImportError:
    HEF = None  # type: ignore[assignment]
    FormatType = None  # type: ignore[assignment]
    VDevice = None  # type: ignore[assignment]
    HAILO_AVAILABLE = False


@dataclass(frozen=True)
class LetterboxTransform:
    """How a source frame was fitted into the square model input --
    everything needed to map detection boxes back to source pixels."""

    scale: float
    pad_x: int
    pad_y: int
    model_size: int


def letterbox(frame: np.ndarray, model_size: int) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize `frame` to fit a `model_size` x `model_size` input while
    preserving aspect ratio, padding the remainder with grey. Returns the
    padded image plus the transform to invert it. Uses cv2 when available,
    falls back to a numpy nearest-neighbour resize (keeps the pure path
    importable without OpenCV for tests)."""
    h, w = frame.shape[:2]
    scale = model_size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    try:
        import cv2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except Exception:  # noqa: BLE001 -- pure fallback for headless/test use
        ys = (np.linspace(0, h - 1, new_h)).astype(int)
        xs = (np.linspace(0, w - 1, new_w)).astype(int)
        resized = frame[ys][:, xs]

    canvas = np.full((model_size, model_size, frame.shape[2]), 114, dtype=frame.dtype)
    pad_x = (model_size - new_w) // 2
    pad_y = (model_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, LetterboxTransform(scale=scale, pad_x=pad_x, pad_y=pad_y, model_size=model_size)


def unletterbox_box(
    ymin: float, xmin: float, ymax: float, xmax: float, transform: LetterboxTransform, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Map an NMS box (normalized 0..1 in model space) back to source
    pixels: (x, y, w, h), clamped to the frame."""
    ms = transform.model_size
    px1 = xmin * ms - transform.pad_x
    py1 = ymin * ms - transform.pad_y
    px2 = xmax * ms - transform.pad_x
    py2 = ymax * ms - transform.pad_y

    x1 = int(round(px1 / transform.scale))
    y1 = int(round(py1 / transform.scale))
    x2 = int(round(px2 / transform.scale))
    y2 = int(round(py2 / transform.scale))

    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w, x2))
    y2 = max(0, min(frame_h, y2))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def parse_nms_output(
    nms_output, labels: list[str], transform: LetterboxTransform, frame_w: int, frame_h: int, min_confidence: float
) -> list[Detection]:
    """Turn a HAILO_NMS output (list indexed by class id, each an (N, 5)
    array of [ymin, xmin, ymax, xmax, score]) into source-frame
    Detections. Robust to empty classes and short/ragged rows."""
    detections: list[Detection] = []
    for class_id, class_dets in enumerate(nms_output):
        if class_dets is None or len(class_dets) == 0:
            continue
        class_name = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
        for det in class_dets:
            if len(det) < 5:
                continue
            ymin, xmin, ymax, xmax, score = float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
            if score < min_confidence:
                continue
            x, y, w, h = unletterbox_box(ymin, xmin, ymax, xmax, transform, frame_w, frame_h)
            if w <= 0 or h <= 0:
                continue
            detections.append(Detection(x=x, y=y, w=w, h=h, confidence=score, class_name=class_name))
    return detections


class HailoDetector:
    """YOLO inference on the Hailo-8 AI HAT+. Raises RuntimeError from the
    constructor if the runtime/HEF isn't usable, so callers can fall back
    to NullDetector cleanly (see build_detector)."""
    source = "hailo"

    def __init__(
        self,
        hef_path: str,
        labels: list[str] | None = None,
        min_confidence: float = 0.05,
        timeout_ms: int = 500,
    ):
        if not HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform runtime not installed (not on a Pi with the AI HAT?)")
        self.labels = labels or COCO_LABELS
        self.min_confidence = min_confidence
        # Bounded wait per inference: a stalled accelerator returns no
        # detections for this frame instead of blocking. Even though
        # inference now runs off the UI thread (AiWorker), a short timeout
        # means the detector recovers quickly rather than sitting on a dead
        # job for seconds. Tracking (CSRT) carries the target across the gap.
        self._timeout_ms = max(50, int(timeout_ms))
        self._frame_times: list[float] = []

        params = VDevice.create_params()
        self._vdevice = VDevice(params)
        self._hef = HEF(hef_path)
        self._infer_model = self._vdevice.create_infer_model(hef_path)
        self._infer_model.set_batch_size(1)
        self._infer_model.input().set_format_type(FormatType.UINT8)
        self._config_ctx = self._infer_model.configure()
        self._configured_model = self._config_ctx.__enter__()

        input_shape = self._hef.get_input_vstream_infos()[0].shape  # (H, W, C)
        self._model_size = int(input_shape[0])
        self._output_name = self._infer_model.output().name
        self._output_shape = self._infer_model.output().shape
        logger.info("Hailo detector ready: %s, input %dx%d", hef_path, self._model_size, self._model_size)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        padded, transform = letterbox(frame, self._model_size)
        output_buffers = {self._output_name: np.empty(self._output_shape, dtype=np.float32)}
        bindings = self._configured_model.create_bindings(output_buffers=output_buffers)
        bindings.input().set_buffer(padded)

        # A timeout here (accelerator busy/stalled) must degrade to "no
        # detection this frame", never raise up into the worker/UI. CSRT
        # keeps the current target locked across the gap.
        try:
            self._configured_model.wait_for_async_ready(timeout_ms=self._timeout_ms)
            job = self._configured_model.run_async([bindings], lambda completion_info: None)
            job.wait(self._timeout_ms)
        except Exception as exc:  # noqa: BLE001 -- realtime path: skip this frame, keep running
            logger.debug("Hailo inference skipped (timeout/error): %s", exc)
            return []

        nms_output = bindings.output().get_buffer()
        self._record_fps()
        h, w = frame.shape[:2]
        return parse_nms_output(nms_output, self.labels, transform, w, h, self.min_confidence)

    def _record_fps(self) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        cutoff = now - 2.0
        self._frame_times = [t for t in self._frame_times if t >= cutoff]

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def close(self) -> None:
        try:
            self._config_ctx.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hailo configured-model exit failed: %s", exc)


def build_detector(cfg: dict):
    """Build the configured detector, or NullDetector on any failure.
    ai.type: "hailo" enables real inference; anything else = NullDetector."""
    ai_cfg = cfg.get("ai", {})
    if str(ai_cfg.get("type", "")).lower() != "hailo":
        return NullDetector()
    hef_path = ai_cfg.get("model") or ai_cfg.get("hef_path")
    if not hef_path:
        logger.warning("ai.type=hailo but no ai.model HEF path set -- using NullDetector")
        return NullDetector()
    labels = ai_cfg.get("labels")
    timeout_ms = int(ai_cfg.get("infer_timeout_ms", 500))
    try:
        return HailoDetector(hef_path, labels=labels, timeout_ms=timeout_ms)
    except Exception as exc:  # noqa: BLE001 -- missing runtime/HEF must degrade, not crash
        logger.warning("Hailo detector unavailable (%s) -- using NullDetector (tap-to-select still works)", exc)
        return NullDetector()
