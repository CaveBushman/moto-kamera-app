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
from pathlib import Path

import numpy as np

from motocam.ai.ai_engine import Detection, NullDetector

logger = logging.getLogger("motocam.ai.hailo")
DEV_HEF_MAGIC = b"MOTOCAM_DEV_HEF\n"

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
        self.last_inference_ok: bool | None = None
        self.consecutive_errors = 0

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
            self.last_inference_ok = False
            self.consecutive_errors += 1
            logger.debug("Hailo inference skipped (timeout/error): %s", exc)
            return []

        nms_output = bindings.output().get_buffer()
        self.last_inference_ok = True
        self.consecutive_errors = 0
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


class SimulatedDetector:
    """Deterministic software detector for staged AI rollout.

    This never touches HailoRT or a HEF file. It lets us exercise AI
    ASSIST / FULL AI / ByteTrack on the Ri5 with predictable load before
    we reintroduce the real accelerator path.
    """

    source = "sim_ai"

    def __init__(self, class_name: str = "bicycle"):
        self.class_name = class_name or "bicycle"
        self._frame_times: list[float] = []
        self._start_time = time.monotonic()

    def infer(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        now = time.monotonic()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if t >= now - 2.0]
        phase = (now - self._start_time) * 0.55
        box_w = max(40, int(w * 0.10))
        box_h = max(80, int(h * 0.24))
        cx = int(w * (0.50 + 0.18 * np.sin(phase)))
        cy = int(h * (0.55 + 0.05 * np.sin(phase * 0.7)))
        x = max(0, min(w - box_w, cx - box_w // 2))
        y = max(0, min(h - box_h, cy - box_h // 2))
        return [Detection(x=x, y=y, w=box_w, h=box_h, confidence=0.92, class_name=self.class_name)]

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0


class DotDetector:
    """Very cheap software detector for bench-testing AI assist on a marker dot.

    It intentionally does not pretend to be cyclist AI. It finds small,
    saturated bright blobs such as the orange test dot in the synthetic
    preview, so we can tune ByteTrack/PID/gimbal response before enabling
    Hailo again.
    """

    source = "dot_ai"

    def __init__(
        self,
        class_name: str = "bicycle",
        min_area: int = 12,
        max_area_ratio: float = 0.08,
        pad: int = 22,
    ):
        self.class_name = class_name or "bicycle"
        self.min_area = max(1, int(min_area))
        self.max_area_ratio = max(0.001, float(max_area_ratio))
        self.pad = max(0, int(pad))
        self._frame_times: list[float] = []

    def infer(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        now = time.monotonic()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if t >= now - 2.0]
        if h <= 0 or w <= 0:
            return []

        try:
            import cv2
        except Exception as exc:  # noqa: BLE001 -- AI must degrade, not break startup
            logger.warning("Dot AI unavailable without OpenCV: %s", exc)
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Saturated and bright catches orange/red/green/blue test markers,
        # but ignores the dark UI background and most video noise.
        mask = cv2.inRange(hsv, np.array([0, 70, 110], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        max_area = float(w * h) * self.max_area_ratio
        best: tuple[float, tuple[int, int, int, int]] | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / float(bh)
            if aspect < 0.35 or aspect > 2.8:
                continue
            score = area / float(max(1, bw * bh))
            if best is None or score > best[0]:
                best = (score, (x, y, bw, bh))

        if best is None:
            return []
        x, y, bw, bh = best[1]
        pad = self.pad
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + pad * 2)
        bh = min(h - y, bh + pad * 2)
        return [Detection(x=x, y=y, w=bw, h=bh, confidence=0.96, class_name=self.class_name)]

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0


class HailoCanaryDetector:
    """Staged Hailo rollout with a software fallback.

    In canary mode Hailo is allowed into the runtime, but it is not allowed
    to be a single point of failure. Missing runtime/model, repeated
    timeouts, or too-slow inference all degrade to DotDetector so the app
    stays controllable while we collect logs and tune the HEF path.
    """

    source = "hailo_canary"

    def __init__(
        self,
        primary: HailoDetector | None,
        fallback: DotDetector,
        max_inference_ms: float = 180.0,
        max_consecutive_errors: int = 3,
    ):
        self.primary = primary
        self.fallback = fallback
        self.max_inference_ms = max(10.0, float(max_inference_ms or 180.0))
        self.max_consecutive_errors = max(1, int(max_consecutive_errors or 3))
        self._disabled_reason: str | None = "unavailable" if primary is None else None
        self._last_primary_ms: float | None = None
        self._last_source = "dot_ai" if primary is None else "hailo"
        self._slow_count = 0

    @property
    def source(self) -> str:  # type: ignore[override]
        if self._disabled_reason:
            return "hailo_canary_dot"
        return "hailo_canary"

    def infer(self, frame: np.ndarray) -> list[Detection]:
        if self.primary is None or self._disabled_reason is not None:
            self._last_source = "dot_ai"
            return self.fallback.infer(frame)

        start = time.monotonic()
        detections = self.primary.infer(frame)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        self._last_primary_ms = elapsed_ms

        if elapsed_ms > self.max_inference_ms:
            self._slow_count += 1
            logger.warning(
                "Hailo canary slow inference %.0f ms > %.0f ms (slow_count=%d)",
                elapsed_ms,
                self.max_inference_ms,
                self._slow_count,
            )
        else:
            self._slow_count = 0

        last_ok = getattr(self.primary, "last_inference_ok", True)
        consecutive_errors = int(getattr(self.primary, "consecutive_errors", 0) or 0)
        if last_ok is False and consecutive_errors >= self.max_consecutive_errors:
            self._disable(f"{consecutive_errors} consecutive Hailo errors")
            self._last_source = "dot_ai"
            return self.fallback.infer(frame)
        if self._slow_count >= self.max_consecutive_errors:
            self._disable(f"{self._slow_count} consecutive slow Hailo inferences")
            self._last_source = "dot_ai"
            return self.fallback.infer(frame)

        self._last_source = "hailo"
        return detections

    def _disable(self, reason: str) -> None:
        if self._disabled_reason is not None:
            return
        self._disabled_reason = reason
        logger.warning("Hailo canary disabled, using Dot AI fallback: %s", reason)
        try:
            close = getattr(self.primary, "close", None)
            close() if close is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hailo close after canary disable failed: %s", exc)

    @property
    def fps(self) -> float:
        if self._last_source == "hailo" and self.primary is not None:
            return self.primary.fps
        return self.fallback.fps


class DevHefDetector(SimulatedDetector):
    """Synthetic detector enabled only by a MotoCam dev HEF marker file.

    This is not a Hailo executable and is deliberately labelled `dev_hef`
    in telemetry. It remains for compatibility with the older dev HEF
    workflow; new staged testing should use ai.type: simulated.
    """

    source = "dev_hef"


def is_dev_hef(path: str | Path) -> bool:
    try:
        with Path(path).open("rb") as fh:
            return fh.read(len(DEV_HEF_MAGIC)) == DEV_HEF_MAGIC
    except OSError:
        return False


def resolve_hef_path(hef_path: str, config_dir: str | Path | None = None) -> str:
    path = Path(hef_path).expanduser()
    if path.is_absolute():
        return str(path)
    if config_dir is not None:
        candidate = Path(config_dir).expanduser() / path
        if candidate.exists():
            return str(candidate)
    return str(path)


def build_detector(cfg: dict):
    """Build the configured detector, or NullDetector on any failure.
    ai.type: "simulated" exercises AI without Hailo; "hailo" enables real inference."""
    ai_cfg = cfg.get("ai", {})
    ai_type = str(ai_cfg.get("type", "")).lower()
    if ai_type in {"simulated", "simulation", "sim"}:
        class_name = str(ai_cfg.get("target_class") or "bicycle")
        logger.warning("Simulated AI detector active -- no Hailo runtime or HEF will be used")
        return SimulatedDetector(class_name=class_name)
    if ai_type in {"dot", "marker", "test_dot"}:
        class_name = str(ai_cfg.get("target_class") or "bicycle")
        logger.warning("Dot AI detector active -- bench-test marker tracking, no Hailo runtime or HEF will be used")
        return DotDetector(
            class_name=class_name,
            min_area=int(ai_cfg.get("dot_min_area", 12) or 12),
            max_area_ratio=float(ai_cfg.get("dot_max_area_ratio", 0.08) or 0.08),
            pad=int(ai_cfg.get("dot_box_pad", 22) or 22),
        )
    if ai_type == "hailo_canary":
        class_name = str(ai_cfg.get("target_class") or "bicycle")
        fallback = DotDetector(
            class_name=class_name,
            min_area=int(ai_cfg.get("dot_min_area", 12) or 12),
            max_area_ratio=float(ai_cfg.get("dot_max_area_ratio", 0.08) or 0.08),
            pad=int(ai_cfg.get("dot_box_pad", 22) or 22),
        )
        primary = _build_hailo_detector(ai_cfg, cfg)
        if primary is None:
            logger.warning("Hailo canary started with Dot AI fallback only")
        else:
            logger.warning("Hailo canary active -- real Hailo inference with Dot AI fallback")
        return HailoCanaryDetector(
            primary=primary,
            fallback=fallback,
            max_inference_ms=float(ai_cfg.get("canary_max_inference_ms", ai_cfg.get("guard_inference_ms", 180)) or 180),
            max_consecutive_errors=int(ai_cfg.get("canary_max_errors", 3) or 3),
        )
    if ai_type != "hailo":
        return NullDetector("null_disabled")
    return _build_hailo_detector(ai_cfg, cfg, null_on_failure=True)


def _build_hailo_detector(ai_cfg: dict, cfg: dict, null_on_failure: bool = False):
    hef_path = ai_cfg.get("model") or ai_cfg.get("hef_path")
    if not hef_path:
        logger.warning("ai.type=hailo but no ai.model HEF path set -- using NullDetector "
                       "(run scripts/hailo_check.py; see docs/HAILO_SETUP.md)")
        return NullDetector("null_model") if null_on_failure else None
    hef_path = resolve_hef_path(str(hef_path), cfg.get("_config_dir"))
    if not Path(hef_path).exists():
        logger.warning(
            "Hailo HEF model not found at %s -- using NullDetector "
            "(run scripts/setup_hailo.sh or set ai.model to an absolute .hef path)",
            hef_path,
        )
        return NullDetector("null_model") if null_on_failure else None
    if is_dev_hef(hef_path):
        class_name = str(ai_cfg.get("target_class") or "bicycle")
        logger.warning("MotoCam dev HEF active at %s -- synthetic detections only, not a real Hailo model", hef_path)
        return DevHefDetector(class_name=class_name)
    if not HAILO_AVAILABLE:
        logger.warning("hailo_platform runtime not installed -- using NullDetector "
                       "(install HailoRT on the Ri5 / AI HAT+)")
        return NullDetector("null_runtime") if null_on_failure else None
    labels = ai_cfg.get("labels")
    timeout_ms = int(ai_cfg.get("infer_timeout_ms", 500))
    try:
        return HailoDetector(hef_path, labels=labels, timeout_ms=timeout_ms)
    except Exception as exc:  # noqa: BLE001 -- missing runtime/HEF must degrade, not crash
        logger.warning("Hailo detector unavailable (%s) -- using NullDetector, tap-to-select "
                       "still works (run scripts/hailo_check.py; see docs/HAILO_SETUP.md)", exc)
        return NullDetector("null_error") if null_on_failure else None
