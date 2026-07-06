"""Pure-logic tests for the Hailo detector: letterbox preprocessing,
un-letterboxing detection boxes back to source pixels, and HAILO_NMS
output parsing. The Hailo runtime itself needs the AI HAT+; these cover
everything deterministic, the same split used for the camera/gimbal
integrations."""
from __future__ import annotations

import numpy as np

from motocam.ai.ai_engine import Detection, NullDetector
from motocam.ai.hailo_detector import (
    DEV_HEF_MAGIC,
    DevHefDetector,
    SimulatedDetector,
    build_detector,
    is_dev_hef,
    letterbox,
    parse_nms_output,
    resolve_hef_path,
    unletterbox_box,
)


def test_letterbox_preserves_aspect_ratio_and_pads_to_square():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, tf = letterbox(frame, 640)
    assert padded.shape == (640, 640, 3)
    # 640-wide source scales 1:1; 480 tall centre-padded to 640
    assert tf.scale == 1.0
    assert tf.pad_x == 0
    assert tf.pad_y == 80


def test_letterbox_portrait_frame_pads_horizontally():
    frame = np.zeros((640, 320, 3), dtype=np.uint8)
    padded, tf = letterbox(frame, 640)
    assert padded.shape == (640, 640, 3)
    assert tf.scale == 1.0
    assert tf.pad_y == 0
    assert tf.pad_x == 160


def test_unletterbox_round_trips_a_centered_box():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, tf = letterbox(frame, 640)
    # A normalized box covering the middle of the model input maps back
    # into the source frame, offset by the vertical padding.
    x, y, w, h = unletterbox_box(0.25, 0.25, 0.75, 0.75, tf, 640, 480)
    assert 150 < x < 170          # 0.25*640 = 160
    assert 0 <= y                 # padding removed, clamped into frame
    assert 300 < w < 340          # half of 640
    assert h > 0


def test_unletterbox_clamps_out_of_frame_boxes():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, tf = letterbox(frame, 640)
    x, y, w, h = unletterbox_box(-0.5, -0.5, 1.5, 1.5, tf, 640, 480)
    assert x == 0 and y == 0
    assert x + w <= 640
    assert y + h <= 480


def test_parse_nms_output_maps_classes_and_filters_confidence():
    frame_w, frame_h = 640, 480
    _, tf = letterbox(np.zeros((480, 640, 3), dtype=np.uint8), 640)
    labels = ["person", "bicycle", "car"]
    # class 1 (bicycle): one strong det + one below threshold; class 0/2 empty
    nms_output = [
        np.zeros((0, 5)),
        np.array([[0.2, 0.2, 0.6, 0.6, 0.9], [0.1, 0.1, 0.2, 0.2, 0.02]]),
        np.zeros((0, 5)),
    ]
    dets = parse_nms_output(nms_output, labels, tf, frame_w, frame_h, min_confidence=0.05)
    assert len(dets) == 1
    assert dets[0].class_name == "bicycle"
    assert dets[0].confidence == 0.9
    assert dets[0].w > 0 and dets[0].h > 0


def test_parse_nms_output_tolerates_empty_and_ragged():
    _, tf = letterbox(np.zeros((100, 100, 3), dtype=np.uint8), 640)
    nms_output = [None, [], np.array([[0.1, 0.1, 0.2]])]  # short row ignored
    dets = parse_nms_output(nms_output, ["a", "b", "c"], tf, 100, 100, min_confidence=0.0)
    assert dets == []


def test_build_detector_falls_back_to_null_without_hailo_type():
    detector = build_detector({"ai": {"type": "mock"}})
    assert isinstance(detector, NullDetector)
    assert detector.source == "null_disabled"


def test_build_detector_uses_simulated_ai_without_hailo_runtime():
    detector = build_detector({"ai": {"type": "simulated", "target_class": "bicycle"}})

    assert isinstance(detector, SimulatedDetector)
    assert detector.source == "sim_ai"
    dets = detector.infer(np.zeros((240, 320, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0].class_name == "bicycle"


def test_build_detector_reports_missing_hailo_model():
    detector = build_detector({"ai": {"type": "hailo", "model": "/nonexistent.hef"}})
    assert isinstance(detector, NullDetector)
    assert detector.source == "null_model"


def test_build_detector_uses_dev_hef_marker_without_hailo_runtime(tmp_path):
    model = tmp_path / "dev_cyclist.hef"
    model.write_bytes(DEV_HEF_MAGIC + b"test marker")

    detector = build_detector({"ai": {"type": "hailo", "model": str(model), "target_class": "bicycle"}})

    assert isinstance(detector, DevHefDetector)
    assert detector.source == "dev_hef"
    dets = detector.infer(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0].class_name == "bicycle"
    assert dets[0].confidence > 0.9


def test_is_dev_hef_requires_marker_prefix(tmp_path):
    dev = tmp_path / "dev.hef"
    realish = tmp_path / "real.hef"
    dev.write_bytes(DEV_HEF_MAGIC + b"payload")
    realish.write_bytes(b"not the marker")

    assert is_dev_hef(dev)
    assert not is_dev_hef(realish)


def test_resolve_hef_path_prefers_config_directory(tmp_path):
    model = tmp_path / "models" / "race.hef"
    model.parent.mkdir()
    model.write_bytes(b"hef")

    assert resolve_hef_path("models/race.hef", tmp_path) == str(model)
