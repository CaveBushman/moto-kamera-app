"""Tests for operator-confirmed tracking-box capture (training data for a
future custom cyclist Hailo model). Pure filesystem + timing logic --
no camera, no Hailo runtime needed."""
from __future__ import annotations

import time

import numpy as np
import pytest

from motocam.ai.training_capture import TrainingDataCapture


def _frame(width: int = 100, height: int = 80) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_no_bbox_never_captures(tmp_path):
    capture = TrainingDataCapture(output_dir=tmp_path, interval_s=0.0)
    assert capture.maybe_capture(_frame(), None) is False
    assert capture.captured_count == 0
    assert not list(tmp_path.glob("*.jpg"))


def test_first_capture_writes_image_and_yolo_label(tmp_path):
    capture = TrainingDataCapture(output_dir=tmp_path, interval_s=0.0, label="cyclist")

    captured = capture.maybe_capture(_frame(width=100, height=80), (10, 20, 40, 20))

    assert captured is True
    assert capture.captured_count == 1
    images = list(tmp_path.glob("*.jpg"))
    labels = list(tmp_path.glob("*.txt"))
    assert len(images) == 1
    # classes.txt also matches *.txt -- the real label file is the other one
    label_files = [p for p in labels if p.name != "classes.txt"]
    assert len(label_files) == 1

    # bbox (10, 20, 40, 20) in a 100x80 frame -> center (30, 30), size (40, 20)
    # normalized: cx=0.30 cy=0.375 w=0.40 h=0.25
    fields = label_files[0].read_text().split()
    assert fields[0] == "0"
    assert float(fields[1]) == pytest.approx(0.30, abs=1e-4)
    assert float(fields[2]) == pytest.approx(0.375, abs=1e-4)
    assert float(fields[3]) == pytest.approx(0.40, abs=1e-4)
    assert float(fields[4]) == pytest.approx(0.25, abs=1e-4)

    assert (tmp_path / "classes.txt").read_text().strip() == "cyclist"


def test_rate_limit_skips_captures_inside_the_interval(tmp_path):
    capture = TrainingDataCapture(output_dir=tmp_path, interval_s=10.0)

    assert capture.maybe_capture(_frame(), (0, 0, 10, 10)) is True
    assert capture.maybe_capture(_frame(), (0, 0, 10, 10)) is False  # too soon
    assert capture.captured_count == 1


def test_rate_limit_allows_capture_after_interval_elapses(tmp_path):
    # interval_s has a 0.1s safety floor (see TrainingDataCapture.__init__)
    capture = TrainingDataCapture(output_dir=tmp_path, interval_s=0.1)

    assert capture.maybe_capture(_frame(), (0, 0, 10, 10)) is True
    time.sleep(0.12)
    assert capture.maybe_capture(_frame(), (0, 0, 10, 10)) is True
    assert capture.captured_count == 2


def test_low_disk_space_pauses_capture(tmp_path, monkeypatch):
    capture = TrainingDataCapture(output_dir=tmp_path, interval_s=0.0, min_free_disk_mb=999_999_999)

    assert capture.maybe_capture(_frame(), (0, 0, 10, 10)) is False
    assert capture.captured_count == 0


def test_classes_file_written_once_not_per_capture(tmp_path):
    capture = TrainingDataCapture(output_dir=tmp_path, interval_s=0.0, label="cyclist")
    capture.maybe_capture(_frame(), (0, 0, 10, 10))
    (tmp_path / "classes.txt").write_text("cyclist\nextra-note\n")  # simulate manual edit

    capture.maybe_capture(_frame(), (0, 0, 10, 10))

    # must not have been clobbered back to just "cyclist\n" by the second capture
    assert "extra-note" in (tmp_path / "classes.txt").read_text()


def test_per_call_label_overrides_default_and_gets_its_own_class_index(tmp_path):
    # Race-start workflow: operator switches Target class to "peloton"
    # before tapping the group, "cyclist" stays the capture default.
    capture = TrainingDataCapture(output_dir=tmp_path, interval_s=0.0, label="cyclist")

    capture.maybe_capture(_frame(), (0, 0, 10, 10))  # default -> cyclist, index 0
    time.sleep(0.12)  # interval_s has a 0.1s safety floor even when 0.0 is passed
    capture.maybe_capture(_frame(), (0, 0, 10, 10), label="peloton")  # index 1

    assert (tmp_path / "classes.txt").read_text().splitlines() == ["cyclist", "peloton"]
    label_files = sorted(p for p in tmp_path.glob("*.txt") if p.name != "classes.txt")
    assert len(label_files) == 2
    assert label_files[0].read_text().split()[0] == "0"
    assert label_files[1].read_text().split()[0] == "1"


def test_repeated_label_reuses_same_class_index(tmp_path):
    capture = TrainingDataCapture(output_dir=tmp_path, interval_s=0.0, label="peloton")

    capture.maybe_capture(_frame(), (0, 0, 10, 10))
    time.sleep(0.12)  # interval_s has a 0.1s safety floor even when 0.0 is passed
    capture.maybe_capture(_frame(), (0, 0, 10, 10))

    label_files = [p for p in tmp_path.glob("*.txt") if p.name != "classes.txt"]
    assert {f.read_text().split()[0] for f in label_files} == {"0"}
    assert (tmp_path / "classes.txt").read_text().strip() == "peloton"
