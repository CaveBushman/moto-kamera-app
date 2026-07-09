"""Training data capture for a future custom cyclist Hailo model.

Rather than saving raw stock-model detections (bicycle/person, whatever
the interim COCO model happens to call a box -- see
docs/HAILO_CYCLIST_MODEL.md), this captures the *operator's own
confirmation*: whenever the tracker is LOCKED -- the rider tapped a
target (tap-to-select) or FULL AI auto-acquired one and it's still being
followed -- that box is a human-confirmed example, independent of
whatever class name the current interim model used to find it. Saved in
Ultralytics YOLO format (one image + one same-named .txt label file per
capture, normalized [x_center, y_center, w, h]) so it drops straight
into a YOLO fine-tuning run; the compiled result then installs via
scripts/prepare_cyclist_hef.sh.

Multiple classes (e.g. a single "cyclist" vs. a whole "peloton" at a
race start) share one output directory: `maybe_capture`'s `label`
argument picks the class for that capture, and `classes.txt` grows a new
line the first time a label is seen -- its line number is the class
index written into that capture's .txt, and existing lines (including
any the operator hand-edited) are never reordered or clobbered. The
caller (main_window) passes the operator's currently selected AI target
class, so switching "Target class" in Settings before tapping a target
is what decides which class a capture lands under.

Capturing every locked frame would flood the SD card with near-duplicate
frames (a ride is mostly small motion between consecutive frames) --
capture_interval_s spaces out captures so a long tracking session still
yields a diverse set. min_free_disk_mb is a hard stop so this can never
be the reason a unit runs out of disk mid-ride.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("motocam.ai.training_capture")

DEFAULT_INTERVAL_S = 2.0
DEFAULT_MIN_FREE_DISK_MB = 500
DEFAULT_LABEL = "cyclist"
JPEG_QUALITY = 90


class TrainingDataCapture:
    """Saves operator-confirmed tracking boxes as YOLO-format training
    examples. `maybe_capture` is cheap to call every frame -- it no-ops
    between captures and while disk space is low."""

    def __init__(
        self,
        output_dir: str | Path,
        interval_s: float = DEFAULT_INTERVAL_S,
        label: str = DEFAULT_LABEL,
        min_free_disk_mb: float = DEFAULT_MIN_FREE_DISK_MB,
    ):
        self.output_dir = Path(output_dir)
        self.interval_s = max(0.1, float(interval_s))
        self.label = label or DEFAULT_LABEL
        self.min_free_disk_mb = max(0.0, float(min_free_disk_mb))
        self._last_capture_at = 0.0
        self.captured_count = 0
        self._low_disk_logged = False

    def maybe_capture(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        label: str | None = None,
    ) -> bool:
        """Save `frame` + `bbox` as one training example if the rate
        limit and disk-space floor both allow it right now. `label`
        overrides the class for this capture (falls back to the
        instance default, e.g. "cyclist") -- pass the operator's live
        Target class so a peloton-mode capture lands under "peloton"
        instead. Returns whether a capture actually happened (tests /
        callers that want to know)."""
        if bbox is None:
            return False
        now = time.monotonic()
        if now - self._last_capture_at < self.interval_s:
            return False
        if not self._has_free_disk_space():
            return False
        self._last_capture_at = now
        self._write_example(frame, bbox, label or self.label)
        self.captured_count += 1
        if self.captured_count % 20 == 0:
            logger.info("Training capture: %d examples saved to %s", self.captured_count, self.output_dir)
        return True

    def _has_free_disk_space(self) -> bool:
        try:
            free_mb = shutil.disk_usage(self.output_dir if self.output_dir.exists() else self.output_dir.parent).free / (1024 * 1024)
        except OSError as exc:
            logger.debug("Training capture disk check failed: %s", exc)
            return True  # can't check -- degrade to "allow" rather than silently never capturing
        if free_mb < self.min_free_disk_mb:
            if not self._low_disk_logged:
                logger.warning(
                    "Training capture paused: only %.0f MB free (floor %.0f MB)", free_mb, self.min_free_disk_mb
                )
                self._low_disk_logged = True
            return False
        self._low_disk_logged = False
        return True

    def _write_example(self, frame: np.ndarray, bbox: tuple[int, int, int, int], label: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{int((time.time() % 1) * 1000):03d}"
        image_path = self.output_dir / f"{stamp}.jpg"
        label_path = self.output_dir / f"{stamp}.txt"
        # _class_index() does its own disk I/O (reads/writes classes.txt) and
        # must stay inside this try -- otherwise a transient disk error there
        # (SD card full/read-only) would raise straight out of maybe_capture
        # uncaught, aborting the caller's whole video frame instead of just
        # being logged and skipped like every other capture-write failure.
        try:
            class_index = self._class_index(label)
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            label_path.write_text(self._yolo_line(frame.shape, bbox, class_index) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("Training capture write failed: %s", exc)

    @staticmethod
    def _yolo_line(frame_shape: tuple[int, ...], bbox: tuple[int, int, int, int], class_index: int) -> str:
        height, width = frame_shape[:2]
        x, y, w, h = bbox
        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        return f"{class_index} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"

    def _class_index(self, label: str) -> int:
        """Line number of `label` in classes.txt, appending it as a new
        class the first time it's seen. Existing lines (including any
        the operator hand-edited in) keep their index and are never
        reordered, so an in-progress dataset stays consistent.

        Matches case-insensitively -- `label` comes from the operator's
        free-text-editable Target class field (see settings_dialog.py),
        so "peloton" typed once and "Peloton" typed another time must
        land in the same class, the same way ai_engine's own detection
        matching (main_window._pick_target_detection) already treats
        class names case-insensitively. The stored spelling is whichever
        casing was written first."""
        classes_path = self.output_dir / "classes.txt"
        labels: list[str] = []
        if classes_path.exists():
            labels = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        lowered = [existing.lower() for existing in labels]
        if label.lower() in lowered:
            return lowered.index(label.lower())
        labels.append(label)
        classes_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        return len(labels) - 1
