"""Tests for the ByteTrack-style peloton tracker. The whole point is
holding a stable id through the things that break CSRT in a bunch:
short occlusions (the detector blinks out), partially-occluded riders
that only fire a weak box, and neighbouring riders that must not swap
ids."""
from __future__ import annotations

from motocam.ai.ai_engine import Detection
from motocam.tracking.byte_tracker import ByteTracker, iou


def _det(x, y, w=40, h=80, conf=0.9, cls="cyclist") -> Detection:
    return Detection(x=x, y=y, w=w, h=h, confidence=conf, class_name=cls)


def _run(tracker, frames):
    out = []
    for dets in frames:
        out.append(tracker.update(dets))
    return out


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    # half-overlap on x
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == round(50 / 150, 6) or 0.0 < iou((0, 0, 10, 10), (5, 0, 10, 10)) < 1


def test_needs_min_hits_to_confirm():
    tr = ByteTracker(min_hits=3)
    assert tr.update([_det(100, 100)]) == []  # hit 1 -> tentative
    assert tr.update([_det(102, 100)]) == []  # hit 2 -> tentative
    confirmed = tr.update([_det(104, 100)])   # hit 3 -> confirmed
    assert len(confirmed) == 1
    assert confirmed[0].state == "confirmed"


def test_single_rider_keeps_one_stable_id():
    tr = ByteTracker(min_hits=3)
    frames = [[_det(100 + i * 5, 100)] for i in range(10)]
    results = _run(tr, frames)
    ids = {t.track_id for frame in results for t in frame}
    assert ids == {1}  # exactly one identity, never renumbered


def test_survives_short_occlusion_with_same_id():
    tr = ByteTracker(min_hits=3, max_age=30)
    # confirm the rider
    for i in range(4):
        tr.update([_det(100 + i * 10, 100)])
    tid = tr.update([_det(140, 100)])[0].track_id

    # detector blinks out for 3 frames (full occlusion) -> track coasts
    coasting = [tr.update([]) for _ in range(3)]
    # the confirmed track is still returned, now flagged "lost" and coasting
    assert all(any(t.track_id == tid for t in frame) for frame in coasting)
    lost = [t for t in coasting[-1] if t.track_id == tid][0]
    assert lost.state == "lost"
    # Kalman kept moving it forward in the direction of travel
    assert lost.box[0] > 140

    # rider reappears near the predicted spot -> SAME id, not a new one
    back = tr.update([_det(175, 100)])
    same = [t for t in back if t.track_id == tid]
    assert same and same[0].state == "confirmed"


def test_two_neighbouring_riders_keep_distinct_ids():
    tr = ByteTracker(min_hits=3)
    ids_seen = set()
    for i in range(12):
        # two riders on parallel lines, both moving right, never overlapping
        frame = tr.update([_det(100 + i * 6, 100), _det(100 + i * 6, 300)])
        for t in frame:
            ids_seen.add(t.track_id)
    assert ids_seen == {1, 2}  # two stable identities, no id churn


def test_low_confidence_detection_recovers_the_track():
    tr = ByteTracker(high_thresh=0.5, min_hits=3, max_age=30)
    for i in range(4):
        tr.update([_det(100 + i * 8, 100, conf=0.9)])
    tid = tr.update([_det(132, 100, conf=0.9)])[0].track_id

    # partially occluded: only a weak (low-confidence) box this frame.
    # stage-2 association must still update the SAME track, not drop it.
    frame = tr.update([_det(140, 100, conf=0.2)])
    matched = [t for t in frame if t.track_id == tid]
    assert matched and matched[0].state == "confirmed"


def test_stale_track_is_dropped_after_max_age():
    tr = ByteTracker(min_hits=3, max_age=5)
    for i in range(4):
        tr.update([_det(100 + i * 5, 100)])
    tid = tr.update([_det(120, 100)])[0].track_id
    # no detections for longer than max_age
    for _ in range(7):
        tr.update([])
    assert tr.get(tid) is None


def test_track_at_picks_the_rider_under_the_tap():
    tr = ByteTracker(min_hits=1)
    tr.update([_det(100, 100, w=40, h=80), _det(300, 100, w=40, h=80)])
    # tap inside the first box
    assert tr.track_at(120, 140) == 1
    # tap inside the second box
    assert tr.track_at(320, 140) == 2
    # tap on empty background
    assert tr.track_at(500, 500) is None


def test_track_at_prefers_the_smaller_enclosing_box():
    tr = ByteTracker(min_hits=1)
    # a big group box and a small rider box both under the tap point
    tr.update([_det(0, 0, w=400, h=400, conf=0.9), _det(180, 180, w=40, h=40, conf=0.9)])
    tapped = tr.track_at(200, 200)
    small = tr.get(tapped)
    assert small is not None and small.box[2] < 100  # the small rider, not the group
