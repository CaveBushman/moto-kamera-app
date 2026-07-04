"""ByteTrack-style multi-object tracker for the peloton (design doc 8.3).

CSRT (tracking/tracker.py) tracks a single tapped box by appearance and
loses the rider the moment another cyclist crosses in front -- exactly
what happens constantly in a bunch. This is the detection-driven
alternative: it consumes the per-frame detections from the AI engine and
maintains *stable track IDs* for every rider, so the operator locks onto
a track id and the follow loop keeps that specific rider even when the
detector blinks out for a few frames (occlusion) -- a constant-velocity
Kalman filter coasts the box and re-associates the same id when the rider
reappears.

The design follows ByteTrack: predict every track with a Kalman filter,
then a two-stage association -- high-confidence detections first, then a
recovery pass against the *low*-confidence detections that a partially
occluded rider still produces. That low-score recovery pass is what keeps
identities alive through a bunch where a stock detector would otherwise
drop them.

Pure logic, no hardware and no heavy deps (greedy IoU association rather
than a Hungarian solver -- good enough for these box counts and keeps it
dependency-free and deterministic). Appearance re-ID (OSNet embeddings)
to survive a *full* cross-over is the documented next step; this motion +
two-stage-IoU core is already a large step up from CSRT for a bunch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motocam.ai.ai_engine import Detection

Box = tuple[float, float, float, float]  # (x, y, w, h)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """A tracked rider. `box` is the current Kalman estimate (predicted when
    this frame had no matching detection, so a follow loop can coast through
    a short occlusion). `state` is "confirmed" when matched this frame or
    "lost" while coasting."""

    track_id: int
    box: Box
    score: float
    state: str  # "tentative" | "confirmed" | "lost"
    hits: int
    time_since_update: int


class _KalmanFilter:
    """Minimal linear Kalman filter (predict/update), enough for the SORT
    constant-velocity box model without pulling in filterpy."""

    def __init__(self, dim_x: int, dim_z: int):
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        identity = np.eye(self.x.shape[0])
        self.P = (identity - K @ self.H) @ self.P


def _box_to_z(box: Box) -> np.ndarray:
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    s = w * h
    r = w / h if h > 1e-6 else 0.0
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def _z_to_box(x: np.ndarray) -> Box:
    flat = np.asarray(x).flatten()
    cx, cy, s, r = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = float(np.sqrt(s * r))
    h = float(s / w) if w > 1e-6 else 0.0
    return (cx - w / 2.0, cy - h / 2.0, w, h)


class _KalmanBoxTracker:
    """SORT constant-velocity box tracker: state [cx, cy, s, r, vcx, vcy, vs]."""

    def __init__(self, box: Box):
        self.kf = _KalmanFilter(dim_x=7, dim_z=4)
        # constant-velocity: cx += vcx, cy += vcy, s += vs
        self.kf.F = np.eye(7)
        self.kf.F[0, 4] = self.kf.F[1, 5] = self.kf.F[2, 6] = 1.0
        self.kf.H = np.zeros((4, 7))
        self.kf.H[0, 0] = self.kf.H[1, 1] = self.kf.H[2, 2] = self.kf.H[3, 3] = 1.0
        # SORT noise defaults
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # high uncertainty on unobserved velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _box_to_z(box)

    def predict(self) -> Box:
        # area velocity must not drive area negative
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return _z_to_box(self.kf.x)

    def update(self, box: Box) -> None:
        self.kf.update(_box_to_z(box))

    @property
    def box(self) -> Box:
        return _z_to_box(self.kf.x)


class _InternalTrack:
    def __init__(self, track_id: int, box: Box, score: float):
        self.track_id = track_id
        self.kf = _KalmanBoxTracker(box)
        self.score = score
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False

    def predict(self) -> None:
        self.kf.predict()
        self.time_since_update += 1

    def update(self, box: Box, score: float) -> None:
        self.kf.update(box)
        self.score = score
        self.hits += 1
        self.time_since_update = 0

    def as_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            box=self.kf.box,
            score=self.score,
            state="confirmed" if self.time_since_update == 0 else "lost",
            hits=self.hits,
            time_since_update=self.time_since_update,
        )


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.5,
        match_iou: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.high_thresh = high_thresh
        self.match_iou = match_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[_InternalTrack] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance one frame with the current detections; returns the alive,
        confirmed tracks (each with its current Kalman box estimate)."""
        for track in self._tracks:
            track.predict()

        highs = [d for d in detections if d.confidence >= self.high_thresh]
        lows = [d for d in detections if d.confidence < self.high_thresh]

        # Stage 1: all tracks vs high-confidence detections.
        matches, unmatched_tracks, unmatched_highs = self._associate(self._tracks, highs)
        for ti, di in matches:
            self._tracks[ti].update(_det_box(highs[di]), highs[di].confidence)

        # Stage 2: still-unmatched tracks vs low-confidence detections
        # (a partially occluded rider still fires a weak box -- this pass is
        # what keeps the id alive through a bunch).
        remaining = [self._tracks[ti] for ti in unmatched_tracks]
        matches2, unmatched_tracks2, _ = self._associate(remaining, lows)
        for ri, di in matches2:
            remaining[ri].update(_det_box(lows[di]), lows[di].confidence)

        # New tracks from leftover high-confidence detections.
        for di in unmatched_highs:
            det = highs[di]
            self._tracks.append(_InternalTrack(self._next_id, _det_box(det), det.confidence))
            self._next_id += 1

        # Confirm tracks with enough hits; drop stale ones.
        for track in self._tracks:
            if not track.confirmed and track.hits >= self.min_hits:
                track.confirmed = True
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        return [t.as_track() for t in self._tracks if t.confirmed]

    def track_at(self, x: float, y: float) -> int | None:
        """Track id whose current box contains (x, y) -- how the operator taps
        to lock onto a specific rider. Prefers the smallest containing box
        (the rider tapped, not a big group box behind them)."""
        best_id: int | None = None
        best_area = float("inf")
        for track in self._tracks:
            if not track.confirmed:
                continue
            bx, by, bw, bh = track.kf.box
            if bx <= x <= bx + bw and by <= y <= by + bh:
                area = bw * bh
                if area < best_area:
                    best_area = area
                    best_id = track.track_id
        return best_id

    def get(self, track_id: int) -> Track | None:
        for track in self._tracks:
            if track.track_id == track_id and track.confirmed:
                return track.as_track()
        return None

    def _associate(self, tracks, dets):
        """Greedy IoU association: repeatedly take the highest-IoU
        track/detection pair above threshold. Returns
        (matches, unmatched_track_indices, unmatched_det_indices)."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        pairs = []
        for ti, track in enumerate(tracks):
            for di, det in enumerate(dets):
                score = iou(track.kf.box, _det_box(det))
                if score >= self.match_iou:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_t: set[int] = set()
        matched_d: set[int] = set()
        matches = []
        for _score, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            matches.append((ti, di))

        unmatched_tracks = [ti for ti in range(len(tracks)) if ti not in matched_t]
        unmatched_dets = [di for di in range(len(dets)) if di not in matched_d]
        return matches, unmatched_tracks, unmatched_dets


def _det_box(det: Detection) -> Box:
    return (float(det.x), float(det.y), float(det.w), float(det.h))
