"""
SORT: Simple Online and Realtime Tracking (Bewley et al., 2016).

Extended with a depth state for closing-velocity estimation via Kalman
filtering.  The state vector is:

    [cx, cy, area, aspect_ratio, depth, vx, vy, v_area, v_depth]

where depth is the metric forward distance (metres) and v_depth is its
rate of change (m/frame).  The Kalman filter jointly estimates bbox
motion **and** depth dynamics, yielding a principled velocity estimate
that is far smoother than post-hoc finite differences on noisy
monocular depth.

Reference
---------
Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft,
"Simple Online and Realtime Tracking", ICIP 2016.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity model on bbox + depth)
# ---------------------------------------------------------------------------

_DIM_X = 9   # state:       cx, cy, area, ar, depth, vx, vy, v_area, v_depth
_DIM_Z = 5   # observation: cx, cy, area, ar, depth


def _bbox_to_z(bbox: np.ndarray, depth: float = 0.0) -> np.ndarray:
    """Convert [x1, y1, x2, y2] + depth to [cx, cy, area, ar, depth]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    area = w * h
    ar = w / max(h, 1e-6)
    return np.array([cx, cy, area, ar, depth], dtype=np.float64)


def _z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert state[:5] → [x1, y1, x2, y2]."""
    w = np.sqrt(max(z[2] * z[3], 1e-6))
    h = max(z[2] / max(w, 1e-6), 1e-6)
    return np.array([
        z[0] - w / 2.0,
        z[1] - h / 2.0,
        z[0] + w / 2.0,
        z[1] + h / 2.0,
    ], dtype=np.float64)


class KalmanBoxTracker:
    """Per-object Kalman tracker.

    State: [cx, cy, s, r, d, vx, vy, vs, vd]  (9-dim)
    Obs:   [cx, cy, s, r, d]                   (5-dim)
    """

    _count = 0

    def __init__(self, bbox: np.ndarray, depth: float = 0.0) -> None:
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count

        # State and covariance
        self.x = np.zeros(_DIM_X, dtype=np.float64)
        self.x[:_DIM_Z] = _bbox_to_z(bbox, depth)
        self.P = np.eye(_DIM_X, dtype=np.float64) * 10.0
        self.P[5:, 5:] *= 1000.0  # high uncertainty on initial velocities

        # Transition: constant velocity
        self.F = np.eye(_DIM_X, dtype=np.float64)
        self.F[0, 5] = 1.0  # cx  += vx
        self.F[1, 6] = 1.0  # cy  += vy
        self.F[2, 7] = 1.0  # area += v_area
        self.F[4, 8] = 1.0  # depth += v_depth

        # Observation matrix: we observe [cx, cy, area, ar, depth]
        self.H = np.zeros((_DIM_Z, _DIM_X), dtype=np.float64)
        self.H[:_DIM_Z, :_DIM_Z] = np.eye(_DIM_Z)

        # Measurement noise
        self.R = np.diag([1.0, 1.0, 10.0, 0.01, 4.0])
        #                  cx   cy   area  ar    depth
        # depth noise ~4 m² reflects monocular depth uncertainty

        # Process noise
        self.Q = np.eye(_DIM_X, dtype=np.float64) * 0.01
        self.Q[5:, 5:] *= 0.01       # bbox velocity process noise
        self.Q[4, 4] = 0.5           # depth can change moderately
        self.Q[8, 8] = 0.1           # depth velocity changes slowly

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self) -> np.ndarray:
        """Advance state and return predicted bbox [x1,y1,x2,y2]."""
        if self.x[2] + self.x[7] <= 0:
            self.x[7] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_bbox(self.x[:_DIM_Z])

    def update(self, bbox: np.ndarray, depth: float = 0.0) -> None:
        """Update state with observed bbox [x1,y1,x2,y2] and depth."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        z = _bbox_to_z(bbox, depth)
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(_DIM_X) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Current bbox estimate [x1,y1,x2,y2]."""
        return _z_to_bbox(self.x[:_DIM_Z])

    @property
    def depth(self) -> float:
        """Kalman-filtered depth estimate (metres)."""
        return float(self.x[4])

    @property
    def v_depth(self) -> float:
        """Kalman-filtered depth velocity (m/frame). Negative = approaching."""
        return float(self.x[8])


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def _iou_batch(bb_det: np.ndarray, bb_trk: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bboxes.

    Parameters
    ----------
    bb_det : (D, 4) array of [x1, y1, x2, y2]
    bb_trk : (T, 4) array of [x1, y1, x2, y2]

    Returns
    -------
    (D, T) IoU matrix.
    """
    D = bb_det.shape[0]
    T = bb_trk.shape[0]
    if D == 0 or T == 0:
        return np.empty((D, T), dtype=np.float64)

    x1 = np.maximum(bb_det[:, None, 0], bb_trk[None, :, 0])
    y1 = np.maximum(bb_det[:, None, 1], bb_trk[None, :, 1])
    x2 = np.minimum(bb_det[:, None, 2], bb_trk[None, :, 2])
    y2 = np.minimum(bb_det[:, None, 3], bb_trk[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_d = (bb_det[:, 2] - bb_det[:, 0]) * (bb_det[:, 3] - bb_det[:, 1])
    area_t = (bb_trk[:, 2] - bb_trk[:, 0]) * (bb_trk[:, 3] - bb_trk[:, 1])
    union = area_d[:, None] + area_t[None, :] - inter
    return inter / np.maximum(union, 1e-10)


# ---------------------------------------------------------------------------
# SORT tracker
# ---------------------------------------------------------------------------

class Sort:
    """Multi-object tracker using SORT algorithm.

    Parameters
    ----------
    max_age : Maximum frames a track survives without a detection match.
    min_hits : Minimum consecutive hits before a track is reported.
    iou_threshold : Minimum IoU to accept a detection–track match.
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 2,
        iou_threshold: float = 0.20,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []

    def reset(self) -> None:
        """Clear all tracks (e.g. on sequence change)."""
        self.trackers.clear()
        KalmanBoxTracker._count = 0

    def update(
        self,
        detections: np.ndarray,
        depths: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, int, KalmanBoxTracker]]:
        """Run one frame of SORT.

        Parameters
        ----------
        detections : (N, 4) array of [x1, y1, x2, y2].
        depths     : (N,) array of metric depth per detection, or None.

        Returns
        -------
        List of (bbox_array[4], track_id, tracker) for confirmed tracks.
        """
        if detections.ndim == 1:
            detections = detections.reshape(-1, 4)
        dets = detections[:, :4]

        N = dets.shape[0]
        if depths is None:
            depths = np.zeros(N, dtype=np.float64)

        # --- Predict existing trackers ---
        trk_bboxes = np.zeros((len(self.trackers), 4), dtype=np.float64)
        to_delete: list[int] = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            trk_bboxes[i] = pred
            if np.any(np.isnan(pred)):
                to_delete.append(i)
        for i in reversed(to_delete):
            self.trackers.pop(i)
            trk_bboxes = np.delete(trk_bboxes, i, axis=0)

        # --- Associate detections to trackers via IoU + Hungarian ---
        matched, unmatched_dets, unmatched_trks = self._associate(
            dets, trk_bboxes
        )

        # Update matched trackers with assigned detections
        for d_idx, t_idx in matched:
            self.trackers[t_idx].update(dets[d_idx], float(depths[d_idx]))

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d_idx], float(depths[d_idx])))

        # Collect results (only tracks with enough hits or recently updated)
        results: list[tuple[np.ndarray, int, KalmanBoxTracker]] = []
        i = len(self.trackers) - 1
        while i >= 0:
            trk = self.trackers[i]
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
            elif trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or trk.age <= self.min_hits
            ):
                results.append((trk.get_state(), trk.id, trk))
            i -= 1

        return results

    def _associate(
        self,
        dets: np.ndarray,
        trks: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian matching on IoU cost matrix."""
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))

        iou_matrix = _iou_batch(dets, trks)
        row_idx, col_idx = linear_sum_assignment(-iou_matrix)

        matched: list[tuple[int, int]] = []
        unmatched_dets = set(range(len(dets)))
        unmatched_trks = set(range(len(trks)))

        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                unmatched_dets.discard(r)
                unmatched_trks.discard(c)

        return matched, sorted(unmatched_dets), sorted(unmatched_trks)
