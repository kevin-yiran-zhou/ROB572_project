"""
tracking._pipeline — bridge between fusion.Obstacle and SORT tracker.

Responsibilities:
1. Convert dynamic (non-static) Obstacles to SORT detections.
2. Run SORT update to get persistent track IDs.
3. Read per-track closing velocity from the Kalman-filtered depth state.

The Kalman filter in sort.py jointly tracks bbox + depth, so velocity
estimation is a natural by-product of the filter — no separate
regression or EMA is needed.

Usage
-----
    tracker = ObstacleTracker()           # create once per sequence
    tracked = tracker.update(obstacles)   # call each frame
    for t in tracked:
        print(t.track_id, t.obstacle, t.v_closing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fusion.obstacle import Obstacle
from tracking.sort import KalmanBoxTracker, Sort


@dataclass
class TrackedObstacle:
    """An Obstacle enriched with tracking information."""

    obstacle: Obstacle
    track_id: int

    # Kalman-smoothed bbox [x1, y1, x2, y2] from SORT.  More stable than
    # the raw connected-component bbox in obstacle.  Use this for drawing.
    smoothed_bbox: tuple[int, int, int, int] | None = None

    # Closing velocity (m/s).  Positive = approaching, negative = receding.
    # None if not enough history to compute (first frame of a track).
    v_closing: Optional[float] = None

    # Time-to-collision (seconds).  None if v_closing <= 0 or unavailable.
    ttc: Optional[float] = None


class ObstacleTracker:
    """Wraps SORT and adds depth-based velocity estimation.

    Parameters
    ----------
    max_age : Frames a track survives without a match before deletion.
    min_hits : Consecutive detections before a track is reported.
    iou_threshold : Minimum IoU for detection-track association.
    fps : Frame rate used to convert per-frame Kalman v_depth to m/s.
    """

    def __init__(
        self,
        *,
        max_age: int = 5,
        min_hits: int = 2,
        iou_threshold: float = 0.20,
        fps: float = 10.0,
    ) -> None:
        self._sort = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
        )
        self._fps = fps
        # Per-track EMA of lateral_offset for smoothing risk score jitter.
        self._lateral_ema: dict[int, float] = {}

    def reset(self) -> None:
        """Clear all tracks (call on sequence change)."""
        self._sort.reset()
        self._lateral_ema.clear()

    def update(
        self,
        obstacles: list[Obstacle],
        *,
        dt: float | None = None,
    ) -> list[TrackedObstacle]:
        """Run one frame of tracking.

        Parameters
        ----------
        obstacles : All obstacles for this frame (both static and dynamic).
                    Only ``is_static=False`` obstacles are tracked.
        dt : Time delta since last frame in seconds.  If None, uses 1/fps.

        Returns
        -------
        List of TrackedObstacle for confirmed dynamic tracks.
        """
        if dt is None:
            dt = 1.0 / self._fps

        # Filter to dynamic obstacles only
        dynamic = [o for o in obstacles if not o.is_static]

        if not dynamic:
            self._sort.update(np.empty((0, 4), dtype=np.float64))
            return []

        # Build detection array [N, 4] and depth array [N] for SORT
        det_array = np.array(
            [[o.x1, o.y1, o.x2, o.y2] for o in dynamic],
            dtype=np.float64,
        )
        depth_array = np.array(
            [o.effective_depth for o in dynamic],
            dtype=np.float64,
        )

        # Run SORT (with depth)
        results = self._sort.update(det_array, depths=depth_array)

        # Match SORT output back to our Obstacle objects via IoU
        tracked: list[TrackedObstacle] = []
        _LATERAL_EMA_ALPHA = 0.4

        for bbox, track_id, trk in results:
            best_obs = self._match_obstacle(bbox, dynamic)
            if best_obs is None:
                continue

            # Smooth lateral_offset via per-track EMA
            raw_lat = best_obs.lateral_offset
            if track_id in self._lateral_ema:
                smoothed_lat = (
                    _LATERAL_EMA_ALPHA * raw_lat
                    + (1 - _LATERAL_EMA_ALPHA) * self._lateral_ema[track_id]
                )
            else:
                smoothed_lat = raw_lat
            self._lateral_ema[track_id] = smoothed_lat
            best_obs.lateral_offset = smoothed_lat

            # Read closing velocity from Kalman state.
            # v_depth is in m/frame (negative = depth decreasing = approaching).
            # Convert to m/s and flip sign so positive = approaching.
            v_closing: float | None = None
            if trk.hits >= 3:  # need a few observations before trusting velocity
                v_depth_per_frame = trk.v_depth        # m/frame
                v_depth_per_sec = v_depth_per_frame / dt  # m/s (if dt = 1/fps)
                v_closing = -v_depth_per_sec  # positive = approaching

            ttc = None
            if v_closing is not None and v_closing > 0.1:
                kalman_depth = max(trk.depth, 0.1)
                ttc = kalman_depth / v_closing

            # Kalman-smoothed bbox
            sb = (
                int(round(bbox[0])),
                int(round(bbox[1])),
                int(round(bbox[2])),
                int(round(bbox[3])),
            )

            tracked.append(
                TrackedObstacle(
                    obstacle=best_obs,
                    track_id=track_id,
                    smoothed_bbox=sb,
                    v_closing=v_closing,
                    ttc=ttc,
                )
            )

        # Prune lateral EMA for dead tracks
        active_ids = {t.track_id for t in tracked}
        active_ids.update(trk.id for trk in self._sort.trackers)
        dead = [k for k in self._lateral_ema if k not in active_ids]
        for k in dead:
            del self._lateral_ema[k]

        return tracked

    @staticmethod
    def _match_obstacle(
        bbox: np.ndarray, obstacles: list[Obstacle]
    ) -> Obstacle | None:
        """Find the obstacle with highest IoU to the given bbox."""
        best: Obstacle | None = None
        best_iou = -1.0
        bx1, by1, bx2, by2 = bbox
        for obs in obstacles:
            ix1 = max(bx1, obs.x1)
            iy1 = max(by1, obs.y1)
            ix2 = min(bx2, obs.x2)
            iy2 = min(by2, obs.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
            area_o = max(0.0, (obs.x2 - obs.x1) * (obs.y2 - obs.y1))
            union = area_b + area_o - inter
            iou = inter / max(union, 1e-10)
            if iou > best_iou:
                best_iou = iou
                best = obs
        return best
